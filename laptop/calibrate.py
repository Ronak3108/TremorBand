"""Derive detector thresholds from labelled recordings (task V5).

    python -m laptop.calibrate                      # analyse data/*.csv, print the report, change nothing
    python -m laptop.calibrate --write              # also save dsp/thresholds.json + docs/calibration_results.md
    python -m laptop.calibrate --data data/synthetic

Labels (column `label`, or the file name up to the first '_' / '-' / digit):
    rest     must NOT show tremor, must NOT trigger Action Mode
    tremor   must show tremor, must NOT trigger Action Mode
    reach    must trigger Action Mode
    typing, walking   reported only (a displayed 'resting tremor' there counts as a false alarm)
Record them with:  python -m laptop.recording record data/rest_ronak.csv --ble -l rest -s 45 --countdown 5
"""
import argparse
import csv
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from dsp import CONFIG, ActionDetector, compute_spectrum, detect_tremor
from dsp.tremor_dsp import _THRESHOLDS_FILE, gyro_level

WINDOW, HOP, TRIM_S = 256, 32, 1.0
NO_ACTION = ("rest", "tremor")
FALSE_ALARM_ONLY = ("rest", "typing", "walking", "reach")


def _label_of(path, row_label):
    if row_label:
        return row_label.strip().lower()
    return re.split(r"[_\-\d.]", Path(path).stem.lower())[0]


def load_clip(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty file")
    accel = np.array([float(r["accel_g"]) for r in rows])
    gyro = np.array([float(r["gyro_dps"]) for r in rows])
    t = np.array([float(r["t_ms"]) for r in rows])
    gaps = np.flatnonzero(np.diff(t) > 25)
    return _label_of(path, rows[0].get("label")), accel, gyro, len(gaps)


@contextmanager
def permissive_tremor():
    """Make detect_tremor report raw peak/ratio (thresholds off) while measuring features."""
    saved = (CONFIG["tremor_min_peak_g"], CONFIG["tremor_min_ratio"])
    CONFIG["tremor_min_peak_g"] = CONFIG["tremor_min_ratio"] = 0.0
    try:
        yield
    finally:
        CONFIG["tremor_min_peak_g"], CONFIG["tremor_min_ratio"] = saved


def clip_windows(accel, gyro):
    """Per 256-sample window: gyro level, in-band peak amplitude, band ratio, local-peak flag."""
    start = int(TRIM_S * 100)
    accel, gyro = accel[start:], gyro[start:]
    out = []
    with permissive_tremor():
        for end in range(WINDOW, len(accel) + 1, HOP):
            a, g = accel[end - WINDOW:end], gyro[end - WINDOW:end]
            f, m = compute_spectrum(a)
            d = detect_tremor(f, m)
            out.append((gyro_level(g), d["peak_mags"], d["band_ratio"], d["is_tremor"], d["peak_hz"]))
    return np.array(out, dtype=float)


def choose_action(levels):
    notes = []
    neg = np.concatenate([levels[c] for c in NO_ACTION if c in levels]) if any(c in levels for c in NO_ACTION) else None
    pos = levels.get("reach")
    if neg is None or pos is None or len(neg) == 0 or len(pos) == 0:
        return None, ["need rest, tremor AND reach clips to choose the Action thresholds"]
    hi_neg = float(np.percentile(neg, 99))
    lo_pos = float(np.percentile(pos, 20))
    if lo_pos <= hi_neg * 1.3:
        notes.append(f"OVERLAP: reach windows (20th pct {lo_pos:.0f} dps) are not clearly above rest/tremor windows "
                     f"(99th pct {hi_neg:.0f} dps). Gyro level alone cannot separate them - see docs/REVIEW.md "
                     "('if gyro overlaps'). Use bigger, more deliberate reaches or add a second feature.")
        on = hi_neg * 1.3
    else:
        on = float(np.sqrt(hi_neg * lo_pos))
    off = hi_neg + 0.4 * (on - hi_neg)
    return dict(gyro_on=round(on, 1), gyro_off=round(max(off, 2.0), 1)), notes


def choose_tremor(feats, gyro_on):
    pos = feats.get("tremor")
    if pos is None:
        return None, ["need tremor clips to choose the tremor thresholds"]
    neg_parts = []
    for c in FALSE_ALARM_ONLY:
        if c in feats:
            w = feats[c]
            neg_parts.append(w[w[:, 0] <= gyro_on])
    neg = np.vstack([p for p in neg_parts if len(p)]) if any(len(p) for p in neg_parts) else np.zeros((0, 5))
    pos = pos[pos[:, 3] > 0.5]
    if len(pos) == 0:
        return None, ["no tremor window has a peak inside 3-7 Hz - was the clip really ~5 Hz?"]

    amps = np.unique(np.round(np.concatenate([np.linspace(0.004, 0.05, 24), np.percentile(pos[:, 1], [2, 5, 10])]), 4))
    ratios = np.round(np.linspace(0.30, 0.90, 25), 3)
    best, best_score = [], -1
    for a in amps:
        for r in ratios:
            tpr = float(np.mean((pos[:, 1] >= a) & (pos[:, 2] >= r)))
            tnr = 1.0 if len(neg) == 0 else float(np.mean(~((neg[:, 3] > 0.5) & (neg[:, 1] >= a) & (neg[:, 2] >= r))))
            score = min(tpr, tnr)
            if score > best_score + 1e-9:
                best, best_score = [(a, r, tpr, tnr)], score
            elif abs(score - best_score) <= 1e-9:
                best.append((a, r, tpr, tnr))
    a, r, tpr, tnr = best[len(best) // 2]
    notes = []
    if best_score < 0.9:
        notes.append(f"Tremor vs non-tremor only reaches min(TPR,TNR)={best_score:.2f}: the classes overlap in "
                     "(amplitude, band-ratio). Check the tremor clips are steady ~5 Hz, or collect more data.")
    return dict(tremor_min_peak_g=float(a), tremor_min_ratio=float(r)), notes


def evaluate(per_clip, config):
    """Run the *stateful* detector exactly as the dashboard does, window by window, per clip."""
    res = {}
    for name, label, w in per_clip:
        det = ActionDetector(on=config["gyro_on"], off=config["gyro_off"])
        n = shown = act = 0
        for level, amp, ratio, local, hz in w:
            active = det.update(np.full(WINDOW, level))
            tremor = bool(local > 0.5 and amp >= config["tremor_min_peak_g"] and ratio >= config["tremor_min_ratio"])
            n += 1; act += active; shown += (tremor and not active)
        res[name] = (label, n, shown / n, act / n)
    return res


def verdict(label, shown_rate, action_rate):
    if label == "rest":
        ok = shown_rate < 0.05 and action_rate < 0.05
    elif label == "tremor":
        ok = shown_rate > 0.90 and action_rate < 0.05
    elif label == "reach":
        ok = action_rate > 0.70
    else:
        return "info" if shown_rate < 0.05 else "FALSE ALARM"
    return "ok" if ok else "WRONG"


def main():
    ap = argparse.ArgumentParser(description="Choose detector thresholds from labelled recordings")
    ap.add_argument("--data", default="data", help="folder of CSV recordings (default: data)")
    ap.add_argument("--write", action="store_true", help="save dsp/thresholds.json and the results table")
    ap.add_argument("--report", default="docs/calibration_results.md")
    args = ap.parse_args()

    files = sorted(Path(args.data).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {args.data}/ - record some first (see the module docstring).")

    per_clip, feats, levels, warnings = [], {}, {}, []
    for p in files:
        try:
            label, accel, gyro, n_gaps = load_clip(p)
        except (ValueError, KeyError) as e:
            print(f"skip {p.name}: {e}"); continue
        if len(accel) < (WINDOW + int(TRIM_S * 100)):
            print(f"skip {p.name}: shorter than {WINDOW / 100 + TRIM_S:.1f} s"); continue
        if n_gaps:
            warnings.append(f"{p.name}: {n_gaps} gaps in the timestamps (lost packets)")
        w = clip_windows(accel, gyro)
        per_clip.append((p.name, label, w))
        feats.setdefault(label, []).append(w)
    feats = {k: np.vstack(v) for k, v in feats.items()}
    levels = {k: v[:, 0] for k, v in feats.items()}

    print(f"\n{len(per_clip)} clips, labels: " + ", ".join(f"{k} ({len(v)} windows)" for k, v in feats.items()))
    print("\nFeature summary per label (percentiles 5 / 50 / 95):")
    print(f"  {'label':8s} {'gyro level (dps)':>22s} {'tremor-band amp (g)':>24s} {'band ratio':>20s}")
    for k, v in feats.items():
        f = lambda col, fmt: " / ".join(format(x, fmt) for x in np.percentile(v[:, col], [5, 50, 95]))
        print(f"  {k:8s} {f(0, '.1f'):>22s} {f(1, '.3f'):>24s} {f(2, '.2f'):>20s}")

    chosen = dict()
    act, act_notes = choose_action(levels)
    if act:
        chosen.update(act)
    gyro_on = chosen.get("gyro_on", CONFIG["gyro_on"])
    trem, trem_notes = choose_tremor(feats, gyro_on)
    if trem:
        chosen.update(trem)
    notes = warnings + act_notes + trem_notes
    final = {**CONFIG, **chosen}

    print("\nChosen thresholds:" if chosen else "\nNo thresholds chosen.")
    for k, v in chosen.items():
        print(f"  {k:20s} {v}   (was {CONFIG[k]})")
    for n in notes:
        print(f"  ! {n}")

    results = evaluate(per_clip, final)
    print("\nDetection results per clip, with the chosen thresholds (stateful, as on the dashboard):")
    print(f"  {'file':22s} {'label':8s} {'windows':>7s} {'tremor shown':>13s} {'action on':>10s}   verdict")
    rows_md = []
    for name, (label, n, shown, actv) in results.items():
        v = verdict(label, shown, actv)
        print(f"  {name:22s} {label:8s} {n:7d} {shown:12.0%} {actv:10.0%}   {v}")
        rows_md.append(f"| {name} | {label} | {n} | {shown:.0%} | {actv:.0%} | {v} |")

    if args.write and chosen:
        _THRESHOLDS_FILE.write_text(json.dumps(
            {"config": {k: chosen[k] for k in chosen}, "source": str(args.data), "clips": len(per_clip)}, indent=2) + "\n")
        print(f"\nWrote {_THRESHOLDS_FILE}")
        rp = Path(args.report); rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text("# Calibration results\n\nSource: `%s` (%d clips)\n\n## Chosen thresholds\n\n%s\n\n%s\n\n"
                      "## Per-clip detection results\n\n| file | label | windows | tremor shown | action on | verdict |\n|---|---|---|---|---|---|\n%s\n"
                      % (args.data, len(per_clip),
                         "\n".join(f"- `{k}` = **{v}**" for k, v in chosen.items()),
                         ("**Notes:**\n\n" + "\n".join(f"- {n}" for n in notes)) if notes else "",
                         "\n".join(rows_md)))
        print(f"Wrote {rp}")
    elif not args.write:
        print("\n(dry run: nothing saved. Add --write to save dsp/thresholds.json and the results table.)")


if __name__ == "__main__":
    main()
