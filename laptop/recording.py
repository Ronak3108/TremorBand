import argparse
import csv
import time
from pathlib import Path

from .protocol import SAMPLE_RATE_HZ, SAMPLES_PER_PACKET, pack_packet, parse_packet
from .rolling_buffer import RollingBuffer
from .simulator import Simulator

HEADER = ["t_ms", "accel_g", "gyro_dps", "label"]
SAMPLE_MS = 1000 / SAMPLE_RATE_HZ  # 10 ms between samples


# ------------------------------------------------------------------
class Recorder:
    """Writes every packet that passes through tee() to a CSV file."""

    def __init__(self, path, label=""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.samples_written = 0
        self._t0 = None
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file, lineterminator="\n")
        self._writer.writerow(HEADER)
        self._file.flush()

    def tee(self, source):
        """Pass packets through unchanged, writing each one to the file first."""
        for packet in source:
            self._write(packet)
            yield packet

    def _write(self, packet):
        arrival = time.monotonic()
        _, samples = parse_packet(packet)
        if self._t0 is None:
            # make the very first sample t = 0
            self._t0 = arrival - (SAMPLES_PER_PACKET - 1) * SAMPLE_MS / 1000
        packet_end_ms = (arrival - self._t0) * 1000
        for i, (accel, gyro) in enumerate(samples):
            # the packet arrives when its last sample was taken; earlier ones are 10 ms apart
            t_ms = packet_end_ms - (SAMPLES_PER_PACKET - 1 - i) * SAMPLE_MS
            self._writer.writerow([f"{t_ms:.1f}", f"{accel:.3f}", f"{gyro:.1f}", self.label])
        self.samples_written += len(samples)
        self._file.flush()  # a crash or Ctrl-C loses at most one packet

    def close(self):
        if not self._file.closed:
            self._file.close()


# ------------------------------------------------------------------ replay
def load_rows(path):
    """Read a recording -> ([(t_ms, accel_g, gyro_dps), ...], label)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such recording: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"t_ms", "accel_g", "gyro_dps"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {sorted(missing)}")
        rows, label = [], ""
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append((float(row["t_ms"]), float(row["accel_g"]), float(row["gyro_dps"])))
            except ValueError:
                raise ValueError(f"{path.name}, line {line_no}: not a number: {row}") from None
            if not label:
                label = (row.get("label") or "").strip()
    if len(rows) < SAMPLES_PER_PACKET:
        raise ValueError(f"{path.name}: needs at least {SAMPLES_PER_PACKET} samples, has {len(rows)}")
    return rows, label


def replay_packets(path, loop=False, realtime=True):
    """Turn a recording back into packets, paced by the t_ms column.

    The file is read and checked right away (so a bad file fails before any thread
    starts); the returned generator then yields the same 18-byte packets the
    simulator or the band would. realtime=False skips the waiting (for verify).
    A trailing partial packet (rows not a multiple of 4) is dropped.
    """
    rows, _ = load_rows(path)
    n_packets = len(rows) // SAMPLES_PER_PACKET
    t_first = rows[0][0]

    def _packets():
        seq = 0
        while True:
            start = time.monotonic()
            for p in range(n_packets):
                chunk = rows[p * SAMPLES_PER_PACKET:(p + 1) * SAMPLES_PER_PACKET]
                if realtime:
                    due = start + (chunk[-1][0] - t_first) / 1000
                    wait = due - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                yield pack_packet(seq, [(a, g) for _, a, g in chunk])
                seq += SAMPLES_PER_PACKET  # pack_packet wraps it at 65536
            if not loop:
                return

    return _packets()


def csv_label(path):
    return load_rows(path)[1]


# ------------------------------------------------------------------ verify
def verify(path):
    """Replay a file at full speed and check it round-trips exactly."""
    rows, label = load_rows(path)
    n_packets = len(rows) // SAMPLES_PER_PACKET
    usable = n_packets * SAMPLES_PER_PACKET
    expected = [(a, g) for _, a, g in rows[:usable]]

    got, windows = [], 0
    buf = RollingBuffer()
    for packet in replay_packets(path, realtime=False):
        _, samples = parse_packet(packet)
        got.extend(samples)
        buf.push(samples)
        if buf.get_window() is not None:
            windows += 1

    t = [r[0] for r in rows]
    duration = (t[usable - 1] - t[0]) / 1000
    ends = [t[(p + 1) * SAMPLES_PER_PACKET - 1] for p in range(n_packets)]
    gaps = [ends[i + 1] - ends[i] for i in range(n_packets - 1)]

    print("|-----SUMMARY-----|")
    print(f"File: {Path(path).name} | label: {label or '(none)'}")
    print(f"Samples: {len(rows)} ({usable} used, {len(rows) - usable} dropped as a partial packet)")
    print(f"Duration: {duration:.2f} s  ({usable / duration:.1f} samples/s)" if duration else "Duration: 0 s")
    if gaps:
        print(f"Packet spacing: mean {sum(gaps) / len(gaps):.1f} ms, min {min(gaps):.1f}, max {max(gaps):.1f} (ideal 40)")
    print(f"Windows from the buffer: {windows} (expected {(usable - 256) // 32 + 1 if usable >= 256 else 0})")
    print(f"Round trip identical: {got == expected}")
    return got == expected


# ------------------------------------------------------------------ command line
def _record_cli(args):
    sim = Simulator(args.mode, seed=args.seed)
    rec = Recorder(args.path, args.label)
    t0 = time.monotonic()
    try:
        for _ in rec.tee(sim.packets()):
            if time.monotonic() - t0 >= args.seconds:
                break
    finally:
        rec.close()
    print(f"Recorded {rec.samples_written} samples ({args.mode}) to {rec.path}")


def _verify_cli(args):
    if not verify(args.path):
        raise SystemExit(1)  # non-zero exit so scripts can notice a bad file


def main():
    parser = argparse.ArgumentParser(description="Record / verify TremorBand CSV recordings")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="record the simulator (headless) to a CSV")
    p_rec.add_argument("path")
    p_rec.add_argument("-m", "--mode", default="TREMOR", help="IDLE, TREMOR or REACH")
    p_rec.add_argument("-s", "--seconds", type=float, default=20.0)
    p_rec.add_argument("-l", "--label", default="")
    p_rec.add_argument("--seed", type=int, default=None)
    p_rec.set_defaults(func=_record_cli)

    p_ver = sub.add_parser("verify", help="check that a recording replays exactly")
    p_ver.add_argument("path")
    p_ver.set_defaults(func=_verify_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()