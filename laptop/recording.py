import argparse
import csv
import time
from pathlib import Path

from .protocol import SAMPLE_RATE_HZ, SAMPLES_PER_PACKET, SEQ_MOD, pack_packet, parse_packet
from .rolling_buffer import RollingBuffer
from .simulator import Simulator

HEADER = ["t_ms", "accel_g", "gyro_dps", "label"]
SAMPLE_MS = 1000 / SAMPLE_RATE_HZ


class Recorder:
    """Writes every packet that passes through tee() to a CSV file."""

    def __init__(self, path, label=""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.samples_written = 0
        self._last_seq = None
        self._index = 0
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
        """Timestamp each sample from the packet's sequence number (10 ms per sample).

        A repeated or backwards seq (the band restarted) continues the timeline after the previous packet.
        """
        seq, samples = parse_packet(packet)
        if self._last_seq is not None:
            step = (seq - self._last_seq) % SEQ_MOD
            if step == 0 or step > SEQ_MOD // 2:
                step = SAMPLES_PER_PACKET
            self._index += step
        self._last_seq = seq
        for i, (accel, gyro) in enumerate(samples):
            t_ms = (self._index + i) * SAMPLE_MS
            self._writer.writerow([f"{t_ms:.1f}", f"{accel:.3f}", f"{gyro:.1f}", self.label])
        self.samples_written += len(samples)
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()


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
    """Turn a recording back into 18-byte packets, paced by the t_ms column (realtime=False skips the pacing).

    The file is validated immediately; a trailing partial packet is dropped.
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
                seq += SAMPLES_PER_PACKET
            if not loop:
                return

    return _packets()


def csv_label(path):
    return load_rows(path)[1]


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


def _record_cli(args):
    receiver = None
    if args.ble:
        from .ble_receiver import BleReceiver
        receiver = BleReceiver()
        receiver.start()
        what = "TremorBand"
        print("Waiting for the band...", end="", flush=True)
        while not receiver.connected:
            time.sleep(0.2)
        print(" connected.")
    else:
        sim = Simulator(args.mode, seed=args.seed)
        what = f"simulator {args.mode}"
    if args.countdown:
        for n in range(args.countdown, 0, -1):
            print(f"  starting in {n}...", end="\r", flush=True)
            time.sleep(1)
    if receiver:
        receiver.flush()
        source = receiver.packets(duration=args.seconds)
    else:
        source = sim.packets()
    rec = Recorder(args.path, args.label)
    print(f"RECORDING '{args.label or '(no label)'}' for {args.seconds:.0f} s ...")
    t0 = time.monotonic()
    try:
        for _ in rec.tee(source):
            if time.monotonic() - t0 >= args.seconds:
                break
    finally:
        rec.close()
        if receiver:
            receiver.close()
    print(f"Recorded {rec.samples_written} samples ({what}) to {rec.path}")
    if receiver:
        link = receiver.stats()
        print(f"Link: lost {link['lost_packets']} packets ({link['lost_percent']:.2f}%)")


def _verify_cli(args):
    if not verify(args.path):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Record / verify TremorBand CSV recordings")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="record the simulator (or, with --ble, the real band) headless to a CSV")
    p_rec.add_argument("path")
    p_rec.add_argument("-m", "--mode", default="TREMOR", help="IDLE, TREMOR or REACH")
    p_rec.add_argument("-s", "--seconds", type=float, default=20.0)
    p_rec.add_argument("-l", "--label", default="")
    p_rec.add_argument("--seed", type=int, default=None)
    p_rec.add_argument("--ble", action="store_true", help="record the real TremorBand instead of the simulator")
    p_rec.add_argument("--countdown", type=int, default=0, help="seconds to wait after connecting (time to get your hand in position)")
    p_rec.set_defaults(func=_record_cli)

    p_ver = sub.add_parser("verify", help="check that a recording replays exactly")
    p_ver.add_argument("path")
    p_ver.set_defaults(func=_verify_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
