import numpy as np
import argparse
from time import monotonic, sleep
from .protocol import SAMPLES_PER_PACKET, PERIOD, SAMPLE_RATE_HZ, SEQ_MOD, pack_packet, parse_packet

MODES = {
    "IDLE":   {"accel_amp": 0.0,  "accel_freq": 0.0, "accel_noise": 0.01,
               "gyro_base": 2.0,  "gyro_amp": 0.0,   "gyro_freq": 0.0, "gyro_noise": 1.0},
    "TREMOR": {"accel_amp": 0.08, "accel_freq": 5.0, "accel_noise": 0.01,
               "gyro_base": 15.0, "gyro_amp": 10.0,  "gyro_freq": 5.0, "gyro_noise": 2.0},
    "REACH": {"accel_amp": 0.4,  "accel_freq": 1.0, "accel_noise": 0.02,
               "gyro_base": 120.0, "gyro_amp": 80.0, "gyro_freq": 1.0, "gyro_noise": 8.0},
}

def make_samples(mode: str, n0: int, rng) -> list[tuple[float, float]]:

    accel_amp = MODES[mode]["accel_amp"]
    accel_freq = MODES[mode]['accel_freq']
    accel_noise = MODES[mode]["accel_noise"]
    pi = np.pi
    gyro_base = MODES[mode]["gyro_base"]
    gyro_amp = MODES[mode]['gyro_amp']
    gyro_freq = MODES[mode]['gyro_freq']
    gyro_noise = MODES[mode]['gyro_noise']

    samples = []
    for i in range(SAMPLES_PER_PACKET):
        acc_g = 1 + accel_amp * (np.sin(2 * pi * accel_freq * (n0 + i) / SAMPLE_RATE_HZ)) + rng.normal(0, accel_noise)
        gyro_dps = max(0.0, gyro_base + gyro_amp * (np.sin(2 * pi * gyro_freq * (n0 + i) / SAMPLE_RATE_HZ)) + rng.normal(0, gyro_noise))

        samples.append((acc_g, gyro_dps))

    return samples

class Simulator:
    def __init__(self, mode: str = 'IDLE', seed: int | None = None, drop_prob: float = 0.0):
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be between 0 and 1, got {drop_prob}")
        self.mode = mode
        self.drop_prob = drop_prob
        self.rng = np.random.default_rng(seed)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        name = str(value).upper()
        if name not in MODES:
            raise ValueError(f"Invalid mode {value!r}. Valid modes: {list(MODES)}")
        self._mode = name

    def packets(self):
        packet_count = 0
        t0 = monotonic()

        while True:
            n0 = packet_count * SAMPLES_PER_PACKET
            samples = make_samples(self.mode, n0, self.rng)
            packet = pack_packet(n0, samples)
            roll = self.rng.random()
            if roll >= self.drop_prob:
                yield packet
            packet_count += 1

            deadline = t0 + packet_count * PERIOD
            remaining = deadline - monotonic()
            if remaining > 0:
                sleep(remaining)


def main():
    parser = argparse.ArgumentParser("A script to create a fake sample data for the project until we move on to real data stream from the ESP32.")
    parser.add_argument('-m', "--mode", type=str, default="TREMOR", help="'mode' specifies the type of fake data the sample creator should spew. There are three modes: IDLE, TREMOR, REACH")
    parser.add_argument('-sd', "--seed", type=int, help="'seed' refers to the seed for the random number generator.")
    parser.add_argument('-dp', "--drop_prob", type=float, help="'drop_prob' refers to the sample drop probability.")
    parser.add_argument('-s', "--seconds", type=float, default=10.0, help="Specifies the amount of time for which simulation will run.")

    args = parser.parse_args()

    if args.drop_prob:
        sim = Simulator(args.mode, args.seed, args.drop_prob)
    else:
        sim = Simulator(args.mode, args.seed)

    start_time = monotonic()
    prev_seq = -SAMPLES_PER_PACKET
    prev_time = None
    max_diff_seq = 0
    packet_count = 0
    lost_count = 0
    spacings_ms = []

    for packet in sim.packets():
        current_time = monotonic()
        if current_time - start_time > args.seconds:
            break
        else:
            packet_count += 1
            data = parse_packet(packet)
            seq = data[0]
            diff = (seq - prev_seq) % SEQ_MOD
            max_diff_seq = max(max_diff_seq, diff)
            lost_count += diff // SAMPLES_PER_PACKET - 1
            if diff == SAMPLES_PER_PACKET and prev_time is not None:
                spacings_ms.append((current_time - prev_time) * 1000)
            prev_seq = seq
            prev_time = current_time

    elapsed = monotonic() - start_time
    total_laps = packet_count + lost_count

    print("|-----SUMMARY-----|")
    print(f"Mode: {sim.mode} | seed: {args.seed} | drop_prob: {sim.drop_prob}")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Packet Count: {packet_count}  ({packet_count / elapsed:.2f} per second; {1 / PERIOD:.0f} when nothing is dropped)")
    if total_laps:
        print(f"Packets lost (from seq gaps): {lost_count}  ({100 * lost_count / total_laps:.1f}% of {total_laps})")
    print(f"Largest Sequence Jump: {max_diff_seq}  (4 = nothing lost)")
    if spacings_ms:
        print(f"Back-to-back spacing: mean {np.mean(spacings_ms):.3f} ms, std {np.std(spacings_ms):.3f} ms, "
              f"max {np.max(spacings_ms):.2f} ms (ideal {PERIOD * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
