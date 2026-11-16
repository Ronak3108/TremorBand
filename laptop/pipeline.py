import numpy as np
import threading
from time import monotonic, sleep

from .rolling_buffer import RollingBuffer
from .protocol import SAMPLES_PER_PACKET, SEQ_MOD, parse_packet
from .simulator import Simulator

MAX_FILL_SAMPLES = 100


def reader(source, buf: RollingBuffer, stop, errors):
    """Feed packets into the buffer, interpolating short gaps and clearing it after a restart or long outage."""
    last_seq = None
    last_sample = None
    try:
        for packet in source:
            if stop.is_set():
                break

            seq, samples = parse_packet(packet)
            if last_seq is not None:
                step = (seq - last_seq) % SEQ_MOD
                missing = step - SAMPLES_PER_PACKET
                if step == 0 or step > SEQ_MOD // 2:
                    buf.clear()  # duplicate or backwards seq: the band restarted
                elif missing > MAX_FILL_SAMPLES:
                    buf.clear()  # outage too long to bridge
                elif missing > 0 and last_sample is not None:
                    # linearly interpolate the lost samples
                    a0, g0 = last_sample
                    a1, g1 = samples[0]
                    t = np.arange(1, missing + 1) / (missing + 1)
                    buf.push([(a0 + (a1 - a0) * f, g0 + (g1 - g0) * f) for f in t])
            buf.push(samples)
            last_seq, last_sample = seq, samples[-1]

    except Exception as e:
        errors.append(e)


def main():
    rb = RollingBuffer()
    stop = threading.Event()
    errors = []
    sim = Simulator('TREMOR', seed=1, drop_prob = 0)

    buffer_thread = threading.Thread(target=reader, kwargs={'source': sim.packets(), 'buf': rb, 'stop': stop, 'errors': errors}, daemon=True)
    buffer_thread.start()

    times = []
    peaks = []

    t0 = monotonic()
    while monotonic() - t0 < 20:
        window = rb.get_window()
        if window is not None:
            accel, gyro = window
            times.append(monotonic() - t0)

            x = accel - accel.mean()
            spectrum = np.abs(np.fft.rfft(x))
            k = np.argmax(spectrum[1:]) + 1
            peaks.append(k * 100 / 256)
        sleep(0.02)

    stop.set()
    t_stop = monotonic()
    buffer_thread.join(timeout=1)
    stop_delay_ms = (monotonic() - t_stop) * 1000

    gaps = np.diff(times)

    print("|-----SUMMARY-----|")
    print(f"Windows in 20 s: {len(times)}")
    print(f"First window at: {times[0]:.2f} s")
    print(f"Spacing: mean {gaps.mean():.3f} s, min {gaps.min():.3f} s, max {gaps.max():.3f} s")
    print(f"Peak frequencies seen: {sorted(set(round(float(p), 2) for p in peaks))}")
    print(f"Thread alive after stop: {buffer_thread.is_alive()} (ended in {stop_delay_ms:.0f} ms)")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
