import numpy as np
import threading
from time import monotonic, sleep

from .rolling_buffer import RollingBuffer
from .protocol import parse_packet
from .simulator import Simulator

def reader(source, buf: RollingBuffer, stop, errors):
    try:
        for packet in source:
            if stop.is_set():
                break

            seq, samples = parse_packet(packet)
            buf.push(samples)

    except Exception as e:
        errors.append(e)


def main():
    rb = RollingBuffer()
    stop = threading.Event()
    errors = []
    sim = Simulator('TREMOR', seed=1, drop_prob = 0)

    buffer_thread = threading.Thread(target=reader, kwargs={'source': sim.packets(), 'buf': rb, 'stop': stop, 'errors': errors}, daemon=True)
    buffer_thread.start()

    times = []   # seconds since the loop started, one per window
    peaks = []   # peak frequency (Hz) of each window

    t0 = monotonic()
    while monotonic() - t0 < 20:
        window = rb.get_window()
        if window is not None:
            accel, gyro = window
            times.append(monotonic() - t0)

            x = accel - accel.mean()              # remove gravity
            spectrum = np.abs(np.fft.rfft(x))     # complex -> magnitudes
            k = np.argmax(spectrum[1:]) + 1       # skip bin 0, then undo the shift
            peaks.append(k * 100 / 256)           # bin -> Hz
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

        
