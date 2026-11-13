import threading
import numpy as np
import matplotlib.pyplot as plt
import time

from rolling_buffer import RollingBuffer
from pipeline import reader
from simulator import Simulator
from stubs import compute_spectrum, detect_tremor, detect_action


def main():
    rb = RollingBuffer()
    stop = threading.Event()
    errors = []
    sim = Simulator('REACH', seed=1, drop_prob = 0)
    buffer_thread = threading.Thread(target=reader, kwargs={'source': sim.packets(), 'buf': rb, 'stop': stop, 'errors': errors}, daemon=True)

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1)

    acc_low = 0.5
    acc_high = 1.5

    x_wave = np.linspace(0, 2.56, 256)
    y_wave = np.zeros_like(x_wave)
    line_wave, = ax_wave.plot(x_wave, y_wave)
    ax_wave.set_ylim(acc_low, acc_high)

    freqs, mags = compute_spectrum(np.zeros(256))

    bars = ax_spec.bar(freqs, np.zeros_like(freqs), width=0.3)
    ax_spec.set_xlim(0, 15)
    ax_spec.set_ylim(0, 13)

    banner = fig.suptitle("Waiting for data...")      # mode text at the top of the figure
    peak_idx = None                                   # index of the bar currently highlighted
    update_times = []                                 # seconds each update() took
    frames = 0                                        # windows drawn



    def update():
        nonlocal peak_idx, frames                     # NEW: so update can change these

        if errors:
            banner.set_text(f"Reader thread crashed: {errors[0]!r}")
            banner.set_color("red")
            fig.canvas.draw_idle()
            timer.stop()
            return
        
        window = rb.get_window()
        if window is None:
            return

        t_start = time.monotonic()

        accel, gyro = window
        line_wave.set_ydata(accel)

        freqs, mags = compute_spectrum(accel)

        for bar, m in zip(bars, mags):
            bar.set_height(m)

        tremor = detect_tremor(freqs, mags)
        action = detect_action(gyro)
 
        if peak_idx is not None:                      # un-colour the previous highlight
            bars[peak_idx].set_color("C0")
            peak_idx = None
 
        if action:
            peak_idx = int(np.argmin(np.abs(freqs - tremor["peak_hz"])))
            bars[peak_idx].set_color("lightgray")     # dimmed peak
            banner.set_text("Action Mode - Resting Tremor Suppressed")
            banner.set_color("tab:orange")
        elif tremor["is_tremor"]:
            peak_idx = int(np.argmin(np.abs(freqs - tremor["peak_hz"])))
            bars[peak_idx].set_color("red")
            banner.set_text(f"Resting tremor detected: {tremor['peak_hz']:.1f} Hz")
            banner.set_color("red")
        else:
            banner.set_text("Normal / rest")
            banner.set_color("tab:green")


        fig.canvas.draw_idle()

    buffer_thread.start()

    timer = fig.canvas.new_timer(interval=40)
    timer.add_callback(update)
    timer.start()

    plt.show()


if __name__ == "__main__":
    main()