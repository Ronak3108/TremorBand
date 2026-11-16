import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

from .rolling_buffer import RollingBuffer
from .pipeline import reader
from .simulator import Simulator
from .recording import Recorder, csv_label, replay_packets
from .protocol import SAMPLES_PER_PACKET
from dsp import CONFIG, ActionDetector, compute_spectrum, detect_tremor


def main():
    parser = argparse.ArgumentParser(description="Live tremor dashboard")
    parser.add_argument("-m", "--mode", default="TREMOR", help="simulator mode: IDLE, TREMOR or REACH")
    parser.add_argument("--ble", action="store_true", help="read the real TremorBand over Bluetooth")
    parser.add_argument("--replay", metavar="CSV", help="replay a recording instead of the simulator")
    parser.add_argument("--loop", action="store_true", help="with --replay: start over when the file ends")
    parser.add_argument("--record", metavar="CSV", help="also save the live stream to this CSV")
    parser.add_argument("-l", "--label", default="", help="label written into the --record file")
    args = parser.parse_args()
    if args.replay and args.record:
        parser.error("--record and --replay cannot be used together")
    if args.ble and args.replay:
        parser.error("--ble and --replay cannot be used together")
    if args.loop and not args.replay:
        parser.error("--loop only makes sense with --replay")

    rb = RollingBuffer()
    stop = threading.Event()
    errors = []

    receiver = None
    sim = None
    if args.ble:
        from .ble_receiver import BleReceiver, DEVICE_NAME
        receiver = BleReceiver()
        receiver.start()
        source = receiver.packets()
        source_name = f"Bluetooth: {DEVICE_NAME}"
    elif args.replay:
        try:
            source = replay_packets(args.replay, loop=args.loop)
            label = csv_label(args.replay)
        except (OSError, ValueError) as e:
            parser.error(str(e))
        source_name = f"replay: {Path(args.replay).name}"
        if label:
            source_name += f" ({label})"
    else:
        sim = Simulator(args.mode, seed=1, drop_prob = 0)
        source = sim.packets()
        source_name = f"simulator: {sim.mode}"

    recorder = None
    if args.record:
        recorder = Recorder(args.record, args.label)
        source = recorder.tee(source)
        source_name += f"  |  recording to {args.record}"

    buffer_thread = threading.Thread(target=reader, kwargs={'source': source, 'buf': rb, 'stop': stop, 'errors': errors}, daemon=True)

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(9, 6))

    wave_half = 0.25
    spec_high = 0.05

    x_wave = np.linspace(0, 2.56, 256)
    y_wave = np.zeros_like(x_wave)
    line_wave, = ax_wave.plot(x_wave, y_wave)
    ax_wave.set_ylim(1 - wave_half, 1 + wave_half)
    ax_wave.set_xlim(0, 2.56)
    ax_wave.set_xlabel("Time in window (s)")
    ax_wave.set_ylabel("Accel (g)")
    ax_wave.set_title(source_name, fontsize=9)

    freqs, mags = compute_spectrum(np.zeros(256))

    bars = ax_spec.bar(freqs, np.zeros_like(freqs), width=0.3)
    ax_spec.set_xlim(0, 15)
    ax_spec.set_ylim(0, spec_high)
    ax_spec.axvspan(*CONFIG["tremor_band_hz"], color="red", alpha=0.06)
    ax_spec.axhline(CONFIG["tremor_min_peak_g"], color="gray", ls=":", lw=1)
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Amplitude (g)")

    banner = fig.suptitle("Waiting for data...")
    readout = ax_spec.text(0.99, 0.95, "", transform=ax_spec.transAxes, ha="right", va="top", fontsize=8,
                           family="monospace")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    peak_idx = None
    update_times = []
    frames = 0
    action_detector = ActionDetector()
    last_status = None
    refill_packets = rb.window // SAMPLES_PER_PACKET

    def update():
        nonlocal peak_idx, frames, last_status

        if errors:
            banner.set_text(f"Reader thread crashed: {errors[0]!r}")
            banner.set_color("red")
            fig.canvas.draw_idle()
            timer.stop()
            return

        if receiver is not None:
            status = receiver.status
            if status != last_status:
                last_status = status
                connected = receiver.connected
                banner.set_text(status + (" - filling buffer..." if connected else ""))
                banner.set_color("red" if status.startswith("Bluetooth error") else "gray")
                fig.canvas.draw_idle()
            if not receiver.connected or receiver.delivered_this_connection < refill_packets:
                return

        window = rb.get_window()
        if window is None:
            if not buffer_thread.is_alive():
                banner.set_text("Replay finished")
                banner.set_color("gray")
                fig.canvas.draw_idle()
                timer.stop()
            return

        t_start = time.monotonic()

        accel, gyro = window
        line_wave.set_ydata(accel)
        half = max(wave_half, 1.15 * float(np.max(np.abs(accel - 1.0))))
        ax_wave.set_ylim(1 - half, 1 + half)

        freqs, mags = compute_spectrum(accel)

        for bar, m in zip(bars, mags):
            bar.set_height(m)
        ax_spec.set_ylim(0, max(spec_high, 1.2 * float(mags.max())))

        tremor = detect_tremor(freqs, mags)
        action = action_detector.update(gyro)
        text = (f"peak {tremor['peak_hz']:.2f} Hz  amp {tremor['peak_mags']:.3f} g  "
                f"ratio {tremor['band_ratio']:.2f}\ngyro {action_detector.level:.0f} dps "
                f"(on>{action_detector.on:.0f} off<{action_detector.off:.0f})")
        if receiver is not None:
            text += f"\nlost {receiver.packets_lost} pkts ({receiver.stats()['lost_percent']:.1f}%)"
        readout.set_text(text)

        if peak_idx is not None:
            bars[peak_idx].set_color("C0")
            peak_idx = None

        if action:
            peak_idx = int(np.argmin(np.abs(freqs - tremor["peak_hz"])))
            bars[peak_idx].set_color("lightgray")
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

        frames += 1
        update_times.append(time.monotonic() - t_start)

    if sim is not None:
        def on_key(event):
            modes = {"i": "IDLE", "t": "TREMOR", "r": "REACH", "1": "IDLE", "2": "TREMOR", "3": "REACH"}
            if event.key in modes:
                sim.mode = modes[event.key]
                ax_wave.set_title(f"simulator: {sim.mode}   (keys: i = idle, t = tremor, r = reach)", fontsize=9)
        fig.canvas.mpl_connect("key_press_event", on_key)
        ax_wave.set_title(f"simulator: {sim.mode}   (keys: i = idle, t = tremor, r = reach)", fontsize=9)

    buffer_thread.start()

    timer = fig.canvas.new_timer(interval=40)
    timer.add_callback(update)
    timer.start()

    plt.show()

    stop.set()
    if receiver:
        receiver.close()
    buffer_thread.join(timeout=1)
    if recorder:
        recorder.close()
        print(f"Recorded {recorder.samples_written} samples to {recorder.path}")
    if update_times:
        ms = np.array(update_times) * 1000
        print(f"Windows drawn: {frames}")
        print(f"update() time: mean {ms.mean():.1f} ms, max {ms.max():.1f} ms")
    if receiver:
        link = receiver.stats()
        print(f"Link: {link['delivered']} packets, lost {link['lost_packets']} ({link['lost_percent']:.2f}%), "
              f"connections {link['connections']}")
    print(f"Reader alive: {buffer_thread.is_alive()} | errors: {errors}")


if __name__ == "__main__":
    main()
