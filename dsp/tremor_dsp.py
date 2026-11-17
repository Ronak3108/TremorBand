import json
from pathlib import Path

import numpy as np

CONFIG = {
    "tremor_band_hz":   (3.0, 7.0),
    "total_band_hz":    (0.5, 15.0),
    "tremor_min_peak_g": 0.015,
    "tremor_min_ratio":  0.50,
    "gyro_on":          40.0,
    "gyro_off":         30.0,
    "gyro_tail_s":       0.64,
    "hold_on_s":         0.30,
    "hold_off_s":        0.60,
    "hop_s":             0.32,
}

_THRESHOLDS_FILE = Path(__file__).with_name("thresholds.json")


def load_thresholds(path=_THRESHOLDS_FILE):
    """Overlay calibrated values from thresholds.json (if present) onto CONFIG."""
    path = Path(path)
    if path.is_file():
        data = json.loads(path.read_text())
        for key, value in data.get("config", data).items():
            if key in CONFIG:
                CONFIG[key] = tuple(value) if isinstance(CONFIG[key], tuple) else value
    return CONFIG


load_thresholds()

GYRO_ON = CONFIG["gyro_on"]
GYRO_OFF = CONFIG["gyro_off"]
HOLD_S = CONFIG["hold_off_s"]
HOP_S = CONFIG["hop_s"]


def compute_spectrum(accel_window, fs=100):
    """Amplitude spectrum (g) of a window of accel magnitude, 0-15 Hz.

    Removes the window mean (gravity), applies a Hann window, rfft, then scales
    so a sine of amplitude A shows up as a peak of about A.
    """
    x = np.asarray(accel_window, dtype=float)
    n = len(x)
    if n != 256:
        raise ValueError(f"Length should be 256, got {n}.")
    x = x - x.mean()  # remove gravity
    w = np.hanning(n)
    mags = np.abs(np.fft.rfft(w * x)) / w.sum()
    mags[1:-1] *= 2  # single-sided amplitude
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    keep = freqs <= 15
    return freqs[keep], mags[keep]


def _parabolic_peak(mags, i, df):
    """Sub-bin peak position (Hz offset from bin i) by fitting a parabola through 3 bins."""
    if i <= 0 or i >= len(mags) - 1:
        return 0.0
    a, b, c = mags[i - 1], mags[i], mags[i + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5) * df)


def detect_tremor(freqs, mags):
    """Look for a sine-like peak in the tremor band.

    Returns dict(peak_hz, band_ratio, is_tremor, peak_mags). A tremor needs both a large enough
    peak (g) and most of the 0.5-15 Hz power inside the tremor band.
    """
    lo, hi = CONFIG["tremor_band_hz"]
    freqs = np.asarray(freqs)
    mags = np.asarray(mags)
    in_band = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    if in_band.size == 0:
        return dict(peak_hz=0.0, band_ratio=0.0, is_tremor=False, peak_mags=0.0)

    k = in_band[np.argmax(mags[in_band])]
    peak_mags = float(mags[k])
    df = float(freqs[1] - freqs[0])
    peak_hz = float(freqs[k] + _parabolic_peak(mags, k, df))

    tlo, thi = CONFIG["total_band_hz"]
    total = np.sum(mags[(freqs >= tlo) & (freqs <= thi)] ** 2)
    band = np.sum(mags[in_band] ** 2)
    band_ratio = float(band / total) if total > 1e-12 else 0.0

    # a band-edge maximum that is only leakage from a larger peak outside the band is not a tremor
    is_local_peak = (k == 0 or mags[k] >= mags[k - 1]) and (k == len(mags) - 1 or mags[k] >= mags[k + 1])

    is_tremor = bool(
        is_local_peak
        and band_ratio >= CONFIG["tremor_min_ratio"]
        and peak_mags >= CONFIG["tremor_min_peak_g"]
    )
    return dict(peak_hz=peak_hz, band_ratio=band_ratio, is_tremor=is_tremor, peak_mags=peak_mags)


def gyro_level(gyro_window, tail_s=None, fs=100):
    """Mean gyro magnitude (deg/s) over the last `tail_s` seconds of the window."""
    tail_s = CONFIG["gyro_tail_s"] if tail_s is None else tail_s
    g = np.asarray(gyro_window, dtype=float)
    n = max(1, int(round(tail_s * fs)))
    return float(np.mean(g[-n:]))


def detect_action(gyro_window):
    """Stateless check (no hysteresis) - fine for scripts; the dashboard uses ActionDetector."""
    return bool(gyro_level(gyro_window) > CONFIG["gyro_on"])


class ActionDetector:
    """Hysteresis (different on/off levels) + hold time, so the banner never flickers."""

    def __init__(self, on=None, off=None, hold_on_s=None, hold_off_s=None, hop_s=None):
        self.on = CONFIG["gyro_on"] if on is None else on
        self.off = CONFIG["gyro_off"] if off is None else off
        if self.off >= self.on:
            raise ValueError(f"gyro_off ({self.off}) must be below gyro_on ({self.on})")
        hop_s = CONFIG["hop_s"] if hop_s is None else hop_s
        hold_on_s = CONFIG["hold_on_s"] if hold_on_s is None else hold_on_s
        hold_off_s = CONFIG["hold_off_s"] if hold_off_s is None else hold_off_s
        self.hold_on = max(1, round(hold_on_s / hop_s))
        self.hold_off = max(1, round(hold_off_s / hop_s))
        self.active = False
        self.count = 0
        self.level = 0.0

    def update(self, gyro_window):
        self.level = gyro_level(gyro_window)
        wants_switch = self.level < self.off if self.active else self.level > self.on
        self.count = self.count + 1 if wants_switch else 0
        needed = self.hold_off if self.active else self.hold_on
        if self.count >= needed:
            self.active = not self.active
            self.count = 0
        return self.active