import numpy as np


def compute_spectrum(accel_window, fs=100):
    """accel_window (256 samples, in g) -> (freqs in Hz, magnitudes), 0-15 Hz only."""
    x = np.asarray(accel_window, dtype=float)
    x = x - x.mean()                          # remove gravity
    mags = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    keep = freqs <= 15
    return freqs[keep], mags[keep]


def detect_tremor(freqs, mags):
    """-> dict(peak_hz, band_ratio, is_tremor). Crude: peak in 3-7 Hz, fixed thresholds."""
    band = (freqs >= 3) & (freqs <= 7)
    total = (freqs >= 0.5) & (freqs <= 15)
    peak_hz = float(freqs[band][np.argmax(mags[band])])
    band_ratio = float(np.sum(mags[band] ** 2) / np.sum(mags[total] ** 2))
    return {
        "peak_hz": peak_hz,
        "band_ratio": band_ratio,
        "is_tremor": bool(band_ratio > 0.5),
    }


def detect_action(gyro_window):
    """gyro_window (256 samples, in deg/s) -> bool. Stub: never in action mode."""
    return False