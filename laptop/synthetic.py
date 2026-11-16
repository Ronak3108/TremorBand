import argparse
from pathlib import Path

import numpy as np

FS = 100
G = 9.81
ACCEL_NOISE_G = 0.004
GYRO_NOISE_DPS = 0.15

SUBJECTS = {
    "ronak": dict(tremor_g=0.06, gyro_scale=1.0, f0=5.0, reach_dps=150),
    "vidit": dict(tremor_g=0.10, gyro_scale=1.4, f0=4.8, reach_dps=190),
    "evaan": dict(tremor_g=0.04, gyro_scale=0.8, f0=5.3, reach_dps=120),
}


def _smooth_noise(rng, n, cutoff_hz):
    """Low-passed white noise, unit std (for wandering frequency / amplitude)."""
    x = rng.normal(size=n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    X[f > cutoff_hz] = 0
    y = np.fft.irfft(X, n)
    return y / (y.std() + 1e-12)


def _reduce(rng, acc_xyz, gyro_xyz, tilt):
    """3-axis -> magnitudes as the firmware computes them. Gravity is split so `tilt`=cos(angle) of the
    hand's vertical axis shows how much of the motion is along gravity."""
    n = acc_xyz.shape[0]
    grav = np.array([np.sqrt(1 - tilt ** 2), 0.0, tilt])
    a = grav[None, :] + acc_xyz + rng.normal(0, ACCEL_NOISE_G, (n, 3))
    g = gyro_xyz + rng.normal(0, GYRO_NOISE_DPS, (n, 3))
    return np.linalg.norm(a, axis=1), np.linalg.norm(g, axis=1)


def rest(rng, n, s):
    t = np.arange(n) / FS
    acc = np.zeros((n, 3)); gyro = np.zeros((n, 3))
    acc[:, 2] += 0.002 * np.sin(2 * np.pi * 0.3 * t)
    acc += 0.0015 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6))[:, None]
    gyro += rng.normal(0, 0.4, 3)[None, :] * _smooth_noise(rng, n, 1.0)[:, None]
    return _reduce(rng, acc, gyro, tilt=rng.uniform(0.9, 1.0))


def tremor(rng, n, s):
    """~5 Hz mimic: frequency wanders +/-0.4 Hz, amplitude breathes +/-30 %.  Gyro follows physics: a sensor
    ~10 cm from the wrist pivot sees omega = a/(omega*r)."""
    t = np.arange(n) / FS
    f = s["f0"] + 0.4 * _smooth_noise(rng, n, 0.3)
    phase = 2 * np.pi * np.cumsum(f) / FS
    amp_g = s["tremor_g"] * (1 + 0.3 * _smooth_noise(rng, n, 0.4))
    tilt = rng.uniform(0.6, 1.0)
    sig = np.sin(phase)
    acc = np.zeros((n, 3))
    acc[:, 2] = amp_g * sig / max(tilt, 0.6)
    acc[:, 0] = amp_g * sig * 0.5
    w = 2 * np.pi * f
    omega_peak = np.degrees(amp_g * G / (w * 0.10)) * s["gyro_scale"]
    gyro = np.zeros((n, 3))
    gyro[:, 1] = omega_peak * np.cos(phase)
    gyro[:, 0] = 0.35 * omega_peak * np.cos(phase + 0.7)
    return _reduce(rng, acc, gyro, tilt)


def reach(rng, n, s):
    """Repeated reach-and-return: bell-shaped gyro bursts 0.5-0.9 s long separated by 0.2-0.5 s pauses."""
    acc = np.zeros((n, 3)); gyro = np.zeros((n, 3))
    i = 0
    while i < n:
        dur = int(rng.uniform(0.5, 0.9) * FS)
        k = np.arange(min(dur, n - i))
        bell = np.sin(np.pi * k / dur) ** 2
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        peak = s["reach_dps"] * rng.uniform(0.6, 1.2)
        gyro[i:i + len(k)] += peak * bell[:, None] * ax[None, :]
        acc[i:i + len(k)] += rng.uniform(0.2, 0.6) * np.sin(np.pi * k / dur)[:, None] * rng.normal(size=3)[None, :] * 0.6
        i += dur + int(rng.uniform(0.2, 0.5) * FS)
    return _reduce(rng, acc, gyro, tilt=rng.uniform(0.5, 1.0))


def typing(rng, n, s):
    """Key taps: random 4-9 per second, each a short damped ~15 Hz pulse; wrist rests, small gyro."""
    acc = np.zeros((n, 3)); gyro = np.zeros((n, 3))
    t = 0.0
    kt = np.arange(0, 20)
    while t < n / FS:
        i = int(t * FS)
        m = min(len(kt), n - i)
        if m > 0:
            pulse = rng.uniform(0.02, 0.07) * np.exp(-kt[:m] / 3.0) * np.sin(2 * np.pi * 15 * kt[:m] / FS)
            acc[i:i + m, 2] += pulse
            gyro[i:i + m, 0] += rng.uniform(2, 8) * np.exp(-kt[:m] / 4.0) * rng.choice([-1, 1])
        t += rng.exponential(1 / rng.uniform(4, 9)) if rng.random() > 0.15 else rng.uniform(0.5, 1.5)
    gyro += rng.normal(0, 1.0, 3)[None, :] * _smooth_noise(rng, n, 2.0)[:, None]
    return _reduce(rng, acc, gyro, tilt=rng.uniform(0.9, 1.0))


def walking(rng, n, s):
    """Arm swing at ~0.95 Hz (gyro +/-60 dps) plus heel-strike impulses at the step rate ~1.9 Hz,
    whose harmonics (3rd = ~5.7 Hz) land in the tremor band - a deliberate false-positive test."""
    t = np.arange(n) / FS
    step = 1.9 + 0.1 * _smooth_noise(rng, n, 0.2)
    ph = 2 * np.pi * np.cumsum(step) / FS
    acc = np.zeros((n, 3)); gyro = np.zeros((n, 3))
    acc[:, 2] = 0.25 * np.sin(ph) + 0.12 * np.sin(2 * ph + 0.5)
    acc[:, 0] = 0.15 * np.sin(ph / 2)
    strike = np.maximum(0, np.sin(ph)) ** 8
    acc[:, 2] += rng.uniform(0.15, 0.3) * strike
    gyro[:, 1] = rng.uniform(40, 80) * s["gyro_scale"] * np.sin(ph / 2)
    gyro[:, 0] = 0.4 * gyro[:, 1] * np.cos(ph / 2)
    return _reduce(rng, acc, gyro, tilt=rng.uniform(0.7, 1.0))


GENERATORS = dict(rest=rest, tremor=tremor, reach=reach, typing=typing, walking=walking)


def write_csv(path, label, accel, gyro):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("t_ms,accel_g,gyro_dps,label\n")
        for i, (a, g) in enumerate(zip(np.round(accel, 3), np.round(gyro, 1))):
            f.write(f"{i * 10.0:.1f},{a:.3f},{g:.1f},{label}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("-s", "--seconds", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    n = int(args.seconds * FS) // 4 * 4
    for si, (name, s) in enumerate(SUBJECTS.items()):
        for li, (label, gen) in enumerate(GENERATORS.items()):
            rng = np.random.default_rng(args.seed + 100 * si + li)
            accel, gyro = gen(rng, n, s)
            write_csv(Path(args.out) / f"{label}_{name}.csv", label, accel, gyro)
    print(f"wrote {len(SUBJECTS) * len(GENERATORS)} files to {args.out}")


if __name__ == "__main__":
    main()
