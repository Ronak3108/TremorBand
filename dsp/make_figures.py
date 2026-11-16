import os
import numpy as np
import matplotlib.pyplot as plt
from tremor_dsp import compute_spectrum

t = np.arange(256) / 100
rng = np.random.default_rng(0)         
rest = 1.0 + 0.005 * rng.standard_normal(256)
tremor = 1.0 + 0.05 * np.sin(2 * np.pi * 5.0 * t)
reach = 1.0 + 0.5 * np.sin(2 * np.pi * 1.0 * t)

signals = {"Rest": rest, "Tremor": tremor, "Reach": reach}

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for col, (name, sig) in enumerate(signals.items()):
    ax_wave = axes[0, col]
    ax_wave.plot(t, sig)
    ax_wave.set_title(name)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Accel (g)")

    freqs, mags = compute_spectrum(sig)
    ax_spec = axes[1, col]
    ax_spec.bar(freqs, mags, width=0.3)
    ax_spec.set_xlim(0, 15)
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude (g)")

fig.suptitle("Simulated input: top and bottom rows use different y-scales")
plt.tight_layout()

os.makedirs("../figures", exist_ok=True)
plt.savefig("../figures/fig1_wave_vs_spectrum.png", dpi=150)


x = np.sin(2 * np.pi * 5.2 * t)
w = np.hanning(256)
freqs = np.fft.rfftfreq(256, d=1/100)
raw = np.abs(np.fft.rfft(x))
hann = np.abs(np.fft.rfft(x * w))

raw_db = 20 * np.log10(raw / raw.max() + 1e-12)
hann_db = 20 * np.log10(hann / raw.max() + 1e-12)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(freqs, raw_db, label="Rectangular (no window)")
ax.plot(freqs, hann_db, label="Hann")
ax.set_xlim(2, 8)
ax.set_ylim(-80, 5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB, relative to raw peak)")
ax.set_title("Leakage of a 5.2 Hz sine")
ax.legend()
plt.tight_layout()
plt.savefig("../figures/fig2_leakage.png", dpi=150)


t200 = np.arange(200) / 100
f_freqs = np.fft.rfftfreq(200, d=1/100)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, f0 in zip(axes, [5.0, 5.2]):
    m = np.abs(np.fft.rfft(np.sin(2 * np.pi * f0 * t200)))
    ax.bar(f_freqs, m, width=0.3)
    ax.set_xlim(3, 7)
    ax.set_title(f"{f0} Hz tone (N = 200)")
    ax.set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Magnitude (raw FFT)")
plt.tight_layout()
plt.savefig("../figures/fig3_bin_resolution.png", dpi=150)
plt.show()


