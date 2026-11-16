import pandas as pd
from collections import Counter
from tremor_dsp import compute_spectrum, detect_tremor, gyro_level

df = pd.read_csv("data/test.csv")  
accel = df["accel_g"].to_numpy()
gyro = df["gyro_dps"].to_numpy()
labels = df["label"].to_numpy()
results = []
for start in range(0, len(accel) - 256 + 1, 32):
    accel_window = accel[start:start + 256]
    gyro_window = gyro[start:start + 256]
    labels_win = labels[start:start + 256]
    top_label, top_count = Counter(labels_win).most_common(1)[0]
    label_major = top_label
    coverage = top_count / len(labels_win)  
    if coverage < 0.8:    
        continue      
    freqs, mags = compute_spectrum(accel_window)
    det = detect_tremor(freqs, mags)         
    peak = det["peak_hz"]
    ratio = det["band_ratio"] 
    peak_mags = det["peak_mags"]
    g = gyro_level(gyro_window)
    
    results.append({"label": label_major, "peak": peak, "ratio": ratio, "gyro": g, "peak_height": peak_mags})
out = pd.DataFrame(results)
summary = out.groupby("label").agg(["min", "median", "max"])
print(summary)