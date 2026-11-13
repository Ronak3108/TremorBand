import numpy as np
def compute_spectrum(accel_window,fs=100):
    if len(accel_window) == 256:
        N = len(accel_window)
        centred_window = accel_window - accel_window.mean() #removed 1g from all accel values
        w = np.hanning(N)
        ftinp = w*centred_window
        X = np.fft.rfft(ftinp) #taking the ft, X is an array of complex numbers
        mags = np.abs(X)
        freqs = np.fft.rfftfreq(N,d=1/fs)
        mags = mags/w.sum() #normalising the ft output
        mags[1:-1]=mags[1:-1]*2
        mask = freqs <= 15
        freqs_kept = freqs[mask]
        mags_kept = mags[mask]
        return freqs_kept,mags_kept
        
    else:
        raise ValueError("Length should be 256.")


def detect_tremor(freqs, mags):
    tresh_band = 0.5 #random value, please configure it later
    tresh_mag = 0.01 #random value, please configure it later
    mask = (freqs<=7) & (freqs>=3)
    work_freqs = freqs[mask]
    work_mags = mags[mask]
    peak_index = np.argmax(work_mags)
    peak_mags = work_mags[peak_index]
    peak_hz = work_freqs[peak_index]
    mask_den = (freqs >= 0.5) & (freqs <= 15)
    work_den = mags[mask_den]
    num = np.sum(work_mags**2)
    den = np.sum(work_den **2)
    band_ratio = num/den
    if (band_ratio >= tresh_band) & (peak_mags >= tresh_mag):
        is_tremor = True
    else:
        is_tremor = False
    return dict(peak_hz=peak_hz, band_ratio=band_ratio, is_tremor=is_tremor, peak_mags = peak_mags)


GYRO_ON = 20.0   #configure it later
GYRO_OFF = 12.0  #configure it later
HOLD_S = 0.5      # seconds the new state must persist
HOP_S = 0.32      # seconds between display updates (32 samples at 100 Hz)


def gyro_level(gyro_window):
    return float(np.mean(gyro_window))


def detect_action(gyro_window):
    return bool(gyro_level(gyro_window) > GYRO_ON)


class ActionDetector:
    def __init__(self, on=GYRO_ON, off=GYRO_OFF, hold_s=HOLD_S, hop_s=HOP_S):
        self.on = on
        self.off = off
        self.hold_needed = max(1, round(hold_s / hop_s))
        self.active = False   
        self.count = 0    

    def update(self, gyro_window):
        level = gyro_level(gyro_window)
        if not self.active:
            wants_switch = level > self.on    
        else:
            wants_switch = level < self.off 
            
        if wants_switch:
            self.count += 1
        else:
            self.count = 0
        if self.count >= self.hold_needed:
            self.active = not self.active
            self.count = 0

        return self.active

        
