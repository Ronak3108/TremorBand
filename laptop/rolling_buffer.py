import numpy as np
import threading
from collections import deque
from protocol import SAMPLES_PER_PACKET


class RollingBuffer():
    def __init__(self, window: int = 256, hop: int = 32):
        self.window = window
        self.hop = hop
        self.accel = deque(maxlen = window)
        self.gyro = deque(maxlen = window)
        self.sample_count = 0
        self.thread_lock = threading.Lock()

    def push(self, samples: list[tuple[float, float]]):
        with self.thread_lock:
            accel_samples = [sample[0] for sample in samples]
            gyro_samples = [sample[1] for sample in samples]
            self.sample_count += len(samples)
            self.accel.extend(accel_samples)
            self.gyro.extend(gyro_samples)
        # print(self.accel, self.gyro)

    def get_window(self):
        with self.thread_lock:
            if(len(self.accel) != self.window or len(self.gyro) != self.window or self.sample_count < self.hop):
                return None
            else:
                self.sample_count = 0
                accel = np.array(self.accel)
                gyro = np.array(self.gyro)  

                return (accel, gyro)         



if __name__ == "__main__":
    rb = RollingBuffer(8, 4)
    rb.push([(1, 1), (2, 2), (3, 3)])
    print("Accessing window...")
    print(rb.get_window())
    rb.push([(4, 4), (5, 5), (6, 6), (7, 7), (8, 8)])
    print("Accessing window...")
    print(rb.get_window())
    print("Accessing window...")
    print(rb.get_window())
    rb.push([(9, 9), (10, 10), (11, 11), (12, 12)])
    print("Accessing window...")
    print(rb.get_window())
    

