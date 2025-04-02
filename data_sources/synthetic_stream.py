import numpy as np
import time
from .base_stream import BaseStream

class SyntheticStream(BaseStream):
    def __init__(self, sample_rate=1000, chunk_size=256):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.t = 0
        self.freq = 10  # Hz
        self.phase_shift = [0, 0.5, -0.3]  # phase differences between axes

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def read_chunk(self):
        if not self.running:
            return np.zeros((4, self.chunk_size))

        t_vals = np.linspace(self.t, self.t + self.chunk_size / self.sample_rate, self.chunk_size, endpoint=False)
        ref = np.sin(2 * np.pi * self.freq * t_vals) + np.random.normal(0, 0.01, self.chunk_size)
        x = np.sin(2 * np.pi * self.freq * t_vals + self.phase_shift[0]) + np.random.normal(0, 0.01, self.chunk_size)
        y = np.sin(2 * np.pi * self.freq * t_vals + self.phase_shift[1]) + np.random.normal(0, 0.01, self.chunk_size)
        z = np.sin(2 * np.pi * self.freq * t_vals + self.phase_shift[2]) + np.random.normal(0, 0.01, self.chunk_size)

        self.t += self.chunk_size / self.sample_rate
        return np.vstack([ref, x, y, z])
