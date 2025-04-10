"""
analyzer.py

Core tools for real-time frequency-domain analysis.
"""

import numpy as np

def compute_fft(signal: np.ndarray, sample_rate: float):
    """
    Compute the FFT and corresponding frequency bins.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal input.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins.
    magnitude : np.ndarray
        Magnitude spectrum (log-scaled).
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)
    fft_vals = np.fft.rfft(signal)
    magnitude = 20 * np.log10(np.abs(fft_vals) + 1e-8)  # dB scale, safe log

    return freqs, magnitude
