"""
Funções de plotagem em tempo real.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_signal(ax, data):
    ax.clear()
    ax.plot(data[0])
    ax.set_title("Sinal no Tempo")
    ax.set_xlabel("Amostras")
    ax.set_ylabel("Aceleração (g)")


def plot_fft(ax, data, fs):
    ax.clear()
    fft_vals = np.fft.rfft(data[0])
    fft_freqs = np.fft.rfftfreq(len(data[0]), d=1 / fs)
    ax.plot(fft_freqs, np.abs(fft_vals))
    ax.set_xlim(0, 100)
    ax.set_title("Espectro (0-100 Hz)")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Magnitude")


def plot_autocorrelation(ax, data):
    ax.clear()
    corr = np.correlate(data[0], data[0], mode='full')
    corr = corr[corr.size // 2:]
    ax.plot(corr)
    ax.set_title("Autocorrelação")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlação")