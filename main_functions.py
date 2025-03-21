# -*- coding: utf-8 -*-
"""
Real-time vibration data acquisition and analysis from NI cDAQ-9181.

This script configures a DAQ task to collect accelerometer data from 4 channels:
- One reference axis accelerometer (single-axis)
- One triaxial accelerometer (X, Y, Z)

It reads data in finite acquisition mode, plots:
- Time-domain signals
- Frequency spectra (FFT)
- Cross-correlation between each axis and reference.

Author: Cristofer Antoni Souza Costa (with GPT-tutor)
Date: March 19, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, AccelUnits, ExcitationSource
from nidaqmx.stream_readers import AnalogMultiChannelReader
from scipy.signal import correlate
from config import CHANNELS


def acquire_vibration_data(channels, sample_rate=1000, samples_per_read=2048):
    """
    Acquire vibration data from specified channels.

    Parameters
    ----------
    channels : list of str
        NI-DAQ physical channel names (e.g., ['cDAQ9181Mod1/ai0', ...]).
    sample_rate : float, optional
        Sampling frequency in Hz. The default is 1000.
    samples_per_read : int, optional
        Number of samples to acquire per channel. The default is 2048.

    Returns
    -------
    data : ndarray
        2D array with shape (4, samples_per_read). Each row corresponds to a channel.

    """
    data = np.zeros((4, samples_per_read))
    with nidaqmx.Task() as task:
        for ch in channels:
            task.ai_channels.add_ai_accel_chan(
                physical_channel=ch,
                sensitivity=100.0,  # mV/g
                terminal_config=TerminalConfiguration.DEFAULT,
                min_val=-50.0,
                max_val=50.0,
                units=AccelUnits.G,
                current_excit_source=ExcitationSource.INTERNAL,
                current_excit_val=0.002  # 2 mA excitation
            )

        task.timing.cfg_samp_clk_timing(rate=sample_rate, sample_mode=AcquisitionType.FINITE,
                                        samps_per_chan=samples_per_read)
        reader = AnalogMultiChannelReader(task.in_stream)
        task.start()
        reader.read_many_sample(data, number_of_samples_per_channel=samples_per_read, timeout=10.0)
    return data


def plot_time_signals(data, sample_rate):
    """
    Plot time-domain signals for each channel.

    Parameters
    ----------
    data : ndarray
        Acquired data array with shape (4, samples_per_read).
    sample_rate : float
        Sampling frequency in Hz.
    """
    t = np.linspace(0, data.shape[1] / sample_rate, data.shape[1])
    labels = ['Reference', 'X-axis', 'Y-axis', 'Z-axis']

    for i, label in enumerate(labels):
        plt.figure()
        plt.plot(t, data[i])
        plt.title(f"{label} - Time Domain")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [g]")
        plt.grid()
        plt.show()


def plot_fft(data, sample_rate):
    """
    Plot FFT magnitude spectra for each channel.

    Parameters
    ----------
    data : ndarray
        Acquired data array with shape (4, samples_per_read).
    sample_rate : float
        Sampling frequency in Hz.
    """
    N = data.shape[1]
    freqs = np.fft.rfftfreq(N, 1 / sample_rate)

    for i, label in enumerate(['Reference', 'X-axis', 'Y-axis', 'Z-axis']):
        fft_vals = np.abs(np.fft.rfft(data[i]))
        plt.figure()
        plt.plot(freqs, fft_vals)
        plt.title(f"{label} - FFT Spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [g]")
        plt.grid()
        plt.show()


def plot_cross_correlation(data):
    """
    Plot cross-correlation between each axis (X, Y, Z) and the reference channel.

    Parameters
    ----------
    data : ndarray
        Acquired data array with shape (4, samples_per_read).
    """
    ref = data[0]
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    for i, axis_data in enumerate(data[1:]):
        corr = correlate(axis_data, ref, mode='full')
        lags = np.arange(-len(ref) + 1, len(ref))
        plt.figure()
        plt.plot(lags, corr)
        plt.title(f"Cross-Correlation: {axis_labels[i]} vs Reference")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.grid()
        plt.show()


def main():
    """
    Main function to perform acquisition and plotting.
    """
    sample_rate = 1000  # Hz
    samples_per_read = 2048

    # Step 1: Acquire data
    data = acquire_vibration_data(CHANNELS, sample_rate, samples_per_read)

    # Step 2: Plot results
    plot_time_signals(data, sample_rate)
    plot_fft(data, sample_rate)
    plot_cross_correlation(data)


if __name__ == "__main__":
    main()
