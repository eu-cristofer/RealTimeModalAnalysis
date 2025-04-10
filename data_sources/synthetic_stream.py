"""
Synthetic Data Stream with Multi-Waveform Support
=================================================

This module implements a synthetic data stream capable of generating multiple 
waveform types with configurable parameters per channel component.
"""

import time
import numpy as np
import logging

from .base_stream import BaseStream

class ChannelState:
    """Represents the state of a single output channel"""
    def __init__(self):
        self.active = False
        self.components = []
        self.start_time = 0
        self.stop_time = 0
        self.fade_progress = 0.


class SyntheticStream(BaseStream):
    """
    A configurable synthetic signal generator supporting multiple waveform components.

    Parameters
    ----------
    sample_rate : int, optional
        Sampling frequency in Hz, default 1000

    Attributes
    ----------
    components : list of list of dict
        Waveform components for each channel (4 channels total)
    noise : float
        Standard deviation of Gaussian noise
    fade_in_time : float
        Fade-in duration in seconds
    fade_out_time : float
        Fade-out duration in seconds
    running : bool
        Stream activity state
    start_time : float
        Timestamp of last start command
    stop_time : float
        Timestamp of stop command
    is_fading_out : bool
        Fade-out state flag

    Methods
    -------
    start()
        Start/reset the signal generator
    stop()
        Initiate fade-out or immediate stop
    read_chunk()
        Generate next data chunk with current parameters
    """

    def __init__(self, sample_rate=1000):
        self.current_sample = 0 
        self.sample_rate = sample_rate
        self.channels = [ChannelState() for _ in range(4)]
        self.noise = 0.01
        self.fade_in_time = 1.0
        self.fade_out_time = 1.0
        # self.running = False
        self.connected = False

    def connect(self):
        """Establish connection without starting generation"""
        self.connected = True

    def start_channel(self, ch_idx):
        """Start/restart a specific channel with fade-in"""
        channel = self.channels[ch_idx]
        channel.active = True
        channel.start_time = time.time()
        channel.stop_time = 0

    def stop_channel(self, ch_idx):
        """Initiate fade-out for specific channel"""
        channel = self.channels[ch_idx]
        if channel.active:
            channel.stop_time = time.time()

    def start(self):
        """Required by BaseStream - start all channels"""
        self.connect()
        # for channel in self.channels:
        #     channel.active = True
        #     channel.start_time = time.time()

    def stop(self):
        """Required by BaseStream - stop all channels"""
        for channel in self.channels:
            channel.active = False

    def start_all(self):
        """Connect and start all channels."""
        if not self.connected:
            self.connect()

        for ch_idx in range(len(self.channels)):
            self.start_channel(ch_idx)

        logging.info("All channels started")

    def stop_all(self):
        """Required by BaseStream - stop all channels"""
        for channel in self.channels:
            channel.active = False
            channel.stop_time = 0

        logging.info("All channels stopped")


    def read_chunk(self):
        """
        Generate a chunk of synthetic data.

        Returns
        -------
        numpy.ndarray
            Array of shape (4, chunk_size) containing generated signals
        """

        if not self.connected:
            return np.zeros((4, 256))

        chunk_size = 256
        t = (np.arange(chunk_size) + self.current_sample) / self.sample_rate
        self.current_sample += chunk_size  # Advance sample count

        signals = []

        for ch_idx, channel in enumerate(self.channels):
            if not channel.active:
                signals.append(np.zeros(chunk_size))
                continue

            # Calculate fade gain
            current_time = time.time()
            if channel.stop_time > 0:  # Fading out
                elapsed = current_time - channel.stop_time
                gain = max(1.0 - elapsed / self.fade_out_time, 0.0)
                if elapsed > self.fade_out_time:
                    channel.active = False
            else:  # Fading in or running
                elapsed = current_time - channel.start_time
                gain = min(elapsed / self.fade_in_time, 1.0)

            channel_signal = np.zeros(chunk_size)

            for comp in channel.components:
                phase = 2 * np.pi * comp['freq'] * t + comp['phase']
                wave = comp['waveform']
                amp = comp['amp']

                if wave == 'sine':
                    sig = amp * np.sin(phase)
                elif wave == 'square':
                    sig = amp * np.sign(np.sin(phase))
                elif wave == 'sawtooth':
                    sig = amp * (2 * (t * comp['freq'] % 1) - 1)
                elif wave == 'triangle':
                    sig = amp * (2 * np.abs(2 * (t * comp['freq'] % 1) - 1) - 1)
                else:
                    sig = np.zeros(chunk_size)

                channel_signal += sig

            # Apply gain and noise
            channel_signal *= gain
            channel_signal += np.random.normal(0, self.noise, chunk_size)
            signals.append(channel_signal)

        return np.array(signals)
