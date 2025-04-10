"""
signal_channel.py

Manages multiple waveform components for a single signal channel.
"""

from typing import List
import numpy as np
from core.waveforms import WaveformComponent


class SignalChannel:
    """
    Represents a signal channel composed of multiple waveform components.
    """

    def __init__(self, sample_rate=1000, duration=1.0):
        """
        Initialize the signal channel.

        Parameters
        ----------
        sample_rate : int
            Sampling rate in Hz.
        duration : float
            Duration of the signal in seconds.
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.components: List[WaveformComponent] = []

    def add_component(self, component: WaveformComponent):
        """
        Add a waveform component to the channel.

        Parameters
        ----------
        component : WaveformComponent
            A waveform object (e.g., SineWave, SquareWave).
        """
        self.components.append(component)

    def remove_component(self, index: int):
        """
        Remove a waveform component by index.

        Parameters
        ----------
        index : int
            Index of the component to remove.
        """
        if 0 <= index < len(self.components):
            del self.components[index]

    def generate_signal(self) -> np.ndarray:
        """
        Generate the composite signal by summing all components.

        Returns
        -------
        np.ndarray
            Composite signal array.
        """
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        signal = np.zeros_like(t)

        for component in self.components:
            signal += component.generate(t)

        return signal
