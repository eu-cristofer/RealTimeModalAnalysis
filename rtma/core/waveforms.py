
""" 
waveforms.py

Defines classes for generating synthetic waveform components.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy import signal


class WaveformComponent(ABC):
    """
    Abstract base class for waveform components.
    """

    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0, offset=0.0):
        """
        Initialize the waveform component.

        Parameters
        ----------
        amplitude : float
            Peak amplitude of the waveform.
        frequency : float
            Frequency in Hz.
        phase : float
            Phase offset in radians.
        offset : float
            Vertical offset.
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    @abstractmethod
    def generate(self, t: np.ndarray) -> np.ndarray:
        """
        Generate waveform data over the given time array.

        Parameters
        ----------
        t : np.ndarray
            Time array in seconds.

        Returns
        -------
        np.ndarray
            Signal values at each time point.
        """
        pass


class SineWave(WaveformComponent):
    """
    Sine wave generator.
    """

    def generate(self, t: np.ndarray) -> np.ndarray:
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase) + self.offset
    

class TriangleWave(WaveformComponent):
    def generate(self, t: np.ndarray) -> np.ndarray:
        return self.amplitude * signal.sawtooth(2 * np.pi * self.frequency * t + self.phase, 0.5) + self.offset


class SawtoothWave(WaveformComponent):
    def generate(self, t: np.ndarray) -> np.ndarray:
        return self.amplitude * signal.sawtooth(2 * np.pi * self.frequency * t + self.phase) + self.offset


class SquareWave(WaveformComponent):
    """
    Square wave generator.
    """

    def generate(self, t: np.ndarray) -> np.ndarray:
        return self.amplitude * np.sign(np.sin(2 * np.pi * self.frequency * t + self.phase)) + self.offset


class NoiseWave(WaveformComponent):
    """
    Gaussian noise generator.
    """
    def generate(self, t: np.ndarray) -> np.ndarray:
        return self.amplitude * np.random.normal(loc=0.0, scale=1.0, size=len(t)) + self.offset


class PinkNoiseWave(WaveformComponent):
    def generate(self,t: np.ndarray) -> np.ndarray:
        """
        Simple pink noise generator using Voss-McCartney algorithm (approximation).
        """
        white = np.random.randn(len(t))
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink = signal.lfilter(b, a, white)
        return self.amplitude * pink + self.offset