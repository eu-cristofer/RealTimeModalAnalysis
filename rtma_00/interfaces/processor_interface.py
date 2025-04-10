from abc import ABC, abstractmethod
import numpy as np

class ProcessorInterface(ABC):
    """
    Abstract base class for processing modules (e.g., FFT, EMA).

    Methods
    -------
    process(data: np.ndarray):
        Process input data and return processed output.
    """

    @abstractmethod
    def process(self, data: np.ndarray):
        pass
