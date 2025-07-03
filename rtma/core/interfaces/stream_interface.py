from abc import ABC, abstractmethod
import numpy as np

class IStreamSource(ABC):
    """
    Interface for real-time streaming sources (SG, NI, OPC, etc.).
    """

    @abstractmethod
    def read_chunk(self, length: int) -> np.ndarray:
        """
        Read a chunk of samples from the stream.
        
        Returns
        -------
        np.ndarray
            Shape (channels, length)
        """
        pass

    @abstractmethod
    def get_channel_count(self) -> int:
        """
        Returns number of channels in stream.
        """
        pass

    @abstractmethod
    def get_channel_labels(self) -> list[str]:
        """
        Returns list of channel labels (e.g., ['Ref', 'X', 'Y', 'Z']).
        """
        pass
