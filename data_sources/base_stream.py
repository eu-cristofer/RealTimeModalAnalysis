from abc import ABC, abstractmethod

class BaseStream(ABC):
    """Abstract base class defining stream interface."""
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def read_chunk(self):
        """Return 2D numpy array: shape (4, chunk_size)"""
        pass