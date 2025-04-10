from abc import ABC, abstractmethod

class StreamInterface(ABC):
    """
    Abstract base class for all signal streams.

    Methods
    -------
    start_stream():
        Starts the data stream.
    
    stop_stream():
        Stops the data stream.
    
    get_latest_data():
        Returns the latest batch of data for processing.
    """

    @abstractmethod
    def start_stream(self):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    @abstractmethod
    def get_latest_data(self):
        pass
