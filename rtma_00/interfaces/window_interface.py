from abc import ABC, abstractmethod
from PyQt6.QtWidgets import QWidget

class WindowInterface(QWidget, ABC):
    """
    Standard base class for all major application windows.

    Provides a consistent structure and setup.
    """

    @abstractmethod
    def setup_ui(self):
        pass

    @abstractmethod
    def update_data(self, data):
        pass
