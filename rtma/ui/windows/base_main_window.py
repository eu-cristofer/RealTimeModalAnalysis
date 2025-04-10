"""
base_main_window.py

Defines the base main window class for the application. This class provides
a common structure and shared setup logic for dockable applications, intended
to be subclassed by specific app implementations.
"""

from PyQt6.QtWidgets import QMainWindow


class BaseMainWindow(QMainWindow):
    """
    BaseMainWindow provides the foundational window structure for the application.
    
    Subclasses should override `setup_ui()` to initialize and add specific
    dockable widgets or tools.
    
    Attributes:
        title (str): The title displayed on the main application window.
    """

    def __init__(self, title="Application"):
        """
        Initialize the base main window with a default title and size.

        Args:
            title (str): The title for the main window. Defaults to "Application".
        """
        super().__init__()
        self.setWindowTitle(title)

    def setup_ui(self):
        """
        Placeholder method to initialize and add dockable tools/widgets.
        
        Should be implemented by subclasses to define the actual UI layout.
        """
        pass
