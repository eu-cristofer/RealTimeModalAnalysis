"""
launcher_window.py

This module defines the LauncherWindow class, which serves as the initial
window for selecting and launching components of the Real-Time Modal Analysis
(RTMA) application suite.

Classes
-------
LauncherWindow : QWidget
    A simple application launcher with options to open the Signal Generator,
    RTMA Analyzer, or both simultaneously. Includes a theme toggle.

Usage
-----
This window is typically instantiated at the start of the application to
allow the user to choose which RTMA components to run.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel

from rtma.utils.logger import get_logger
from rtma.window_selector import run_main_window
from rtma.ui.components.toggle_switch import ToggleSwitch
from rtma.utils.theme_manager import theme_manager

log = get_logger(__name__)

class LauncherWindow(QWidget):
    """
    App Selector Window for launching Signal Generator, RTMA Analyzer, or both.

    This widget displays a simple interface with buttons for launching
    different RTMA modules and a toggle to switch application themes.

    Attributes
    ----------
    theme_toggle : ToggleSwitch
        A custom toggle widget for switching between light and dark themes.
    title : QLabel
        A label displaying the application name.
    sg_window : QMainWindow, optional
        The Signal Generator main window (if launched).
    rtma_window : QMainWindow, optional
        The RTMA Analyzer main window (if launched).
    app_window : QMainWindow, optional
        The selected single-mode application window (if launched).
    """
    def __init__(self):
        """
        Initialize the LauncherWindow UI.

        Sets up the window title, applies the theme, and builds the UI layout.
        """
        log.debug(f"Instantiating LauncherWindow")
        super().__init__()
        self.setWindowTitle("RTMA App Suite")
        self.resize(300, 200)

        # Apply current theme to match app styling
        theme_manager.apply_theme("dark")
        
        # Connecting theme_changed signal to a slot
        # theme_manager.theme_changed.connect(self._apply_theme_to_self)

        # Creating the User Interface Layout
        self._setup_ui()

    def _setup_ui(self):
        """
        Construct and arrange UI components in the layout.
        """
        layout = QVBoxLayout()
        self.theme_toggle = ToggleSwitch()
        layout.addWidget(self.theme_toggle)

        self.title = QLabel("🔧 Real-Time Modal Analysis Suite")
        self.title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(self.title)

        sg_button = QPushButton("🎛️ Launch Signal Generator")
        rtma_button = QPushButton("📈 Launch RTMA Analyzer")
        both_button = QPushButton("🔀 Launch Both")

        sg_button.clicked.connect(lambda: self._launch("sg"))
        rtma_button.clicked.connect(lambda: self._launch("rtma"))
        both_button.clicked.connect(lambda: self._launch("both"))

        layout.addWidget(sg_button)
        layout.addWidget(rtma_button)
        layout.addWidget(both_button)

        self.setLayout(layout)

    def _launch(self, target):
        """
        Launch the selected application module(s) and hide the launcher.

        Parameters
        ----------
        target : str
            One of "sg", "rtma", or "both", indicating which application(s)
            to launch.
        """
        from PyQt6.QtWidgets import QMainWindow

        if target == "both":
            self.sg_window: QMainWindow = run_main_window("sg")
            self.rtma_window: QMainWindow = run_main_window("rtma")
            self.sg_window.show()
            self.rtma_window.show()
        else:
            self.app_window: QMainWindow = run_main_window(target)
            self.app_window.show()

        self.hide()  # Hide only after windows are launched

    def _apply_theme_to_self(self, theme: str):
        """
        Slot for handling theme changes.

        Currently unused, but can be implemented to adjust Launcher-specific
        visual styles on theme change.

        Parameters
        ----------
        theme : str
            The current theme, either "light" or "dark".
        """
        pass
