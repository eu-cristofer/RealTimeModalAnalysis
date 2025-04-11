"""
launcher_window.py

App Selector Window for launching SG, RTMA, or both.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from app.main import run_main_window
from ui.components.toggle_switch import ToggleSwitch
from utils.theme_manager import theme_manager


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎛️ RTMA App Suite")
        self.resize(300, 200)

        # Apply current theme to match app styling
        theme_manager.apply_theme("dark")
        theme_manager.theme_changed.connect(self._apply_theme_to_self)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.theme_toggle = ToggleSwitch()
        layout.addWidget(self.theme_toggle)

        self.title = QLabel("🔧 Real-Time Modal Suite")
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
        Optional: Adjust launcher-specific styles if needed.
        """
        # You can adjust colors here if needed
        pass
