"""
launcher_window.py

App Selector Window for launching SG, RTMA, or both.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from app.main import run_main_window


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎛️ RTMA App Suite")
        self.resize(300, 200)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        title = QLabel("🔧 Real-Time Modal Suite")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)

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
        self.hide()
