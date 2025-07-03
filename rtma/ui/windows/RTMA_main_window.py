"""
RTMA_main_window.py

Main window for the Real-Time Modal Analyzer (RTMA).
"""

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QToolBar, QLabel, QComboBox,
    QPushButton, QDockWidget
)
from PyQt6.QtCore import QTimer, Qt
from rtma.utils.theme_manager import theme_manager
from rtma.ui.components.toggle_switch import ToggleSwitch
from rtma.ui.components.stream_manager_dock import StreamManagerDock
from rtma.ui.components.stream_dock_widget import StreamDockWidget
from rtma.ui.components.sg_stream_widget import SGStreamSource
import subprocess


class RTMAMainWindow(QMainWindow):
    """
    Main application window for real-time modal signal analysis.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Modal Analyzer")
        self.resize(1200, 800)

        self._setup_toolbar()
        self._setup_timer()

        # Apply dark theme
        theme_manager.apply_theme("dark")

        # Manage stream docks
        self.stream_docks = {}

        # Add Stream Manager dock on the left
        self.stream_manager = StreamManagerDock()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.stream_manager)

        # Connect signals
        self.stream_manager.add_sg_stream.connect(self._add_sg_stream)
        self.stream_manager.add_ni_stream.connect(self._add_ni_stream)
        self.stream_manager.remove_stream.connect(self._remove_stream)

    def _setup_toolbar(self):
        toolbar = QToolBar("RTMA Toolbar")
        self.addToolBar(toolbar)

        self.stream_selector = QComboBox()
        self.stream_selector.addItems(["None", "Signal Generator", "NI cDAQ"])

        self.pause_button = QPushButton("⏸ Pause")
        self.capture_button = QPushButton("📸 Capture")

        self.theme_switch = ToggleSwitch()

        toolbar.addWidget(QLabel("Input:"))
        toolbar.addWidget(self.stream_selector)
        toolbar.addSeparator()
        toolbar.addWidget(self.pause_button)
        toolbar.addWidget(self.capture_button)
        toolbar.addSeparator()
        toolbar.addWidget(self.theme_switch)

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(50)

    def _add_sg_stream(self):
        name = "SG Stream"
        if name in self.stream_docks:
            return

        import subprocess
        import time
        from core.shared_buffer import SHM_NAME
        from multiprocessing import shared_memory
        from ui.components.sg_stream_widget import SGStreamSource
        from ui.components.stream_dock_widget import StreamDockWidget
        from PyQt6.QtWidgets import QDockWidget

        subprocess.Popen(["python", "rtma/app_RTMS.py", "--mode", "sg"])

        # Wait for shared memory to become available
        for _ in range(20):  # try for ~2 seconds
            try:
                shm = shared_memory.SharedMemory(name=SHM_NAME)
                shm.close()
                break
            except FileNotFoundError:
                time.sleep(0.1)
        else:
            print("ERROR: Shared buffer not found. SG app failed to start?")
            return

        stream = SGStreamSource()
        content = StreamDockWidget(stream, title=name)

        dock_widget = QDockWidget(name)
        dock_widget.setWidget(content)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_widget)
        self.stream_docks[name] = dock_widget
        self.stream_manager.add_stream_name(name)



    def _add_ni_stream(self):
        # Placeholder for NI stream integration
        name = "NI Stream"
        if name in self.stream_docks:
            return
        # You can later replace this with NIStreamSource
        # stream = NIStreamSource()
        # dock = StreamDockWidget(stream, title=name)
        # self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        # self.stream_docks[name] = dock
        # self.stream_manager.add_stream_name(name)
        pass

    def _remove_stream(self, name: str):
        if name in self.stream_docks:
            dock_widget = self.stream_docks.pop(name)
            self.removeDockWidget(dock_widget)

