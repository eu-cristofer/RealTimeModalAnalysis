"""
SG_main_window.py

Main window for the Signal Generator Station.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QToolBar, QPushButton, QComboBox,
    QWidget, QVBoxLayout, QTabWidget, QLabel
)
from PyQt6.QtCore import QTimer, Qt

from ui.components.chart_widget import ChartWidget
from ui.components.toggle_switch import ToggleSwitch
from ui.windows.signal_generator_desk import SignalGeneratorDesk
from utils.theme_manager import theme_manager
from core.shared_buffer import create_shared_buffer

import numpy as np


class SGMainWindow(QMainWindow):
    """
    The main application window for the Signal Generator.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Generator Station")
        self.resize(792, 900)

        self.plot_enabled = [True, True, True, True]
        self.channel_tabs = []
        self.write_index = 0  # for shared memory ring buffer

        
        self._setup_ui()
        self._setup_timer()

        # Apply theme
        theme_manager.apply_theme("dark")

        # 🔌 Create shared memory buffer for RTMA streaming
        self.shm, self.shared_buffer = create_shared_buffer()

    def _setup_ui(self):
        self._setup_toolbar()

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        self.chart = ChartWidget(title="Combined Channel Output")
        main_layout.addWidget(self.chart)

        self.tab_widget = QTabWidget()
        for i in range(4):
            tab = SignalGeneratorDesk(channel_id=i)
            tab.plot_checkbox.stateChanged.connect(lambda state, idx=i: self._toggle_plot(idx, state))
            self.channel_tabs.append(tab)
            self.tab_widget.addTab(tab, f"CH{i+1}")

        main_layout.addWidget(self.tab_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        self.start_btn = QPushButton("▶️ Start Signal")
        self.stop_btn = QPushButton("⏹️ Stop Signal")
        self.theme_toggle = ToggleSwitch()
        self.broadcast_btn = QPushButton("📡 Start Broadcast")

        self.output_selector = QComboBox()
        self.output_selector.addItems(["RTMA", "OPC"])

        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addWidget(self.theme_toggle)
        toolbar.addWidget(self.broadcast_btn)
        toolbar.addWidget(QLabel("Output:"))
        toolbar.addWidget(self.output_selector)

        self.start_btn.clicked.connect(self.start_signal)
        self.stop_btn.clicked.connect(self.stop_signal)

    def _toggle_plot(self, channel_index, state):
        self.plot_enabled[channel_index] = bool(state)

    def _toggle_theme(self):
        theme_manager.toggle_theme()

    def start_signal(self):
        self.timer.start()

    def stop_signal(self):
        self.timer.stop()

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._update_chart)

    def _update_chart(self):
        """
        Collects signals from each channel, updates chart, and writes to shared memory.
        """
        signal_data = {}
        chunk_size = 1000  # 1 second worth of data for simplicity (or use 50ms * 1000Hz = 50)

        for i, tab in enumerate(self.channel_tabs):
            if tab.plot_checkbox.isChecked():
                signal = tab.generate_signal()
                signal_data[i] = signal[:chunk_size]

        self.chart.plot_multiple(signal_data)
        self._write_to_shared_buffer(signal_data)

    def _write_to_shared_buffer(self, signals: dict[int, np.ndarray]):
        """
        Write current signal chunk into shared buffer with circular indexing.
        """
        CHUNK = len(next(iter(signals.values())))
        for ch, data in signals.items():
            buffer = self.shared_buffer[ch]
            end = self.write_index + CHUNK
            if end <= buffer.shape[0]:
                buffer[self.write_index:end] = data
            else:
                split = buffer.shape[0] - self.write_index
                buffer[self.write_index:] = data[:split]
                buffer[:CHUNK - split] = data[split:]

        self.write_index = (self.write_index + CHUNK) % self.shared_buffer.shape[1]
