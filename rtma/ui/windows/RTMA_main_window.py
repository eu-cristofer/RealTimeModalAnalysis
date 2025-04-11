"""
rtma_main_window.py

Main window for the Real-Time Modal Analyzer (RTMA).
"""

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton,
    QComboBox, QToolBar, QLabel
)
from PyQt6.QtCore import QTimer

from ui.components.chart_widget import ChartWidget
from ui.components.fft_chart_widget import FFTChartWidget
from ui.components.toggle_switch import ToggleSwitch
from core.analyzer import compute_fft
from utils.theme_manager import theme_manager

import numpy as np


class RTMAMainWindow(QMainWindow):
    """
    Main application window for real-time modal signal analysis.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Modal Analyzer")
        self.resize(1200, 700)

        self._setup_ui()
        self._setup_timer()

        # Apply dark theme and set chart backgrounds accordingly
        theme_manager.apply_theme("dark")


    def _setup_ui(self):
        self._setup_toolbar()

        # Central layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Charts: Time-domain and FFT
        self.time_chart = ChartWidget("Live Time Signal")
        self.fft_chart = FFTChartWidget("Frequency Spectrum")

        chart_row = QHBoxLayout()
        chart_row.addWidget(self.time_chart)
        chart_row.addWidget(self.fft_chart)

        main_layout.addLayout(chart_row)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _setup_toolbar(self):
        toolbar = QToolBar("RTMA Toolbar")
        self.addToolBar(toolbar)

        # Stream selector
        self.stream_selector = QComboBox()
        self.stream_selector.addItems(["Synthetic", "Sensor A", "Sensor B"])

        # Controls
        self.pause_button = QPushButton("⏸ Pause")
        self.capture_button = QPushButton("📸 Capture")

        # Theme switch
        self.theme_switch = ToggleSwitch()  # Already handles connection internally

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
        self.timer.timeout.connect(self._update_chart)
        self.timer.start()

    def _update_chart(self):
        # Simulate synthetic signal input
        t = np.linspace(0, 0.02, 1000)
        signal = np.sin(2 * np.pi * 60 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

        self.time_chart.update_data(signal)

        freqs, mag = compute_fft(signal, sample_rate=1000)
        self.fft_chart.update_spectrum(freqs, mag)

    def _toggle_theme(self):
        theme_manager.toggle_theme()

