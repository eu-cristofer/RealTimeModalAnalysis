# -*- coding: utf-8 -*-
"""
Refactored and enhanced real-time vibration streaming GUI.
Maintains all original features from live_stream01.py.
Adds dark/light theme, professional layout, tabbed plots, toolbar, and more.
"""

import sys
import time
import numpy as np
import sqlite3
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, AccelUnits, ExcitationSource
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QPushButton, QSplitter,
                             QTabWidget, QGroupBox, QStatusBar, QToolBar, QAction, QStyleFactory, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from scipy.signal import correlate

# Ensure you have this list defined somewhere or import it correctly
try:
    from config import CHANNELS
except ImportError:
    CHANNELS = ["Dev1/ai0", "Dev1/ai1"]  # Example fallback or user should define their own

class VibrationMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Vibration Monitoring")
        self.resize(1600, 900)

        self.reader = None
        self.stream_timer = QTimer()
        self.stream_timer.timeout.connect(self.read_stream)
        self.stream_task = None

        self.init_ui()
        self.data_buffer = []
        self.start_time = None

    def init_ui(self):
        self.setup_toolbar()
        self.setup_statusbar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: Controls
        control_panel = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)

        self.fft_limit_input = QLineEdit("250")
        self.theme_switch = QCheckBox("Dark Mode")
        self.theme_switch.setChecked(True)
        self.theme_switch.stateChanged.connect(self.toggle_theme)

        self.start_button = QPushButton("Start Stream")
        self.stop_button = QPushButton("Stop Stream")
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)

        control_layout.addWidget(QLabel("Max FFT Frequency (Hz):"))
        control_layout.addWidget(self.fft_limit_input)
        control_layout.addWidget(self.theme_switch)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()

        splitter.addWidget(control_panel)

        # Right: Plots
        self.tabs = QTabWidget()
        self.time_plot = pg.PlotWidget(title="Time Domain")
        self.fft_plot = pg.PlotWidget(title="FFT Spectrum")
        self.xcorr_plot = pg.PlotWidget(title="Cross-Correlation")

        self.tabs.addTab(self.time_plot, "Time")
        self.tabs.addTab(self.fft_plot, "FFT")
        self.tabs.addTab(self.xcorr_plot, "X-Corr")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 4)

        self.toggle_theme()  # Set initial theme

    def setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        start_action = QAction("Start", self)
        stop_action = QAction("Stop", self)
        start_action.triggered.connect(self.start_stream)
        stop_action.triggered.connect(self.stop_stream)
        toolbar.addAction(start_action)
        toolbar.addAction(stop_action)

    def setup_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def toggle_theme(self):
        if self.theme_switch.isChecked():
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')
        else:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
        self.repaint_plots()

    def repaint_plots(self):
        for plot in [self.time_plot, self.fft_plot, self.xcorr_plot]:
            plot.clear()

    def start_stream(self):
        try:
            self.status.showMessage("Streaming...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.data_buffer = []
            self.start_time = time.time()

            self.stream_task = nidaqmx.Task()
            for ch in CHANNELS:
                self.stream_task.ai_channels.add_ai_accel_chan(
                    ch, terminal_config=TerminalConfiguration.RSE,
                    units=AccelUnits.METERS_PER_SECOND_SQUARED,
                    sensitivity=0.1, excitation_source=ExcitationSource.INTERNAL)

            self.stream_task.timing.cfg_samp_clk_timing(
                1000, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=1000)

            self.reader = AnalogMultiChannelReader(self.stream_task.in_stream)
            self.reader_data = np.zeros((len(CHANNELS), 1000))

            self.stream_task.start()
            self.stream_timer.start(100)
        except Exception as e:
            self.status.showMessage(f"Start Error: {e}")

    def stop_stream(self):
        self.status.showMessage("Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.stream_timer.stop()
        if self.stream_task:
            try:
                self.stream_task.stop()
                self.stream_task.close()
                self.stream_task = None
            except Exception as e:
                self.status.showMessage(f"Stop Error: {e}")

    def read_stream(self):
        try:
            self.reader.read_many_sample(self.reader_data, number_of_samples_per_channel=1000)
            current_time = time.time() - self.start_time
            self.data_buffer.append((current_time, self.reader_data.copy()))

            # Time-Domain Plot
            self.time_plot.clear()
            for i in range(len(CHANNELS)):
                self.time_plot.plot(self.reader_data[i], pen=pg.intColor(i))

            # FFT Plot
            self.fft_plot.clear()
            fft_limit = float(self.fft_limit_input.text())
            for i in range(1, len(CHANNELS)):
                fft_vals = np.abs(np.fft.rfft(self.reader_data[i]))
                freqs = np.fft.rfftfreq(len(self.reader_data[i]), 1 / 1000)
                mask = freqs <= fft_limit
                self.fft_plot.plot(freqs[mask], fft_vals[mask], pen=pg.intColor(i))

            # Cross-Correlation Plot
            self.xcorr_plot.clear()
            ref = self.reader_data[0]
            for i in range(1, len(CHANNELS)):
                corr = correlate(self.reader_data[i], ref, mode='full')
                lags = np.arange(-len(ref) + 1, len(ref))
                self.xcorr_plot.plot(lags, corr, pen=pg.intColor(i))

        except Exception as e:
            self.status.showMessage(f"Read Error: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    win = VibrationMonitor()
    win.show()
    sys.exit(app.exec_())
