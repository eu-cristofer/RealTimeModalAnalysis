# -*- coding: utf-8 -*-
""" Real-Time Modal Analysis and Machine Diagnostics GUI

Features:
- Live vibration analysis from multi-axis accelerometer
- Displays time-domain plots, FFT spectra, and cross-correlations
- Customizable scaling, FFT limit, and analysis controls
- Stylish and ergonomic interface for real-time streaming and diagnostics

Author: Cristofer Antoni Souza Costa (UI-inspired by ui_3)
Date: March 2025
"""

import sys
import numpy as np
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import (
    AcquisitionType, TerminalConfiguration, AccelUnits, ExcitationSource
)
from PyQt5.QtWidgets import (
    QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QStatusBar,
    QWidget, QGroupBox, QTabWidget, QSlider, QComboBox
)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from scipy.signal import correlate
import time

class ModalAnalysisLiveStream(QWidget):
    """Main application for real-time vibration analysis."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modal Analysis and Machine Diagnostics")
        self.resize(1200, 800)
        
        self.setup_ui()
        self.setup_data_buffers()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stream)
        self.daq_task = None
        
    def setup_ui(self):
        """Create and configure UI components."""
        main_layout = QVBoxLayout(self)
        tab_layout = QTabWidget()
        
        # Control Panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)

        control_layout.addWidget(QLabel("<h3>Controls</h3>"))
        # FFT Limit Control
        fft_group = QGroupBox("FFT Configuration")
        fft_layout = QHBoxLayout(fft_group)
        fft_layout.addWidget(QLabel("Max FFT Freq (Hz):"))
        self.fft_limit_input = QLineEdit("250")
        fft_layout.addWidget(self.fft_limit_input)
        control_layout.addWidget(fft_group)

        # Streaming Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)
        
        # Display Scaling Options
        scaling_group = QGroupBox("Display Options")
        scaling_layout = QVBoxLayout(scaling_group)
        scaling_layout.addWidget(QLabel("Scaling Mode:"))
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["Auto", "Manual"])
        self.scaling_combo.currentTextChanged.connect(self.toggle_scaling_mode)
        scaling_layout.addWidget(self.scaling_combo)
        control_layout.addWidget(scaling_group)

        main_layout.addWidget(control_panel)

        # Tab Section for Plots
        self.realtime_tab = self.create_plot_tab("Time Domain", "Time (s)", "Amplitude (g)")
        self.fft_tab = self.create_plot_tab("FFT Spectrum", "Frequency (Hz)", "Magnitude")
        self.correlation_tab = self.create_plot_tab("Cross-Correlation", "Lag (s)", "Correlation")

        tab_layout.addTab(self.realtime_tab, "Time-Domain")
        tab_layout.addTab(self.fft_tab, "FFT Spectrum")
        tab_layout.addTab(self.correlation_tab, "Cross-Correlation")
        main_layout.addWidget(tab_layout)
        
        # Status Bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
    
    def create_plot_tab(self, title, x_label, y_label):
        """Create a tab with a plot."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        plot = pg.PlotWidget(title=title)
        plot.setLabel('bottom', x_label)
        plot.setLabel('left', y_label)
        plot.showGrid(x=True, y=True)
        layout.addWidget(plot)
        tab.plot = plot
        tab.layout = layout
        return tab
    
    def setup_data_buffers(self):
        """Initialize data buffers and configurations."""
        self.sample_rate = 1000  # Hz
        self.chunk_size = 256    # Samples per read
        self.time_window = 10  # Time domain span (seconds)
        self.max_points_window = int(self.sample_rate * self.time_window)
        self.total_time_elapsed = 0
        self.buffers = [np.zeros(self.max_points_window) for _ in range(4)]
    
    def start_stream(self):
        """Configure DAQ and start live stream."""
        try:
            self.daq_task = nidaqmx.Task()
            for ch in ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"]:  # Replace with real channels
                self.daq_task.ai_channels.add_ai_accel_chan(
                    physical_channel=ch, sensitivity=100.0, 
                    terminal_config=TerminalConfiguration.DEFAULT,
                    min_val=-50.0, max_val=50.0, units=Accel 
