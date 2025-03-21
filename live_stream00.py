# -*- coding: utf-8 -*-
"""
Real-time vibration live streaming GUI with PyQt5 and pyqtgraph.

Features:
- Continuously streams data from 4 channels (1 reference + 3-axis accelerometer).
- Displays:
    - Time-domain plots starting with a 10-second timespan.
    - When reading reaches 5 seconds, plot continues with the time window centered around the current reading (scrolling mid-screen) and accumulating total elapsed time.
    - FFT spectra with user-defined frequency limit.
    - Cross-correlation between each axis and the reference.
- Start and Stop streaming buttons.
- Improved charts with grid lines, legends, and axis labeling.

Author: Cristofer (tutored by GPT)
Date: March 2025
"""

import sys
import numpy as np
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, AccelUnits, ExcitationSource
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QLineEdit, QLabel, QStatusBar
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from scipy.signal import correlate
from config import CHANNELS


class VibrationLiveStream(QWidget):
    """Main window class for real-time vibration live streaming GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Vibration Live Stream")

        # Layouts
        self.layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # Buttons and input fields
        self.start_button = QPushButton("Start Live Stream")
        self.stop_button = QPushButton("Stop Live Stream")
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)

        self.fft_limit_label = QLabel("Max FFT Frequency (Hz):")
        self.fft_limit_input = QLineEdit("250")

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.fft_limit_label)
        button_layout.addWidget(self.fft_limit_input)

        self.layout.addLayout(button_layout)

        # shows short messages at the bottom of the window to give 
        # feedback to the user.
        self.status_bar = QStatusBar()
        self.layout.addWidget(self.status_bar)

        self.setup_plots()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stream)

        # Acquisition parameters for live stream
        self.sample_rate = 1000  # Hz
        self.chunk_size = 256    # Samples per read
        self.window_seconds = 10  # Start with a 10-second timespan
        self.max_points_window = int(self.sample_rate * self.window_seconds)
        self.total_time_elapsed = 0

        # Buffers for rolling time-domain display
        self.time_data_buffers = [np.zeros(self.max_points_window) for _ in range(4)]
        self.current_index = 0
        self.daq_task = None

    def setup_plots(self):
        # Plot widgets
        self.time_plot = pg.PlotWidget(title="Time-Domain Signals")
        self.time_plot.showGrid(x=True, y=True)
        self.time_plot.addLegend()
        self.time_plot.setLabel('bottom', 'Time (s)')
        self.time_plot.setLabel('left', 'Acceleration (g)')

        self.fft_plot = pg.PlotWidget(title="FFT Spectra")
        self.fft_plot.showGrid(x=True, y=True)
        self.fft_plot.addLegend()
        self.fft_plot.setLabel('bottom', 'Frequency (Hz)')
        self.fft_plot.setLabel('left', 'Amplitude')

        self.corr_plot = pg.PlotWidget(title="Cross-Correlation")
        self.corr_plot.showGrid(x=True, y=True)
        self.corr_plot.addLegend()
        self.corr_plot.setLabel('bottom', 'Lag (s)')
        self.corr_plot.setLabel('left', 'Correlation')

        self.layout.addWidget(self.time_plot)
        self.layout.addWidget(self.fft_plot)
        self.layout.addWidget(self.corr_plot)
        self.setLayout(self.layout)



    def start_stream(self):
        """Starts DAQ configuration and begins live streaming."""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_bar.showMessage("Starting live stream...")

        try:
            self.daq_task = nidaqmx.Task()
            for ch in CHANNELS:
                self.daq_task.ai_channels.add_ai_accel_chan(
                    physical_channel=ch,
                    sensitivity=100.0,
                    terminal_config=TerminalConfiguration.DEFAULT,
                    min_val=-50.0,
                    max_val=50.0,
                    units=AccelUnits.G,
                    current_excit_source=ExcitationSource.INTERNAL,
                    current_excit_val=0.002
                )

            self.daq_task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.chunk_size * 20
            )

            self.reader = AnalogMultiChannelReader(self.daq_task.in_stream)
            self.chunk_buffer = np.zeros((4, self.chunk_size))
            self.daq_task.start()

            channel_names = ["Reference", "X", "Y", "Z"]
            self.time_curves = [self.time_plot.plot(pen=pg.intColor(i), name=channel_names[i]) for i in range(4)]
            self.fft_curves = [self.fft_plot.plot(pen=pg.intColor(i), name=channel_names[i]) for i in range(4)]
            self.corr_curves = [self.corr_plot.plot(pen=pg.intColor(i+1), name=f"{channel_names[i+1]} vs Ref") for i in range(3)]

            self.total_time_elapsed = 0
            self.timer.start(50)
            self.status_bar.showMessage("Streaming in progress...")
        except Exception as e:
            self.status_bar.showMessage(f"DAQ initialization failed: {e}")

    def update_stream(self):
        """Updates real-time data, dynamic plotting with mid-screen scrolling."""
        try:
            self.reader.read_many_sample(
                self.chunk_buffer,
                number_of_samples_per_channel=self.chunk_size,
                timeout=1.0
            )
        except Exception as e:
            self.status_bar.showMessage(f"DAQ Read Error: {e}")
            return

        points_to_insert = self.chunk_size
        self.total_time_elapsed += points_to_insert / self.sample_rate

        for i in range(4):
            buffer = self.time_data_buffers[i]
            end_idx = self.current_index + points_to_insert

            if self.total_time_elapsed < self.window_seconds:
                # Initially, just fill up the buffer
                buffer[self.current_index:end_idx] = self.chunk_buffer[i]
            else:
                # Shift old data left and append new data to the right
                buffer[:-points_to_insert] = buffer[points_to_insert:]  # Shift left
                buffer[-points_to_insert:] = self.chunk_buffer[i]  # Insert new data at the end

            # Define dynamic mid-screen scrolling window after 5 seconds
            start_time = max(0, self.total_time_elapsed - self.window_seconds)
            end_time = start_time + self.window_seconds

            x_vals = np.linspace(start_time, end_time, self.max_points_window)
            self.time_curves[i].setData(x_vals, buffer)

        self.current_index = (self.current_index + points_to_insert) % self.max_points_window

        window = np.hanning(self.chunk_size)

        freqs = np.fft.rfftfreq(self.chunk_size, 1 / self.sample_rate)
        try:
            freq_limit = float(self.fft_limit_input.text())
        except ValueError:
            freq_limit = self.sample_rate / 2

        mask = freqs <= freq_limit
        for i in range(4):
            fft_vals = np.abs(np.fft.rfft(self.chunk_buffer[i] * window))
            self.fft_curves[i].setData(freqs[mask], fft_vals[mask])

        ref = self.chunk_buffer[0]
        for i, axis_data in enumerate(self.chunk_buffer[1:]):
            corr = correlate(axis_data, ref, mode='full')
            lags = np.arange(-len(ref) + 1, len(ref)) / self.sample_rate
            self.corr_curves[i].setData(lags, corr)

    def stop_stream(self):
        self.timer.stop()
        if self.daq_task:
            try:
                self.daq_task.stop()
                self.daq_task.close()
                self.status_bar.showMessage("Streaming stopped and resources released.")
            except Exception as e:
                self.status_bar.showMessage(f"Error closing DAQ: {e}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = VibrationLiveStream()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
