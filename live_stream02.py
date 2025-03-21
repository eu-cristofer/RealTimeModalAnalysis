
# -*- coding: utf-8 -*-
"""
Real-time vibration live streaming GUI with PyQt5 and pyqtgraph.

Extended Features:
- Allows saving the last N seconds of data (specified by the user) to an SQLite database.
- The SQLite table includes an absolute UNIX timestamp for each sample for accurate time tracking.
- Database filenames are created with a timestamp to avoid overwriting.
- Interface includes configurable fields for sample rate and save duration.

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
import sqlite3
import time


class VibrationLiveStream(QWidget):
    """Main window class for real-time vibration live streaming GUI with save-to-SQLite feature."""

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

        # User input for save duration and button to trigger SQLite saving
        self.save_duration_label = QLabel("Save Last N Seconds:")
        self.save_duration_input = QLineEdit("5")  # Default save of last 5 seconds
        self.save_button = QPushButton("Save to SQLite")
        self.save_button.clicked.connect(self.save_to_sqlite)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.fft_limit_label)
        button_layout.addWidget(self.fft_limit_input)
        button_layout.addWidget(self.save_duration_label)
        button_layout.addWidget(self.save_duration_input)
        button_layout.addWidget(self.save_button)

        self.layout.addLayout(button_layout)

        self.status_bar = QStatusBar()
        self.layout.addWidget(self.status_bar)

        self.setup_plots()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stream)

        # Acquisition parameters
        self.sample_rate = 1000  # Hz (default)
        self.chunk_size = 256    # Samples per read
        self.window_seconds = 10  # 10-second rolling buffer
        self.max_points_window = int(self.sample_rate * self.window_seconds)
        self.total_time_elapsed = 0

        # Buffers for each channel
        self.time_data_buffers = [np.zeros(self.max_points_window) for _ in range(4)]
        self.current_index = 0
        self.daq_task = None

    def setup_plots(self):
        # Plot setup with time domain, FFT, and cross-correlation
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
        """Start DAQ live stream with real-time visualization."""
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
                samps_per_chan=self.chunk_size * 40
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
        """Update plots with new streaming data."""
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
                buffer[self.current_index:end_idx] = self.chunk_buffer[i]
            else:
                buffer[:-points_to_insert] = buffer[points_to_insert:]
                buffer[-points_to_insert:] = self.chunk_buffer[i]

            start_time = max(0, self.total_time_elapsed - self.window_seconds)
            end_time = start_time + self.window_seconds
            x_vals = np.linspace(start_time, end_time, self.max_points_window)
            self.time_curves[i].setData(x_vals, buffer)

        self.current_index = (self.current_index + points_to_insert) % self.max_points_window

        window = np.hanning(self.chunk_size)
        freqs = np.fft.rfftfreq(self.chunk_size, 1 / self.sample_rate)
        freq_limit = float(self.fft_limit_input.text())
        mask = freqs <= freq_limit

        for i in range(4):
            fft_vals = np.abs(np.fft.rfft(self.chunk_buffer[i] * window))
            self.fft_curves[i].setData(freqs[mask], fft_vals[mask])

        ref = self.chunk_buffer[0]
        for i, axis_data in enumerate(self.chunk_buffer[1:]):
            corr = correlate(axis_data, ref, mode='full')
            lags = np.arange(-len(ref) + 1, len(ref)) / self.sample_rate
            self.corr_curves[i].setData(lags, corr)

    def save_to_sqlite(self):
        """Save the last N seconds of buffered data into a timestamped SQLite file."""
        try:
            duration = float(self.save_duration_input.text())
            points_to_save = int(duration * self.sample_rate)

            if points_to_save > self.max_points_window:
                self.status_bar.showMessage(f"Cannot save {duration}s. Buffer only holds {self.window_seconds}s.")
                return

            end_idx = self.current_index
            start_idx = end_idx - points_to_save if end_idx - points_to_save >= 0 else self.max_points_window + (end_idx - points_to_save)

            data_segments = []
            for i in range(4):
                buf = self.time_data_buffers[i]
                if start_idx < end_idx:
                    data = buf[start_idx:end_idx]
                else:
                    data = np.concatenate((buf[start_idx:], buf[:end_idx]))
                data_segments.append(data)

            # Generate absolute timestamps for each sample using current time
            save_end_time = time.time()
            timestamps = np.linspace(save_end_time - duration, save_end_time, points_to_save)

            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            db_name = f"vibration_data_{timestamp_str}.sqlite"
            conn = sqlite3.connect(db_name)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS vibration_samples (
                            abs_timestamp REAL,
                            reference REAL,
                            x REAL,
                            y REAL,
                            z REAL
                         )""")

            for i in range(points_to_save):
                c.execute("INSERT INTO vibration_samples VALUES (?, ?, ?, ?, ?)", (
                    timestamps[i], data_segments[0][i], data_segments[1][i], data_segments[2][i], data_segments[3][i]
                ))

            conn.commit()
            conn.close()
            self.status_bar.showMessage(f"Saved {duration}s of data to {db_name}.")
        except Exception as e:
            self.status_bar.showMessage(f"SQLite save error: {e}")

    def stop_stream(self):
        """Stop the data stream and clean up resources."""
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
