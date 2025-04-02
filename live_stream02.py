# -*- coding: utf-8 -*-
"""
Improved Real-time Vibration GUI for Accelerometers and Proximity Probes

Features:
- User-selectable mode: Accelerometer or Proximity Probe (3300 XL 8 mm)
- Real-time time-domain, FFT, and cross-correlation plots
- Buffered data storage with selectable duration
- Dynamic scaling (g or µm) depending on mode
- Modular DAQ configuration and data handling

Author: Cristofer (guided by GPT)
Date: March 2025
"""

import sys
import numpy as np
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, AccelUnits, ExcitationSource, VoltageUnits
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QLineEdit, QLabel, QComboBox, QCheckBox, QStatusBar
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from scipy.signal import correlate
import time
import sqlite3
from config import CHANNELS

class VibrationLiveStream(QWidget):
    def __init__(self):
        super().__init__()

        # Inital configuration
        self.setWindowTitle("Real-Time Vibration Live Stream")
        self.sample_rate = 10000  # High rate to capture dynamic events
        self.chunk_size = 4096 # To improve fft resolution
        self.window_seconds = 15
        self.max_points_window = int(self.sample_rate * self.window_seconds)
        self.total_time_elapsed = 0
        self.current_index = 0

        self.time_data_buffers = []
        self.chunk_buffer = None
        self.daq_task = None

        self.setup_ui()
        self.setup_plots()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stream)

    def setup_ui(self):
        self.layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)

        self.fft_limit_label = QLabel("FFT Max Frequency (Hz):")
        self.fft_limit_input = QLineEdit("100")

        self.save_label = QLabel("Save Last N Seconds:")
        self.save_input = QLineEdit("5")
        self.save_button = QPushButton("Save to SQLite")
        self.save_button.clicked.connect(self.save_to_sqlite)

        self.mode_select = QComboBox()
        self.mode_select.addItems(["Accelerometer", "Proximity Probe (3300 XL)"])

        self.ac_only_checkbox = QCheckBox("Show AC (remove DC offset)")
        self.ac_only_checkbox.setChecked(True)

        for widget in [self.start_button, self.stop_button, self.fft_limit_label, self.fft_limit_input,
                       self.save_label, self.save_input, self.save_button, QLabel("Sensor Type:"),
                       self.mode_select, self.ac_only_checkbox]:
            button_layout.addWidget(widget)

        self.status_bar = QStatusBar()
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.status_bar)
        self.setLayout(self.layout)

    def setup_plots(self):
        self.time_plot = pg.PlotWidget(title="Time-Domain")
        self.time_plot.setLabel('bottom', 'Time (s)')
        self.fft_plot = pg.PlotWidget(title="FFT")
        self.fft_plot.setLabel('bottom', 'Frequency (Hz)')
        self.corr_plot = pg.PlotWidget(title="Cross-Correlation")
        self.corr_plot.setLabel('bottom', 'Lag (s)')
        for plot in [self.time_plot, self.fft_plot, self.corr_plot]:
            plot.showGrid(x=True, y=True)
            plot.addLegend()
            self.layout.addWidget(plot)

    def configure_task(self):
        self.daq_task = nidaqmx.Task()
        mode = self.mode_select.currentText()
        is_accel = mode == "Accelerometer"

        for ch in CHANNELS:
            if is_accel:
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
            else:
                self.daq_task.ai_channels.add_ai_voltage_chan(
                    physical_channel=ch,
                    terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                    min_val=-5.0,
                    max_val=5.0,
                    units=VoltageUnits.VOLTS
                )

        self.daq_task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.chunk_size # * 10
        )
        self.reader = AnalogMultiChannelReader(self.daq_task.in_stream)
        self.chunk_buffer = np.zeros((len(CHANNELS), self.chunk_size))
        self.time_data_buffers = [np.zeros(self.max_points_window) for _ in CHANNELS]

    def start_stream(self):
        try:
            self.configure_task()
            self.daq_task.start()
            self.time_curves = [self.time_plot.plot(pen=pg.intColor(i), name=f"CH{i+1}") for i in range(len(CHANNELS))]
            self.fft_curves = [self.fft_plot.plot(pen=pg.intColor(i), name=f"CH{i+1}") for i in range(len(CHANNELS))]
            self.corr_curves = [self.corr_plot.plot(pen=pg.intColor(i+1), name=f"CH{i+2} vs CH1") for i in range(len(CHANNELS)-1)]

            # self.total_time_elapsed = 0
            # self.current_index = 0
            self.timer.start(50)
            self.status_bar.showMessage("Streaming...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        except Exception as e:
            print(e)
            self.status_bar.showMessage(f"Start error: {e}")

    def update_stream(self):
        try:
            self.reader.read_many_sample(self.chunk_buffer,
                                         number_of_samples_per_channel=self.chunk_size,
                                         timeout=1.0)
        except Exception as e:
            self.status_bar.showMessage(f"Read error: {e}")
            return

        self.total_time_elapsed += self.chunk_size / self.sample_rate
        is_proximity = self.mode_select.currentText().startswith("Proximity")
        scale_factor = 127.1 if is_proximity else 1  # convert V to um if needed

        for i, buf in enumerate(self.time_data_buffers):
            if self.total_time_elapsed < self.window_seconds:
                buf[self.current_index:self.current_index+self.chunk_size] = self.chunk_buffer[i] * scale_factor
            else:
                buf[:-self.chunk_size] = buf[self.chunk_size:]
                buf[-self.chunk_size:] = self.chunk_buffer[i] * scale_factor

            x_vals = np.linspace(max(0, self.total_time_elapsed - self.window_seconds), self.total_time_elapsed, self.max_points_window)
            self.time_curves[i].setData(x_vals, buf)

        self.current_index = (self.current_index + self.chunk_size) % self.max_points_window
        window = np.hanning(self.chunk_size)
        freqs = np.fft.rfftfreq(self.chunk_size, 1 / self.sample_rate)
        fft_limit = float(self.fft_limit_input.text())
        mask = freqs <= fft_limit

        for i in range(len(CHANNELS)):
            fft_vals = np.abs(np.fft.rfft(self.chunk_buffer[i] * window))
            self.fft_curves[i].setData(freqs[mask], fft_vals[mask])

        ref = self.chunk_buffer[0]
        for i in range(1, len(CHANNELS)):
            corr = correlate(self.chunk_buffer[i], ref, mode='full')
            lags = np.arange(-len(ref)+1, len(ref)) / self.sample_rate
            self.corr_curves[i-1].setData(lags, corr)

    def stop_stream(self):
        self.timer.stop()
        if self.daq_task:
            try:
                self.daq_task.stop()
                self.daq_task.close()
            except Exception as e:
                self.status_bar.showMessage(f"Stop error: {e}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Stopped.")

    def save_to_sqlite(self):
        try:
            duration = float(self.save_input.text())
            pts = int(duration * self.sample_rate)
            end_idx = self.current_index
            start_idx = end_idx - pts if end_idx - pts >= 0 else self.max_points_window + (end_idx - pts)
            segments = []
            for buf in self.time_data_buffers:
                if start_idx < end_idx:
                    data = buf[start_idx:end_idx]
                else:
                    data = np.concatenate((buf[start_idx:], buf[:end_idx]))
                segments.append(data)
            abs_t = np.linspace(time.time() - duration, time.time(), pts)
            rel_t = np.linspace(self.total_time_elapsed - duration, self.total_time_elapsed, pts)
            db_name = f"vibdata_{time.strftime('%Y%m%d_%H%M%S')}.sqlite"
            conn = sqlite3.connect(db_name)
            c = conn.cursor()
            cols = ", ".join([f"ch{i+1} REAL" for i in range(len(CHANNELS))])
            c.execute(f"CREATE TABLE IF NOT EXISTS samples (abs_time REAL, rel_time REAL, {cols})")
            for i in range(pts):
                values = tuple(seg[i] for seg in segments)
                c.execute(f"INSERT INTO samples VALUES (?, ?, {', '.join(['?']*len(CHANNELS))})", (abs_t[i], rel_t[i], *values))
            conn.commit()
            conn.close()
            self.status_bar.showMessage(f"Saved {duration}s to {db_name}")
        except Exception as e:
            self.status_bar.showMessage(f"Save error: {e}")

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
