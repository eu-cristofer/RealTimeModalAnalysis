# -*- coding: utf-8 -*-
"""
Real-time vibration live streaming GUI with PyQt5 and pyqtgraph.

- Continuously streams data from 4 channels (1 reference + 3-axis accelerometer).
- Displays:
    - Dynamically updating time-domain plots (streaming, x-axis auto-scroll)
    - FFT spectra (updated every cycle, limited frequencies)
    - Cross-correlation plots (updated every cycle)
- Provides Start and Stop buttons.

Author: Cristofer (tutored by GPT)
Date: March 2025
"""

import sys
import numpy as np
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, AccelUnits, ExcitationSource
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from scipy.signal import correlate
from config import CHANNELS


class VibrationLiveStream(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Vibration Live Stream")
        
        self.layout = QVBoxLayout()

        # Start and Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Live Stream")
        self.stop_button = QPushButton("Stop Live Stream")
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        self.layout.addLayout(button_layout)

        # Plot widgets
        self.time_plot = pg.PlotWidget(title="Live Time-Domain Signals")
        self.fft_plot = pg.PlotWidget(title="FFT Spectra (Limited Frequency Range)")
        self.corr_plot = pg.PlotWidget(title="Cross-Correlation")

        self.layout.addWidget(self.time_plot)
        self.layout.addWidget(self.fft_plot)
        self.layout.addWidget(self.corr_plot)

        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stream)

        self.sample_rate = 1000
        self.chunk_size = 256
        self.max_time_points = 5000

        self.time_data_buffers = [np.zeros(self.max_time_points) for _ in range(4)]
        self.current_index = 0
        self.daq_task = None

    def start_stream(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

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

        self.time_curves = [self.time_plot.plot(pen=pg.intColor(i)) for i in range(4)]
        self.fft_curves = [self.fft_plot.plot(pen=pg.intColor(i)) for i in range(4)]
        self.corr_curves = [self.corr_plot.plot(pen=pg.intColor(i)) for i in range(3)]

        self.timer.start(50)

    def update_stream(self):
        try:
            self.reader.read_many_sample(
                self.chunk_buffer,
                number_of_samples_per_channel=self.chunk_size,
                timeout=1.0
            )
        except Exception as e:
            print(f"DAQ Read Error: {e}")
            return

        # Update live time-domain buffers
        for i in range(4):
            buffer = self.time_data_buffers[i]
            points_to_insert = self.chunk_size
            end_index = self.current_index + points_to_insert

            if end_index <= self.max_time_points:
                buffer[self.current_index:end_index] = self.chunk_buffer[i]
            else:
                part1_len = self.max_time_points - self.current_index
                buffer[self.current_index:] = self.chunk_buffer[i][:part1_len]
                buffer[:points_to_insert - part1_len] = self.chunk_buffer[i][part1_len:]

            self.time_curves[i].setData(
                np.arange(len(buffer)) / self.sample_rate, buffer
            )

        self.current_index = (self.current_index + points_to_insert) % self.max_time_points

        # FFT plots limited to half Nyquist (e.g., 0 to fs/4)
        freqs = np.fft.rfftfreq(self.chunk_size, 1 / self.sample_rate)
        freq_limit = self.sample_rate / 4
        freq_mask = freqs <= freq_limit

        for i in range(4):
            fft_vals = np.abs(np.fft.rfft(self.chunk_buffer[i]))
            self.fft_curves[i].setData(freqs[freq_mask], fft_vals[freq_mask])

        # Cross-correlation plots
        ref = self.chunk_buffer[0]
        for i, axis_data in enumerate(self.chunk_buffer[1:]):
            corr = correlate(axis_data, ref, mode='full')
            lags = np.arange(-len(ref) + 1, len(ref)) / self.sample_rate
            self.corr_curves[i].setData(lags, corr)

    def stop_stream(self):
        self.timer.stop()
        if self.daq_task:
            self.daq_task.stop()
            self.daq_task.close()
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
