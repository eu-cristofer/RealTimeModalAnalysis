"""
stream_dock_widget.py

Dockable widget for real-time analysis of a single stream source.
"""
import collections

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel, QHBoxLayout
from PyQt6.QtCore import QTimer
from ui.components.chart_widget import ChartWidget
from ui.components.fft_chart_widget import FFTChartWidget
from core.analyzer import compute_fft
from core.interfaces.stream_interface import IStreamSource

import numpy as np
import pyqtgraph as pg


class StreamDockWidget(QWidget):
    """
    A dockable UI widget that shows full real-time analysis from a data stream.

    Layout:
    ┌──────────── Tabs ───────────┐
    │ ▸ All FFT      ▸ FFT (dB)   │
    └─────────────────────────────┘
    Time-domain view   | Correlation view
    """

    def __init__(self, stream: IStreamSource, title="Stream Viewer", parent=None):
        super().__init__(parent)
        self.stream = stream
        self.channel_labels = stream.get_channel_labels()
        self.channel_count = stream.get_channel_count()

        self.setWindowTitle(title)

        # Rolling buffer for time-domain view
        self.buffer_size = 1000 * 30  # 30 seconds at 1000 Hz
        self.time_data = [collections.deque(maxlen=self.buffer_size) for _ in range(self.channel_count)]
        self.time_x = collections.deque(maxlen=self.buffer_size)
        self.t_elapsed = 0

        
        self._init_ui()
        self._init_timer()

    def _init_ui(self):
        layout = QVBoxLayout()

        # FFT tabs
        self.fft_tabs = QTabWidget()
        self.fft_all_chart = FFTChartWidget("FFT (Linear)")
        self.fft_db_chart = FFTChartWidget("FFT (dB)")

        self.fft_tabs.addTab(self.fft_all_chart, "All Channels (Linear)")
        self.fft_tabs.addTab(self.fft_db_chart, "Channel 1 (dB)")

        layout.addWidget(self.fft_tabs)

        # Time + Correlation
        mid_layout = QHBoxLayout()
        self.time_chart = ChartWidget("Time-Domain Signal")
        self.corr_chart = pg.PlotWidget(title="Cross-Correlation")
        self.corr_chart.showGrid(x=True, y=True)
        self.corr_chart.addLegend()
        self.corr_chart.setLabel("bottom", "Lag (s)")
        self.corr_chart.setLabel("left", "Correlation")

        mid_layout.addWidget(self.time_chart)
        mid_layout.addWidget(self.corr_chart)

        layout.addLayout(mid_layout)
        self.setLayout(layout)

    def _init_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._update)
        self.timer.start()

    def _update(self):
        data = self.stream.read_chunk(1000)  # shape: (channels, 1000)

        # Rolling buffer update
        dt = data.shape[1] / 1000  # duration of current chunk in seconds
        self.t_elapsed += dt
        step = dt / data.shape[1]
        new_x = np.linspace(self.t_elapsed - dt, self.t_elapsed, data.shape[1])
        self.time_x.extend(new_x)

        for i in range(self.channel_count):
            self.time_data[i].extend(data[i])

        safe_data = {}
        for i in range(self.channel_count):
            try:
                x = np.array(self.time_x, dtype=np.float32)
                y = np.array(self.time_data[i], dtype=np.float32)

                if x.ndim != 1 or y.ndim != 1:
                    print(f"[WARN] Skipping ch {i}: x.ndim={x.ndim}, y.ndim={y.ndim}")
                    continue

                if len(x) != len(y):
                    print(f"[WARN] Length mismatch ch {i}: len(x)={len(x)}, len(y)={len(y)}")
                    continue

                if len(x) == 0:
                    print(f"[WARN] Empty buffer ch {i}")
                    continue

                safe_data[i] = (x, y)

            except Exception as e:
                print(f"[ERROR] While preparing channel {i}: {e}")
        self.time_chart.plot_multiple(safe_data)




        # Update FFT
        freqs = np.fft.rfftfreq(data.shape[1], 1 / 1000)
        self.fft_all_chart.plot_widget.clear()
        for i in range(self.channel_count):
            spectrum = np.abs(np.fft.rfft(data[i]))
            self.fft_all_chart.plot_widget.plot(freqs, spectrum, pen=pg.intColor(i), name=self.channel_labels[i])



        db_mag = 20 * np.log10(np.abs(np.fft.rfft(data[0])) + 1e-6)
        self.fft_db_chart.update_spectrum(freqs, db_mag)

        # Correlation (ch1–3 vs ch0)
        ref = data[0]
        self.corr_chart.clear()
        for i in range(1, self.channel_count):
            corr = np.correlate(data[i], ref, mode='full')
            lags = np.arange(-len(ref) + 1, len(ref))
            self.corr_chart.plot(lags, corr, pen=pg.intColor(i), name=f"{self.channel_labels[i]} vs {self.channel_labels[0]}")
