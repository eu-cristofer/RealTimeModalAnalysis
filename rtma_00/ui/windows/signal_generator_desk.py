import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QMainWindow, QLabel,
    QScrollArea, QStatusBar, QPushButton, QTabWidget
)
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtCore import QSize, Qt, QTimer
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from rtma.ui.windows.signal_channel_control import SignalChannelControl
from rtma.ui.themes.theme_manager import ThemeManager
from rtma.utils.log_setup import setup_logging

logger = setup_logging("SignalGenerator")

class WaveformPreviewWindow(QWidget):
    def __init__(self, channels, theme='dark'):
        super().__init__()
        self.setWindowTitle("🎶 Waveform Preview")
        self.resize(900, 500)
        self.theme = theme
        self.channels = channels
        self.buffers_x = [[] for _ in range(4)]
        self.buffers_y = [[] for _ in range(4)]
        self.t = 0
        self.color_schemes = {
            'dark': ['#FF4C4C', '#4CFF4C', '#4C4CFF', '#FFFF4C'],
            'light': ['#C00000', '#00A000', '#0000A0', '#A0A000']
        }

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setStyleSheet("QTabWidget::pane { border: 0; background: transparent; }")

        self.time_tab = QWidget()
        self.fft_tab = QWidget()
        self.tabs.addTab(self.time_tab, "Time Domain")
        self.tabs.addTab(self.fft_tab, "Frequency Domain")

        # Time Plot
        self.time_plot = PlotWidget(title="Signal Waveforms")
        self.time_plot.addLegend()
        self.time_layout = QVBoxLayout(self.time_tab)
        self.time_layout.setContentsMargins(0, 0, 0, 0)
        self.time_layout.addWidget(self.time_plot)
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setLabel('bottom', 'Time', units='s')

        # FFT Plot
        self.fft_plot = PlotWidget(title="FFT Spectrum")
        self.fft_plot.addLegend()
        self.fft_layout = QVBoxLayout(self.fft_tab)
        self.fft_layout.setContentsMargins(0, 0, 0, 0)
        self.fft_layout.addWidget(self.fft_plot)
        self.fft_plot.setLabel('left', 'Magnitude')
        self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')

        self.curves = []
        self.fft_curves = []

        colors = self.color_schemes[self.theme]
        for i, label in enumerate(["CH1", "CH2", "CH3", "CH4"]):
            c = pg.mkPen(colors[i], width=2)
            self.curves.append(self.time_plot.plot(pen=c, name=label))
            self.fft_curves.append(self.fft_plot.plot(pen=c, name=label))

        self.apply_theme(self.theme)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

    def update_plot(self):
        self.t += 0.1
        for i, ch in enumerate(self.channels):
            val = ch.get_waveform_value(self.t, ch.amp if ch.running else 0)
            self.buffers_x[i].append(self.t)
            self.buffers_y[i].append(val)
            if len(self.buffers_x[i]) > 200:
                self.buffers_x[i].pop(0)
                self.buffers_y[i].pop(0)

            self.curves[i].setData(self.buffers_x[i], self.buffers_y[i])

            if len(self.buffers_y[i]) >= 32:
                fft = np.fft.fft(self.buffers_y[i])
                freqs = np.fft.fftfreq(len(self.buffers_y[i]), d=0.1)
                mask = freqs >= 0
                self.fft_curves[i].setData(freqs[mask], np.abs(fft[mask]))

    def apply_theme(self, theme):
        self.theme = theme
        bg = '#1e1e1e' if theme == 'dark' else 'w'
        self.time_plot.setBackground(bg)
        self.fft_plot.setBackground(bg)
        colors = self.color_schemes[theme]
        for i, curve in enumerate(self.curves):
            curve.setPen(pg.mkPen(colors[i], width=2))
            self.fft_curves[i].setPen(pg.mkPen(colors[i], width=2))


class SignalGeneratorDesk(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing Signal Generator Desk UI...")

        self.setWindowTitle("🎧 RTMA - Signal Generator")
        self.resize(960, 600)

        self.theme_manager = ThemeManager("dark")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.toolbar = QToolBar("Toolbar")
        self.toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(self.toolbar)

        self.theme_toggle = QAction("🌙 Theme", self)
        self.theme_toggle.triggered.connect(self.toggle_theme)
        self.toolbar.addAction(self.theme_toggle)

        self.start_all_btn = QPushButton("▶")
        self.start_all_btn.setFixedSize(32, 32)
        self.start_all_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        self.start_all_btn.clicked.connect(self.start_all_channels)
        self.toolbar.addWidget(self.start_all_btn)

        self.stop_all_btn = QPushButton("■")
        self.stop_all_btn.setFixedSize(32, 32)
        self.stop_all_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.stop_all_btn.clicked.connect(self.stop_all_channels)
        self.toolbar.addWidget(self.stop_all_btn)

        self.preview_window = None
        self.preview_btn = QPushButton("👁 Preview")
        self.preview_btn.setToolTip("Open waveform preview window")
        self.preview_btn.setFixedSize(90, 32)
        self.preview_btn.clicked.connect(self.open_preview_window)
        self.toolbar.addWidget(self.preview_btn)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.channel_container = QWidget()
        self.channel_layout = QHBoxLayout()
        self.channel_container.setLayout(self.channel_layout)

        self.channels = []
        for i in range(4):
            ch = SignalChannelControl(f"Channel {i+1}")
            self.channels.append(ch)
            self.channel_layout.addWidget(ch)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.channel_container)
        self.main_layout.addWidget(scroll_area)

        self.theme_manager.apply_theme(self)
        logger.info("Signal Generator Desk initialized")

    def open_preview_window(self):
        if self.preview_window is None or not self.preview_window.isVisible():
            self.preview_window = WaveformPreviewWindow(self.channels, self.theme_manager.theme_name)
            self.preview_window.show()
            logger.info("Opened waveform preview window")

    def start_all_channels(self):
        for ch in self.channels:
            ch.start_channel()
        self.status_bar.showMessage("All channels started.")
        logger.info("All channels started")

    def stop_all_channels(self):
        for ch in self.channels:
            ch.stop_channel()
        self.status_bar.showMessage("All channels stopped.")
        logger.info("All channels stopped")

    def toggle_theme(self):
        current = self.theme_manager.theme_name
        new_theme = "light" if current == "dark" else "dark"
        self.theme_manager = ThemeManager(new_theme)
        self.theme_manager.apply_theme(self)
        self.status_bar.showMessage(f"Theme: {new_theme}")
        if self.preview_window and self.preview_window.isVisible():
            self.preview_window.apply_theme(new_theme)
        logger.info(f"Theme changed to {new_theme}")


if __name__ == "__main__":
    logger.info("Launching application...")
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = SignalGeneratorDesk()
    window.show()
    sys.exit(app.exec())
