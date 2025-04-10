"""
fft_chart_widget.py

A real-time FFT spectrum viewer using PyQtGraph.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from utils.theme_manager import get_current_theme
import pyqtgraph as pg
import numpy as np


class FFTChartWidget(QWidget):
    """
    Widget for displaying live FFT spectrum.
    """

    def __init__(self, title="FFT Spectrum", parent=None):
        super().__init__(parent)
        self._init_ui(title)
        self.set_theme(get_current_theme())

    def _init_ui(self, title):
        layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget(title=title)
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLabel('left', 'Magnitude (dB)')
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.showGrid(x=True, y=True)

        self.curve = self.plot_widget.plot(pen='c')  # cyan for visibility
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def update_spectrum(self, freqs: np.ndarray, magnitude: np.ndarray):
        self.curve.setData(freqs, magnitude)

    def set_theme(self, theme: str):
        """
        Adjust chart colors based on the current theme.

        Parameters
        ----------
        theme : str
            Either "dark" or "light"
        """
        if theme == "light":
            self.plot_widget.setBackground("w")
            self.plot_widget.getAxis("left").setPen("black")
            self.plot_widget.getAxis("bottom").setPen("black")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        else:
            self.plot_widget.setBackground("k")
            self.plot_widget.getAxis("left").setPen("white")
            self.plot_widget.getAxis("bottom").setPen("white")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
