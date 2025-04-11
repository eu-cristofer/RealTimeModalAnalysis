"""
fft_chart_widget.py

A real-time FFT spectrum viewer using PyQtGraph.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout
from .themed_plot_widget import ThemedPlotWidget
from utils.theme_manager import theme_manager

class FFTChartWidget(ThemedPlotWidget):
    """
    Widget for displaying live FFT spectrum.
    """

    def __init__(self, title="FFT Spectrum", parent=None):
        super().__init__(parent)
        self._init_ui(title)

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