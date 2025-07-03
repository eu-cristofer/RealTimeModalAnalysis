"""
chart_widget.py

Widget for plotting real-time signals using PyQtGraph.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout
from rtma.ui.components.themed_plot_widget import ThemedPlotWidget



class ChartWidget(ThemedPlotWidget):
    """
    Chart widget for plotting real-time signals.

    Supports multi-channel signal plotting for comparative visualization.
    """

    def __init__(self, title="Signal", parent=None):
        super().__init__(parent)
        self.title = title
        self.channel_curves = {}
        self._init_ui()
        

    def _init_ui(self):
        layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget(title=self.title)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', "Amplitude")
        self.plot_widget.setLabel('bottom', "Time", units='s')
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def update_data(self, data: np.ndarray):
        """
        Update single signal (fallback/compatibility).
        """
        if "main" not in self.channel_curves:
            self.channel_curves["main"] = self.plot_widget.plot(pen='y')
        self.channel_curves["main"].setData(data)

    
    def plot_multiple(self, data_dict):
        for ch, value in data_dict.items():
            if ch not in self.channel_curves:
                self.channel_curves[ch] = self.plot_widget.plot(pen=pg.intColor(ch))

            # Handle two formats: (y_array) OR (x_array, y_array)
            if isinstance(value, tuple) and len(value) == 2:
                t, signal = value
            else:
                signal = np.array(value)
                t = np.linspace(0, len(signal) / 1000, len(signal))  # Assume 1kHz for synthetic

            if t.ndim != 1 or signal.ndim != 1 or len(t) != len(signal):
                print(f"[SKIP] Invalid data for channel {ch}")
                continue

            self.channel_curves[ch].setData(t, signal)

