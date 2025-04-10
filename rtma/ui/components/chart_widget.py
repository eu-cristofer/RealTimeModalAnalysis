"""
chart_widget.py

Widget for plotting real-time signals using PyQtGraph.
"""

from utils.theme_manager import get_current_theme
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np


class ChartWidget(QWidget):
    """
    Chart widget for plotting real-time signals.

    Supports multi-channel signal plotting for comparative visualization.
    """

    def __init__(self, title="Signal", parent=None):
        super().__init__(parent)
        self.title = title
        self._init_ui()
        self.set_theme(get_current_theme())
        self.channel_curves = {}

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


    def plot_multiple(self, signal_dict: dict):
        """
        Plot multiple signals on the same chart.

        Parameters
        ----------
        signal_dict : dict[int, np.ndarray]
            Dictionary mapping channel index to its signal array.
        """
        # color_list = ['r', 'g', 'b', 'y']
        color_list = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
        t = np.linspace(0, 0.02, 1000)

        # Update or create a plot curve for each channel
        for ch, signal in signal_dict.items():
            if ch not in self.channel_curves:
                self.channel_curves[ch] = self.plot_widget.plot(pen=color_list[ch % len(color_list)])
            self.channel_curves[ch].setData(t, signal)

        # Hide unused curves
        for ch in list(self.channel_curves.keys()):
            if ch not in signal_dict:
                self.channel_curves[ch].hide()
            else:
                self.channel_curves[ch].show()
