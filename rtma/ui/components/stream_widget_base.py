from abc import ABC, abstractmethod
import numpy as np

class IStreamSource(ABC):
    """
    Interface for any stream source (SG or NI).
    """

    @abstractmethod
    def read_chunk(self, length: int) -> np.ndarray:
        """
        Read a chunk of samples from the stream.

        Parameters
        ----------
        length : int
            Number of samples per channel to read.

        Returns
        -------
        np.ndarray
            Array of shape (channels, length)
        """
        pass

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from ui.components.chart_widget import ChartWidget
from ui.components.fft_chart_widget import FFTChartWidget
from core.analyzer import compute_fft
from PyQt6.QtCore import QTimer


class BaseStreamWidget(QWidget):
    """
    Abstract base stream UI widget that displays time and FFT data from a stream source.
    """

    def __init__(self, stream_source: IStreamSource, title: str = "Stream", parent=None):
        super().__init__(parent)
        self.stream_source = stream_source
        self.setWindowTitle(title)

        self._init_ui()

        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._update)
        self.timer.start()

    def _init_ui(self):
        self.layout = QVBoxLayout()
        self.rms_label = QLabel("RMS: 0.000")

        self.time_chart = ChartWidget("Time-Domain Signal")
        self.fft_chart = FFTChartWidget("FFT Spectrum")

        self.layout.addWidget(self.rms_label)
        self.layout.addWidget(self.time_chart)
        self.layout.addWidget(self.fft_chart)
        self.setLayout(self.layout)

    def _update(self):
        data = self.stream_source.read_chunk(1000)
        self.time_chart.update_data(data[0])
        freqs, mag = compute_fft(data[0], 1000)
        self.fft_chart.update_spectrum(freqs, mag)

        rms = np.sqrt(np.mean(data[0] ** 2))
        self.rms_label.setText(f"RMS: {rms:.3f}")
