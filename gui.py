"""
Interface gráfica principal usando PyQt5.
"""

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import plotting
import acquisition
import database


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coleta Modal Experimental")
        self.layout = QVBoxLayout()

        self.test_button = QPushButton("Testar Sinal")
        self.start_button = QPushButton("Iniciar Aquisição")
        self.stop_button = QPushButton("Parar Aquisição")

        self.layout.addWidget(self.test_button)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self.ax_time = self.fig.add_subplot(311)
        self.ax_fft = self.fig.add_subplot(312)
        self.ax_corr = self.fig.add_subplot(313)

        self.setLayout(self.layout)

        self.collector = acquisition.DataCollector()
        self.collector.data_signal.connect(self.update_plot)

        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.test_button.clicked.connect(self.test_plot)

    def start_acquisition(self):
        self.collector.start()

    def stop_acquisition(self):
        self.collector.stop()
        database.save_acquisition(np.array(self.collector.buffer).T)

    def test_plot(self):
        import numpy as np
        dummy_data = np.random.randn(4, 1000)
        self.update_plot(dummy_data)

    def update_plot(self, data):
        plotting.plot_time_signal(self.ax_time, data)
        plotting.plot_fft(self.ax_fft, data, 1000)
        plotting.plot_autocorrelation(self.ax_corr, data)
        self.canvas.draw()