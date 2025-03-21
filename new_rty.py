import sys
import sqlite3
import numpy as np
import nidaqmx
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLineEdit, QHBoxLayout
from PyQt5.QtCore import QTimer
from scipy.fftpack import fft
from scipy.signal import correlate


class DAQHandler:
    """Handles DAQ data acquisition."""

    def __init__(self, sampling_rate=1000, num_samples=1024):
        """
        Initialize the DAQ system.

        Parameters
        ----------
        sampling_rate : int
            Sampling rate in Hz.
        num_samples : int
            Number of samples per acquisition.
        """
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples
        self.task = None

        try:
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_accel_chan("cDAQ9181-1FC6921Mod1/ai0")
            self.task.timing.cfg_samp_clk_timing(self.sampling_rate, samps_per_chan=self.num_samples)
        except nidaqmx.DaqError as e:
            print(f"⚠️ Erro ao inicializar DAQ: {e}")
            self.task = None

    def read_data(self):
        """Reads a batch of samples from the DAQ device."""
        if self.task:
            return np.array(self.task.read(number_of_samples_per_channel=self.num_samples), dtype=np.float32)
        return np.zeros(self.num_samples, dtype=np.float32)

    def close(self):
        """Closes the DAQ task safely."""
        if self.task:
            self.task.close()


class DatabaseHandler:
    """Handles database operations."""

    def __init__(self, db_name="modal_analysis.db"):
        """
        Initialize the database.

        Parameters
        ----------
        db_name : str
            Name of the database file.
        """
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_table()
        self.data_buffer = np.empty((0, 1024), dtype=np.float32)

    def _create_table(self):
        """Creates the signals table if it does not exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                signal BLOB
            )
        """)
        self.conn.commit()

    def buffer_data(self, data):
        """Buffers data and writes to the database in batches."""
        self.data_buffer = np.vstack((self.data_buffer, data))

        if len(self.data_buffer) >= 10:  # Write every 10 updates (~1 sec)
            self.cursor.executemany("INSERT INTO signals (signal) VALUES (?)",
                                    [(d.tobytes(),) for d in self.data_buffer])
            self.conn.commit()
            self.data_buffer = np.empty((0, 1024), dtype=np.float32)  # Clear buffer

    def close(self):
        """Closes the database connection."""
        self.conn.close()


class ModalAnalysisApp(QMainWindow):
    """Main application for real-time modal analysis."""

    def __init__(self):
        """Initialize the application, UI, DAQ, and database."""
        super().__init__()

        # Initialize components
        self.daq = DAQHandler()
        self.db = DatabaseHandler()

        self.initUI()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # Update every 100ms

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle("Análise Modal Experimental")
        self.setGeometry(100, 100, 1000, 600)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Create plots
        self.plot_time = pg.PlotWidget(title="Sinal no Tempo")
        self.plot_freq = pg.PlotWidget(title="Espectro de Frequência")
        self.plot_corr = pg.PlotWidget(title="Função de Correlação")

        self.layout.addWidget(self.plot_time)
        self.layout.addWidget(self.plot_freq)
        self.layout.addWidget(self.plot_corr)

        # Custom colors for plots
        self.curve_time = self.plot_time.plot(pen='g')  # Green for time signal
        self.curve_freq = self.plot_freq.plot(pen='r')  # Red for FFT
        self.curve_corr = self.plot_corr.plot(pen='b')  # Blue for correlation

        # Frequency Range Controls
        self.freq_range_layout = QHBoxLayout()
        self.freq_input = QLineEdit()
        self.freq_input.setPlaceholderText("Digite o limite de frequência (Hz)")

        self.apply_button = QPushButton("Aplicar")
        self.apply_button.clicked.connect(self.apply_freq_range)

        self.reset_button = QPushButton("Resetar")
        self.reset_button.clicked.connect(self.reset_freq_range)

        self.freq_range_layout.addWidget(self.freq_input)
        self.freq_range_layout.addWidget(self.apply_button)
        self.freq_range_layout.addWidget(self.reset_button)
        self.layout.addLayout(self.freq_range_layout)

        self.freq_limit = None  # Default: No limit

    def apply_freq_range(self):
        """Apply the user-defined frequency range to the FFT plot."""
        try:
            self.freq_limit = float(self.freq_input.text())
            self.plot_freq.setXRange(0, self.freq_limit)
        except ValueError:
            self.freq_input.clear()
            print("⚠️ Erro: Insira um número válido para a frequência.")

    def reset_freq_range(self):
        """Reset the frequency range to the full available range."""
        self.freq_limit = None
        self.plot_freq.setXRange(0, self.daq.sampling_rate / 2)
        self.freq_input.clear()

    def update_plot(self):
        """Update the plots with new data from the DAQ system."""
        data = self.daq.read_data()
        self.db.buffer_data(data)

        # Update time-domain plot
        self.curve_time.setData(data)

        # Compute FFT
        freq_data = np.abs(fft(data))[:self.daq.num_samples // 2]
        freq_axis = np.linspace(0, self.daq.sampling_rate / 2, self.daq.num_samples // 2)

        # Apply frequency limit
        if self.freq_limit:
            mask = freq_axis <= self.freq_limit
            freq_data = freq_data[mask]
            freq_axis = freq_axis[mask]

        self.curve_freq.setData(freq_axis, freq_data)

        # Compute correlation function
        corr_data = correlate(data, data, mode='full')
        corr_x = np.arange(-len(data) + 1, len(data))
        self.curve_corr.setData(corr_x, corr_data)

    def closeEvent(self, event):
        """Handle application closure, ensuring resources are released."""
        self.daq.close()
        self.db.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModalAnalysisApp()
    window.show()
    sys.exit(app.exec_())
