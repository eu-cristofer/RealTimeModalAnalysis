import sys
import sqlite3
import numpy as np
import nidaqmx
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer
from scipy.fftpack import fft
from scipy.signal import correlate

class SensorConfig:
    """
    Configuration class for the sensor.
    """
    def __init__(self, channel="cDAQ9181-1FC6921Mod1/ai0", sampling_rate=1000, num_samples=1024):
        """
        Initializes sensor configuration.
        
        Parameters
        ----------
        channel : str
            The channel name for the sensor acquisition.
        sampling_rate : int
            Sampling rate in Hz.
        num_samples : int
            Number of samples per acquisition.
        """
        self.channel = channel
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples

class DatabaseHandler:
    """
    Handles database operations for storing signal data.
    """
    def __init__(self, db_name="modal_analysis.db"):
        """
        Initializes the database connection and creates the table if needed.
        """
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                signal BLOB
            )
        """)
        self.conn.commit()
    
    def save_signal(self, data):
        """
        Saves a signal in the database.
        
        Parameters
        ----------
        data : np.ndarray
            The signal data to be stored.
        """
        self.cursor.execute("INSERT INTO signals (signal) VALUES (?)", (data.tobytes(),))
        self.conn.commit()
    
    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()

class ModalAnalysisApp(QMainWindow):
    """
    Main application for modal analysis with real-time signal visualization.
    """
    def __init__(self, sensor_config):
        """
        Initializes the GUI, DAQ, and database handler.
        """
        super().__init__()
        self.sensor_config = sensor_config
        self.db_handler = DatabaseHandler()
        self.initUI()
        self.initDAQ()
        self.data_accumulated = []
        self.acquiring = False
    
    def initUI(self):
        """
        Initializes the graphical user interface.
        """
        self.setWindowTitle("Análise Modal Experimental")
        self.setGeometry(100, 100, 1000, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.plot_time = pg.PlotWidget(title="Sinal no Tempo")
        self.plot_freq = pg.PlotWidget(title="Espectro de Frequência")
        self.plot_corr = pg.PlotWidget(title="Função de Correlação")
        
        self.start_button = QPushButton("Start Acquisition")
        self.start_button.clicked.connect(self.toggle_acquisition)
        
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.plot_time)
        self.layout.addWidget(self.plot_freq)
        self.layout.addWidget(self.plot_corr)
        
        self.curve_time = self.plot_time.plot()
        self.curve_freq = self.plot_freq.plot()
        self.curve_corr = self.plot_corr.plot()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
    
    def initDAQ(self):
        """
        Initializes the data acquisition system.
        """
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_accel_chan(self.sensor_config.channel)
        self.task.timing.cfg_samp_clk_timing(self.sensor_config.sampling_rate, samps_per_chan=self.sensor_config.num_samples)
    
    def toggle_acquisition(self):
        """
        Toggles the acquisition process on and off.
        """
        if self.acquiring:
            self.timer.stop()
            self.start_button.setText("Start Acquisition")
        else:
            self.timer.start(100)  # Updates every 100ms
            self.start_button.setText("Stop Acquisition")
        self.acquiring = not self.acquiring
    
    def update_plot(self):
        """
        Updates the signal plots with new data.
        """
        data = self.task.read(number_of_samples_per_channel=self.sensor_config.num_samples)
        data = np.array(data)
        
        # Accumulate time-domain data
        self.data_accumulated.extend(data)
        self.curve_time.setData(self.data_accumulated)
        
        # Save to database
        self.db_handler.save_signal(data)
        
        # Compute FFT and limit to 100Hz
        freqs = np.fft.fftfreq(self.sensor_config.num_samples, 1/self.sensor_config.sampling_rate)
        freq_data = np.abs(fft(data))[:self.sensor_config.num_samples // 2]
        valid_indices = freqs[:self.sensor_config.num_samples // 2] <= 100
        self.curve_freq.setData(freqs[valid_indices], freq_data[valid_indices])
        
        # Compute correlation and update correlation plot
        corr_data = correlate(data, data, mode='full')
        self.curve_corr.setData(corr_data)
    
    def closeEvent(self, event):
        """
        Handles the close event by shutting down the DAQ task and database.
        """
        self.task.close()
        self.db_handler.close()
        event.accept()
        
if __name__ == "__main__":
    sensor_config = SensorConfig()
    app = QApplication(sys.argv)
    window = ModalAnalysisApp(sensor_config)
    window.show()
    sys.exit(app.exec_())
