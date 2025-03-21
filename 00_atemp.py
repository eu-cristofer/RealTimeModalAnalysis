import sys
import sqlite3
import numpy as np
import nidaqmx
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from scipy.fftpack import fft
from scipy.signal import correlate

class ModalAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initDAQ()
        self.initDB()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # Atualiza a cada 100ms
    
    def initUI(self):
        self.setWindowTitle("Análise Modal Experimental")
        self.setGeometry(100, 100, 1000, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.plot_time = pg.PlotWidget(title="Sinal no Tempo")
        self.plot_freq = pg.PlotWidget(title="Espectro de Frequência")
        self.plot_corr = pg.PlotWidget(title="Função de Correlação")
        
        self.layout.addWidget(self.plot_time)
        self.layout.addWidget(self.plot_freq)
        self.layout.addWidget(self.plot_corr)
        
        self.curve_time = self.plot_time.plot()
        self.curve_freq = self.plot_freq.plot()
        self.curve_corr = self.plot_corr.plot()
    
    def initDAQ(self):
        self.sampling_rate = 1000  # Hz
        self.num_samples = 1024  # Número de amostras por aquisição
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_accel_chan("cDAQ9181-1FC6921Mod1/ai0")  # Ajuste conforme necessário
        self.task.timing.cfg_samp_clk_timing(self.sampling_rate, samps_per_chan=self.num_samples)
    
    def initDB(self):
        self.conn = sqlite3.connect("modal_analysis.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                signal BLOB
            )
        """)
        self.conn.commit()
    
    def update_plot(self):
        data = self.task.read(number_of_samples_per_channel=self.num_samples)
        data = np.array(data)
        
        # Salvar no banco de dados
        self.cursor.execute("INSERT INTO signals (signal) VALUES (?)", (data.tobytes(),))
        self.conn.commit()
        
        # Atualizar gráficos
        self.curve_time.setData(data)
        
        freq_data = np.abs(fft(data))[:self.num_samples // 2]
        self.curve_freq.setData(freq_data)
        
        corr_data = correlate(data, data, mode='full')
        self.curve_corr.setData(corr_data)
    
    def closeEvent(self, event):
        self.task.close()
        self.conn.close()
        event.accept()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModalAnalysisApp()
    window.show()
    sys.exit(app.exec_())
