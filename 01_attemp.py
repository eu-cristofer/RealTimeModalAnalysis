import sys
import time
import threading
import sqlite3
import numpy as np
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.stream_readers
from scipy.fftpack import fft
from scipy.signal import correlate
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Configurações do acelerômetro
DEVICE = "cDAQ9181-1FC6921Mod1"
CHANNELS = [f"{DEVICE}/ai0"]  # Modifique conforme necessário
SAMPLE_RATE = 1000  # Hz
DURATION = 30  # Segundos máximos de aquisição
DB_NAME = "modal_analysis.db"

def create_database():
    """Cria a tabela do banco de dados SQLite3 se não existir."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS acquisitions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        channel TEXT,
                        data BLOB)''')
    conn.commit()
    conn.close()

def save_to_database(channel, data):
    """Salva os dados adquiridos no banco SQLite3."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO acquisitions (timestamp, channel, data) VALUES (datetime('now'), ?, ?)", (channel, data.tobytes()))
    conn.commit()
    conn.close()

class ModalAnalysisApp(QMainWindow):
    """Classe principal para a interface gráfica de aquisição de dados."""
    def __init__(self):
        super().__init__()
        self.initUI()
        self.running = False
        self.task = None
        self.data = {ch: [] for ch in CHANNELS}  # Armazena dados adquiridos

    def initUI(self):
        """Inicializa a interface gráfica."""
        self.setWindowTitle("Aquisição de Dados para Análise Modal")
        self.setGeometry(100, 100, 800, 600)
        
        self.start_button = QPushButton("Iniciar Aquisição", self)
        self.start_button.clicked.connect(self.start_acquisition)
        
        self.stop_button = QPushButton("Parar Aquisição", self)
        self.stop_button.clicked.connect(self.stop_acquisition)
        
        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def acquire_data(self):
        """Função que coleta os dados do acelerômetro."""
        with nidaqmx.Task() as task:
            for ch in CHANNELS:
                task.ai_channels.add_ai_accel_chan(
                    physical_channel=ch,
                    sensitivity=100.0,
                    min_val=-50.0,
                    max_val=50.0,
                    units=constants.AccelUnits.G,
                    current_excit_source=constants.ExcitationSource.INTERNAL,
                    current_excit_val=0.002
                )

            task.timing.cfg_samp_clk_timing(rate=SAMPLE_RATE, sample_mode=constants.AcquisitionType.CONTINUOUS)
            task.in_stream.input_buf_size = 10 * SAMPLE_RATE  # Aumenta o buffer
            
            reader = nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream)
            num_samples_per_read = SAMPLE_RATE  # Lê 1 segundo de dados por vez
            data_chunk = np.zeros((len(CHANNELS), num_samples_per_read))
            
            self.running = True
            start_time = time.time()

            while self.running and (time.time() - start_time) < DURATION:
                reader.read_many_sample(data_chunk, number_of_samples_per_channel=num_samples_per_read, timeout=10)
                
                for i, ch in enumerate(CHANNELS):
                    self.data[ch].extend(data_chunk[i, :])
                    save_to_database(ch, np.array(self.data[ch]))
                
                self.update_plot()
                time.sleep(0.1)
    
    def start_acquisition(self):
        """Inicia a aquisição de dados em uma thread separada."""
        if not self.running:
            self.data = {ch: [] for ch in CHANNELS}  # Resetar dados
            self.thread = threading.Thread(target=self.acquire_data)
            self.thread.start()
    
    def stop_acquisition(self):
        """Interrompe a aquisição de dados."""
        self.running = False
    
    def update_plot(self):
        """Atualiza os gráficos com os novos dados."""
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        
        for ch, data in self.data.items():
            if len(data) > 0:
                time_axis = np.linspace(0, len(data)/SAMPLE_RATE, len(data))
                freq_axis = np.fft.fftfreq(len(data), 1/SAMPLE_RATE)
                fft_data = np.abs(fft(data))
                correlation = correlate(data, data, mode='full')
                
                self.axs[0].plot(time_axis, data, label=f"Sinal - {ch}")
                self.axs[1].plot(freq_axis[:len(freq_axis)//2], fft_data[:len(fft_data)//2], label=f"FFT - {ch}")
                self.axs[2].plot(correlation, label=f"Correlação - {ch}")
        
        self.axs[0].set_title("Sinal no Tempo")
        self.axs[1].set_title("Espectro de Frequências (FFT)")
        self.axs[2].set_title("Função de Correlação")
        
        for ax in self.axs:
            ax.legend()
            ax.grid()
        
        self.canvas.draw()

if __name__ == "__main__":
    create_database()
    app = QApplication(sys.argv)
    mainWin = ModalAnalysisApp()
    mainWin.show()
    sys.exit(app.exec_())
