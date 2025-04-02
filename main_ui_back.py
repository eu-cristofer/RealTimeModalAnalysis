from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QToolBar, QLabel, QPushButton, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from styles import apply_theme
from theme_engine import load_theme, generate_stylesheet
import pyqtgraph as pg
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget, QGroupBox, QSlider

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("Realtime Modal Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Current theme
        self.current_theme = "dark"

        # Scaling mode
        self.auto_scaling = True
        self.y_min = -2
        self.y_max = 2

        # Start the user interface
        self.init_ui()
        
        # Apply initial style
        self.change_theme(self.current_theme)


    def init_ui(self):
        # Central widget and layout
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)



        # main_layout = QVBoxLayout(central)
        # Create UI components
        self.create_toolbar()

        self.tabs = QTabWidget()
        self.create_time_tab()
        self.create_fft_tab()

        main_layout.addWidget(self.tabs)

    

    def create_time_tab(self):
        time_tab = QWidget()
        layout = QVBoxLayout(time_tab)

        # Time-domain signal plot
        self.time_plot = pg.PlotWidget(title="Time Domain Signal")
        self.time_plot.showGrid(x=True, y=True)
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setLabel('bottom', 'Time')

        # Cross-correlation plot
        self.corr_plot = pg.PlotWidget(title="Cross-correlation")
        self.corr_plot.showGrid(x=True, y=True)
        self.corr_plot.setLabel('left', 'Correlation')
        self.corr_plot.setLabel('bottom', 'Lag')

        layout.addWidget(self.time_plot)
        layout.addWidget(self.corr_plot)

        self.tabs.addTab(time_tab, "Time & Correlation")

    def create_fft_tab(self):
        fft_tab = QWidget()
        layout = QHBoxLayout(fft_tab)  # Horizontal layout to allow for side-by-side panels

        # --- Control panel on the left ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel("FFT Controls")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        control_layout.addWidget(title)

        params_group = QGroupBox("FFT Scale Settings")
        params_layout = QVBoxLayout(params_group)

        # Frequency scale control
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Max Freq (kHz):"))
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 10)  # Example range 1 to 10 kHz
        self.freq_slider.setValue(5)
        self.freq_slider.valueChanged.connect(self.update_fft_scale)
        freq_layout.addWidget(self.freq_slider)
        self.freq_value = QLabel("5")
        self.freq_value.setMinimumWidth(40)
        freq_layout.addWidget(self.freq_value)
        params_layout.addLayout(freq_layout)
        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)

        # Stats display group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("No data available")
        self.stats_label.setStyleSheet("font-family: monospace;")
        stats_layout.addWidget(self.stats_label)
        control_layout.addWidget(stats_group)
        
        # Add stretch to push controls to top
        control_layout.addStretch()

        layout.addWidget(control_panel)

        # --- FFT Plot on the right ---
        fft_plot_container = QWidget()
        fft_plot_layout = QVBoxLayout(fft_plot_container)

        self.fft_plot = pg.PlotWidget(title="FFT Spectrum")
        self.fft_plot.showGrid(x=True, y=True)
        self.fft_plot.setLabel('left', 'Magnitude')
        self.fft_plot.setLabel('bottom', 'Frequency (Hz)')

        fft_plot_layout.addWidget(self.fft_plot)
        layout.addWidget(fft_plot_container)

        self.tabs.addTab(fft_tab, "Spectrum")

    def update_fft_scale(self):
        max_freq = self.freq_slider.value() * 1000  # Convert from kHz to Hz
        self.freq_value.setText(str(self.freq_slider.value()))
        self.fft_plot.setXRange(0, max_freq, padding=0)


    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Add toolbar actions
        self.start_btn = QPushButton(QIcon.fromTheme("media-playback-start"), "Start")
        self.stop_btn = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop")
        self.export_btn = QPushButton(QIcon.fromTheme('document-save'), "Export")

        # TODO
        # Add toolbar actions
        # self.start_button.clicked.connect(self.start_analysis)
        # self.stop_button.clicked.connect(self.stop_analysis)
        # self.export_button.clicked.connect(self.export_data)

        

        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addWidget(self.export_btn)
        
        # Add spacer
        toolbar.addSeparator()

        # Theme selector
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Dark", "Light"])
        self.theme_selector.currentTextChanged.connect(self.change_theme)
        toolbar.addWidget(QLabel("Theme:"))
        toolbar.addWidget(self.theme_selector)

    def apply_theme(self, name):
        theme = load_theme(name)
        stylesheet = generate_stylesheet(theme)
        self.setStyleSheet(stylesheet)

    def change_theme(self, theme_name):
        self.current_theme = theme_name.lower()
        apply_theme(self, self.current_theme)
        self.apply_theme(theme_name)
