"""
Realtime Modal Analyzer
=======================

This module implements a PyQt5-based GUI application for real-time modal analysis.
It provides interactive tools for visualizing signals in the time domain, frequency domain, 
and cross-correlation plots, along with controls for adjusting the FFT parameters and themes.

Dependencies
------------
- PyQt5
- pyqtgraph
- styles (custom module for theme application)
- theme_engine (custom module for theme generation)

Classes
-------
MainWindow : QMainWindow
    The main application window that contains the user interface.

Notes
-----
This module is a starting point for a real-time modal analysis application. 
Currently, it does not process real data but provides the necessary framework 
to integrate signal processing and analysis functionalities.

"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QLabel, QPushButton, QComboBox, QTabWidget, QGroupBox, QSlider
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
from styles import apply_theme
from theme_engine import load_theme, generate_stylesheet


class MainWindow(QMainWindow):
    def __init__(self):
        """
        Initializes the main window and sets up the user interface.
        """
        super().__init__()

        self.setWindowTitle("Realtime Modal Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        self.current_theme = "dark"
        self.auto_scaling = True
        self.y_min = -2
        self.y_max = 2

        self.init_ui()
        self.change_theme(self.current_theme)

    def init_ui(self):
        """
        Sets up the main layout and initializes all UI components.
        """
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        self.create_toolbar()

        self.tabs = QTabWidget()
        self.create_time_tab()
        self.create_fft_tab()

        main_layout.addWidget(self.tabs)

    def create_time_tab(self):
        """
        Configures the "Time & Correlation" tab with time-domain
        and cross-correlation plots.
        """
        time_tab = QWidget()
        layout = QVBoxLayout(time_tab)

        self.time_plot = pg.PlotWidget(title="Time Domain Signal")
        self.time_plot.showGrid(x=True, y=True)
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setLabel('bottom', 'Time')

        self.corr_plot = pg.PlotWidget(title="Cross-correlation")
        self.corr_plot.showGrid(x=True, y=True)
        self.corr_plot.setLabel('left', 'Correlation')
        self.corr_plot.setLabel('bottom', 'Lag')

        layout.addWidget(self.time_plot)
        layout.addWidget(self.corr_plot)

        self.tabs.addTab(time_tab, "Time & Correlation")

    def create_fft_tab(self):
        """
        Configures the "Spectrum" tab with FFT plot and interactive controls.
        """
        fft_tab = QWidget()
        layout = QHBoxLayout(fft_tab)

        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel("FFT Controls")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        control_layout.addWidget(title)

        params_group = QGroupBox("FFT Scale Settings")
        params_layout = QVBoxLayout(params_group)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Max Freq (kHz):"))
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 10)
        self.freq_slider.setValue(5)
        self.freq_slider.valueChanged.connect(self.update_fft_scale)
        self.freq_slider.setToolTip("Adjust the maximum frequency displayed in the FFT plot")
        freq_layout.addWidget(self.freq_slider)

        self.freq_value = QLabel("5")
        self.freq_value.setMinimumWidth(40)
        freq_layout.addWidget(self.freq_value)
        params_layout.addLayout(freq_layout)
        control_layout.addWidget(params_group)

        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("No data available")
        self.stats_label.setStyleSheet("font-family: monospace;")
        self.stats_label.setToolTip("Displays statistical information about the signal")
        stats_layout.addWidget(self.stats_label)
        control_layout.addWidget(stats_group)

        control_layout.addStretch()
        layout.addWidget(control_panel)

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
        """
        Updates the FFT plot x-axis range based on the slider value.
        """
        max_freq = self.freq_slider.value() * 1000
        self.freq_value.setText(str(self.freq_slider.value()))
        self.fft_plot.setXRange(0, max_freq, padding=0)

    def create_toolbar(self):
        """
        Creates the top toolbar with Start, Stop, Export buttons and theme switcher.
        """
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.start_btn = QPushButton(QIcon.fromTheme("media-playback-start"), "Start")
        self.start_btn.setToolTip("Start data acquisition and visualization")
        self.stop_btn = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop")
        self.stop_btn.setToolTip("Stop data acquisition")
        self.export_btn = QPushButton(QIcon.fromTheme('document-save'), "Export")
        self.export_btn.setToolTip("Export data or image")

        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addWidget(self.export_btn)
        toolbar.addSeparator()

        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Dark", "Light"])
        self.theme_selector.currentTextChanged.connect(self.change_theme)
        self.theme_selector.setToolTip("Choose between dark and light themes")
        toolbar.addWidget(QLabel("Theme:"))
        toolbar.addWidget(self.theme_selector)

    def apply_theme(self, name):
        """
        Applies the Qt stylesheet from the selected theme.

        Parameters
        ----------
        name : str
            Name of the theme ("dark" or "light")
        """
        theme = load_theme(name)
        stylesheet = generate_stylesheet(theme)
        self.setStyleSheet(stylesheet)

    def change_theme(self, theme_name):
        """
        Updates the theme for the entire application, including
        Qt styles and plot backgrounds.

        Parameters
        ----------
        theme_name : str
            The selected theme name ("Dark" or "Light").
        """
        self.current_theme = theme_name.lower()
        apply_theme(self, self.current_theme)
        self.apply_theme(theme_name)

        # Change background color of plots based on theme
        bg_color = "k" if self.current_theme == "dark" else "w"
        self.time_plot.setBackground(bg_color)
        self.corr_plot.setBackground(bg_color)
        self.fft_plot.setBackground(bg_color)
