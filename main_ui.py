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
    QStatusBar, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QLineEdit, QLabel, QPushButton, QComboBox, QTabWidget, QGroupBox, QSlider
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
import numpy as np
from scipy.signal import correlate

import sqlite3
import time

from styles import apply_theme
from theme_engine import load_theme, generate_stylesheet
from data_sources.synthetic_stream import SyntheticStream
from data_sources.sensor_stream import SensorStream



class MainWindow(QMainWindow):
    """
    Main application window for the Realtime Modal Analyzer.

    This class manages the user interface, including the toolbar, tabs, 
    plots for signal visualization, and theme management.

    Attributes
    ----------
    current_theme : str
        The currently applied theme ("dark" or "light").
    auto_scaling : bool
        Whether auto-scaling is enabled for plots.
    y_min : float
        Minimum y-axis value for plots (used when auto-scaling is disabled).
    y_max : float
        Maximum y-axis value for plots (used when auto-scaling is disabled).
    time_plot : pg.PlotWidget
        Plot widget for displaying time-domain signals.
    corr_plot : pg.PlotWidget
        Plot widget for displaying cross-correlation data.
    fft_plot : pg.PlotWidget
        Plot widget for displaying FFT spectrum.
    freq_slider : QSlider
        Slider for adjusting the maximum frequency of the FFT plot.
    freq_value : QLabel
        Label displaying the current frequency value from the slider.
    stats_label : QLabel
        Label for displaying statistics information about the FFT plot.

    Methods
    -------
    __init__()
        Initializes the main window and sets up the user interface.
    init_ui()
        Creates the main layout and components of the user interface.
    create_time_tab()
        Configures the "Time & Correlation" tab with plots.
    create_fft_tab()
        Configures the "Spectrum" tab with FFT controls and plots.
    update_fft_scale()
        Updates the FFT plot's x-axis range based on the slider value.
    create_toolbar()
        Creates the main toolbar with control buttons and theme selector.
    apply_theme(name)
        Applies the specified theme to the application.
    change_theme(theme_name)
        Changes the current theme and updates the application stylesheet.
    start_stream()
        Starts the data streaming and initializes the plots.
    stop_stream()
        Stops the data streaming.
    save_to_sqlite()
        Saves the most recent data to an SQLite database.
    switch_source(source_name)
        Switches the data source (Synthetic or Sensor).
    update_plots()
        Updates the time, FFT, and cross-correlation plots with new data.
    save_last_n_seconds(duration_sec=5)
        Saves the last `duration_sec` seconds of data to SQLite.

    """
    def __init__(self):
        """
        Initializes the main window and sets up the user interface.

        """
        super().__init__()

        # Window setup
        self.setWindowTitle("Realtime Modal Analyzer")
        # self.setGeometry(100, 100, 1200, 800)

        # Current theme
        self.current_theme = "dark"

        # Scaling mode
        self.auto_scaling = True
        self.y_min = -2
        self.y_max = 2

        # Data streaming attributes
        self.data_stream = None
        self.sample_rate = 1000  # Hz
        self.chunk_size = 256  # Number of samples per chunk

        # Start the user interface
        self.init_ui()

        # Apply initial style
        self.change_theme(self.current_theme)

        # Adding stream
        self.data_stream = None
        self.sample_rate = 1000
        self.chunk_size = 256
        
    def set_data_source(self, stream):
        self.data_stream = stream
    
    def start_stream(self):
        self.data_stream.start()
        self.init_plot_data()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)
        self.status_bar.showMessage("Streaming...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def init_plot_data(self):
        """
        Initializes circular buffers and plot curves for visualization.
        Ensures that legends are added only once.
        """
        self.buffer_size = self.sample_rate * 10
        self.buffers = [np.zeros(self.buffer_size) for _ in range(4)]
        self.index = 0

        # Initialize time-domain plot curves
        self.time_curves = []
        if not hasattr(self.time_plot, 'legend') or self.time_plot.legend is None:
            self.time_plot.addLegend()
        for i in range(4):
            self.time_curves.append(self.time_plot.plot(pen=pg.intColor(i), name=f"CH{i}"))

        # Initialize FFT plot curves
        self.fft_curves = []
        if not hasattr(self.fft_plot, 'legend') or self.fft_plot.legend is None:
            self.fft_plot.addLegend()
        for i in range(4):
            self.fft_curves.append(self.fft_plot.plot(pen=pg.intColor(i), name=f"CH{i}"))

        # Initialize cross-correlation plot curves
        self.corr_curves = []
        if not hasattr(self.corr_plot, 'legend') or self.corr_plot.legend is None:
            self.corr_plot.addLegend()
        for i in range(3):
            self.corr_curves.append(self.corr_plot.plot(pen=pg.intColor(i + 1), name=f"AX{i + 1} vs Ref"))


    def update_plots(self):
        if self.data_stream is None:
            return

        data = self.data_stream.read_chunk()
        n = data.shape[1]
        for i in range(4):
            buf = self.buffers[i]
            buf[:-n] = buf[n:]
            buf[-n:] = data[i]

        t = np.linspace(0, 10, self.buffer_size)
        for i in range(4):
            self.time_curves[i].setData(t, self.buffers[i])

        # FFT
        win = np.hanning(self.chunk_size)
        freqs = np.fft.rfftfreq(self.chunk_size, 1 / self.sample_rate)
        for i in range(4):
            fft_vals = np.abs(np.fft.rfft(data[i] * win))
            self.fft_curves[i].setData(freqs, fft_vals)

        # Correlation
        ref = data[0]
        for i in range(3):
            corr = correlate(data[i + 1], ref, mode='full')
            lags = np.arange(-len(ref) + 1, len(ref)) / self.sample_rate
            self.corr_curves[i].setData(lags, corr)

    def init_ui(self):
        """
        Sets up the main user interface layout and components.

        """
        # Central widget and layout
        central = QWidget()
        self.setCentralWidget(central)



        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create UI components
        self.create_toolbar()

        self.tabs = QTabWidget()
        self.create_time_tab()
        self.create_fft_tab()

        main_layout.addWidget(self.tabs)

        # shows short messages at the bottom of the window to give 
        # feedback to the user.
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)




    def create_time_tab(self):
        """
        Configures the "Time & Correlation" tab with plots for time-domain
        signals and cross-correlation data.

        """
        time_tab = QWidget()
        layout = QHBoxLayout(time_tab)
        
        # --- Control panel on the left ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel("Recording Controls")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        control_layout.addWidget(title)

        params_group = QGroupBox("SQLite")
        params_layout = QVBoxLayout(params_group)

        # Modification to add SQLITE3 capabilities to the application
        sqlite_group = QGroupBox("SQLite")
        sqlite_layout = QHBoxLayout(sqlite_group)
        sqlite_layout.addWidget(QLabel("Save Last N Seconds:"))
        self.duration_input = QLineEdit("5")
        sqlite_layout.addWidget(self.duration_input)
        sqlite_button = QPushButton("Save to SQLite")
        sqlite_button.clicked.connect(self.save_to_sqlite)
        sqlite_layout.addWidget(sqlite_button)
        control_layout.addWidget(sqlite_group)


        # sqlite_layout =  QHBoxLayout()
        # save_layout.addWidget(QLabel("Save Last N Seconds:"))
        # save_layout.addWidget(QLineEdit("5"))
        # save_button = QPushButton("Save to SQLite")
        # save_button.clicked.connect(self.save_to_sqlite)
        # save_layout.addWidget(save_button)
        # control_layout.addWidget(params_group)

        # # Frequency scale control
        # freq_layout = QHBoxLayout()
        # freq_layout.addWidget(QLabel("Save Last N Seconds"))
        # self.freq_slider = QSlider(Qt.Horizontal)
        # self.freq_slider.setRange(1, 10)  # Example range 1 to 10 kHz
        # self.freq_slider.setValue(5)
        # self.freq_slider.valueChanged.connect(self.update_fft_scale)
        # self.freq_slider.setToolTip("Adjust the maximum frequency displayed in the FFT plot")
        # freq_layout.addWidget(self.freq_slider)
        # self.freq_value = QLabel("5")
        # self.freq_value.setMinimumWidth(40)
        # freq_layout.addWidget(self.freq_value)
        # params_layout.addLayout(freq_layout)
        # params_group.setLayout(params_layout)
        # control_layout.addWidget(params_group)

        # Stats display group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("No data available")
        self.stats_label.setStyleSheet("font-family: monospace;")
        self.stats_label.setToolTip("Displays statistical information about the signal")
        stats_layout.addWidget(self.stats_label)
        control_layout.addWidget(stats_group)

        # Add stretch to push controls to the top
        control_layout.addStretch()

        layout.addWidget(control_panel)

        # --- Charts panel on the rigth ---
        charts_panel = QWidget()
        charts_layout = QVBoxLayout(charts_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # Time-domain signal plot
        self.time_plot = pg.PlotWidget(title="Time Domain Signal")
        self.time_plot.showGrid(x=True, y=True)
        self.time_plot.addLegend()
        self.time_plot.setLabel('left', 'Acceleration (g)')
        self.time_plot.setLabel('bottom', 'Time (s)')

        # Cross-correlation plot
        self.corr_plot = pg.PlotWidget(title="Cross-correlation")
        self.corr_plot.showGrid(x=True, y=True)
        self.corr_plot.addLegend()
        self.corr_plot.setLabel('left', 'Correlation')
        self.corr_plot.setLabel('bottom', 'Lag (s)')

        charts_layout.addWidget(self.time_plot)
        charts_layout.addWidget(self.corr_plot)
        layout.addWidget(charts_panel)

        self.tabs.addTab(time_tab, "Time & Correlation")

    def create_fft_tab(self):
        """
        Configures the "Spectrum" tab with controls for FFT parameters
        and a plot for the FFT spectrum.

        """
        fft_tab = QWidget()
        layout = QHBoxLayout(fft_tab)  # Horizontal layout for side-by-side panels

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
        freq_layout.addWidget(QLabel("Max Freq (Hz):"))
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(10, 1000)  # Example range 1 to 10 Hz
        self.freq_slider.setValue(500)
        self.freq_slider.valueChanged.connect(self.update_fft_scale)
        self.freq_slider.setToolTip("Adjust the maximum frequency displayed in the FFT plot")
        freq_layout.addWidget(self.freq_slider)
        self.freq_value = QLabel("500")
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
        self.stats_label.setToolTip("Displays statistical information about the signal")
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
        self.fft_plot.addLegend()
        self.fft_plot.setLabel('left', 'Magnitude')
        self.fft_plot.setLabel('bottom', 'Frequency (Hz)')

        fft_plot_layout.addWidget(self.fft_plot)
        layout.addWidget(fft_plot_container)

        self.tabs.addTab(fft_tab, "Spectrum")

    def update_fft_scale(self):
        """
        Updates the FFT plot's x-axis range based on the slider value.

        """
        max_freq = self.freq_slider.value() # Hz
        self.freq_value.setText(str(self.freq_slider.value()))
        self.fft_plot.setXRange(0, max_freq, padding=0)

    def create_toolbar(self):
        """
        Creates the main toolbar with control buttons and a theme selector.

        """
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Add toolbar actions
        self.start_btn = QPushButton(QIcon.fromTheme("media-playback-start"), "Start Live Stream")
        self.start_btn.setToolTip("Start data acquisition and visualization")
        self.start_btn.clicked.connect(self.start_stream)

        self.stop_btn = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop Live Stream")
        self.stop_btn.setToolTip("Stop data acquisition")
        self.stop_btn.clicked.connect(self.stop_stream)
        self.stop_btn.setEnabled(False)

        self.export_btn = QPushButton(QIcon.fromTheme('document-save'), "Export")
        self.export_btn.setToolTip("Export data or image")

        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addWidget(self.export_btn)

        # # Modification to add SQLITE3 capabilities to the application 
        # self.save_duration_label = QLabel("Save Last N Seconds:")
        # self.save_duration_input = QLineEdit("5")  # Default suggestion
        # self.save_button = QPushButton("Save to SQLite")
        # self.save_button.clicked.connect(self.save_to_sqlite)

        # toolbar.addWidget(self.save_duration_label)
        # toolbar.addWidget(self.save_duration_input)
        # toolbar.addWidget(self.save_button)

        # Add spacer
        toolbar.addSeparator()

        # Theme selector
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Dark", "Light"])
        self.theme_selector.currentTextChanged.connect(self.change_theme)
        self.theme_selector.setToolTip("Choose between dark and light themes")
        toolbar.addWidget(QLabel("Theme:"))
        toolbar.addWidget(self.theme_selector)

        # Stream selector
        self.source_selector = QComboBox()
        self.source_selector.addItems(["Synthetic", "NI Sensor"])
        self.source_selector.currentTextChanged.connect(self.switch_source)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Source:"))
        toolbar.addWidget(self.source_selector)


    

    def stop_stream(self):
        """
        Stops the data streaming, resets all charts, and removes legends.
        """
        # Stop the timer to halt data updates
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()

        # Enable the start button and disable the stop button
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Reset the status bar message
        self.status_bar.showMessage("Stopped and charts reset.")

        # Clear all plots for time-domain signals
        for curve in self.time_curves:
            curve.clear()

        # Clear all plots for FFT spectrum
        for curve in self.fft_curves:
            curve.clear()

        # Clear all plots for cross-correlation
        for curve in self.corr_curves:
            curve.clear()

        # Remove legends from the plots
        if hasattr(self, "time_legend") and self.time_legend is not None:
            self.time_plot.scene().removeItem(self.time_legend)
            self.time_legend = None

        if hasattr(self, "fft_legend") and self.fft_legend is not None:
            self.fft_plot.scene().removeItem(self.fft_legend)
            self.fft_legend = None

        if hasattr(self, "corr_legend") and self.corr_legend is not None:
            self.corr_plot.scene().removeItem(self.corr_legend)
            self.corr_legend = None

        # Optionally reset buffers to zero
        if hasattr(self, "buffers"):
            for i in range(len(self.buffers)):
                self.buffers[i].fill(0)


    def save_to_sqlite(self):
        print("Save to SQLite")
        """
        Saves the most recent data to an SQLite database.
        """
        try:
            duration = float(self.duration_input.text())
        except ValueError:
            duration = 5
        self.save_last_n_seconds(duration)


    def switch_source(self, source_name):
        """
        Switches the data source (Synthetic or Sensor).

        Parameters
        ----------
        source_name : str
            The name of the data source to switch to.
        """
        if hasattr(self, "data_stream") and self.data_stream:
            self.data_stream.stop()

        if source_name == "Synthetic":
            self.data_stream = SyntheticStream(sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        elif source_name == "NI Sensor":
            self.data_stream = SensorStream(sample_rate=self.sample_rate, chunk_size=self.chunk_size)

        self.data_stream.start()
        self.init_plot_data()
        self.timer.start(50)
        self.status_bar.showMessage(f"Switched to {source_name} stream")


    def apply_theme(self, name):
        """
        Applies the specified theme to the application.

        Parameters
        ----------
        name : str
            Name of the theme to apply (e.g., "dark" or "light").

        """
        theme = load_theme(name)
        stylesheet = generate_stylesheet(theme)
        self.setStyleSheet(stylesheet)

        # Propagate style to all children
        for child in self.findChildren(QWidget):
            child.setStyleSheet(stylesheet)

    def change_theme(self, theme_name):
        """
        Changes the current theme and updates the application stylesheet.

        Parameters
        ----------
        theme_name : str
            Name of the new theme (e.g., "Dark" or "Light").

        """
        self.current_theme = theme_name.lower()
        apply_theme(self, self.current_theme)
        self.apply_theme(theme_name)

        

        # Change background color of plots based on theme
        bg_color = "k" if self.current_theme == "dark" else "w"
        self.time_plot.setBackground(bg_color)
        self.corr_plot.setBackground(bg_color)
        self.fft_plot.setBackground(bg_color)




    def save_last_n_seconds(self, duration_sec=5):
        """
        Saves the last `duration_sec` seconds of data to SQLite.

        Parameters
        ----------
        duration_sec : int, optional
            Duration in seconds to save (default is 5).
        """
        if not hasattr(self, "buffers"):
            return

        n_points = int(duration_sec * self.sample_rate)
        if n_points > len(self.buffers[0]):
            print("Not enough data to save.")
            return

        end_time = time.time()
        start_time = end_time - duration_sec
        timestamps = np.linspace(start_time, end_time, n_points)

        filename = time.strftime("vibration_data_%Y%m%d_%H%M%S.sqlite")
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS vibration_samples (
                        abs_timestamp REAL,
                        timestamp REAL,
                        reference REAL,
                        x REAL,
                        y REAL,
                        z REAL
                    )""")

        for i in range(n_points):
            row = (
                timestamps[i],
                i / self.sample_rate,
                self.buffers[0][-n_points + i],
                self.buffers[1][-n_points + i],
                self.buffers[2][-n_points + i],
                self.buffers[3][-n_points + i],
            )
            c.execute("INSERT INTO vibration_samples VALUES (?, ?, ?, ?, ?, ?)", row)

        conn.commit()
        conn.close()
        self.status_bar.showMessage(f"Saved {duration_sec}s to {filename}")
