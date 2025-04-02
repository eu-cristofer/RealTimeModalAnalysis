import sys
import random
from datetime import datetime
import numpy as np
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSlider, QSpinBox, 
                            QTabWidget, QTableWidget, QTableWidgetItem, QSplitter,
                            QStatusBar, QToolBar, QStyleFactory, QFileDialog, QGroupBox,
                            QDoubleSpinBox)
from PyQt5.QtGui import QIcon, QColor
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DataGenerator(QThread):
    """Thread for generating realtime data"""
    new_data = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.frequency = 1.0
        self.amplitude = 1.0
        self.offset = 0.0
        self.noise = 0.1
        
    def run(self):
        """Generate sine wave data with noise"""
        t = 0
        while self.running:
            # Generate synthetic data
            value = self.amplitude * np.sin(2 * np.pi * self.frequency * t) + self.offset
            value += random.gauss(0, self.noise)
            
            data = {
                'timestamp': datetime.now(),
                'value': value,
                'frequency': self.frequency,
                'amplitude': self.amplitude,
                'offset': self.offset
            }
            
            self.new_data.emit(data)
            t += 0.1
            self.msleep(100)  # Update every 100ms
            
    def update_parameters(self, freq, amp, offset, noise):
        """Update data generation parameters"""
        self.frequency = freq
        self.amplitude = amp
        self.offset = offset
        self.noise = noise

class RealtimeAnalyzer(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.setWindowTitle("Advanced Realtime Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Current theme
        self.current_theme = "dark"
        
        # Scaling mode
        self.auto_scaling = True
        self.y_min = -2
        self.y_max = 2
        
        # Apply initial style
        self.apply_theme()
        
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create UI components
        self.create_toolbar()
        self.create_control_panel()
        self.create_main_display()
        
        # Data generator
        self.data_generator = DataGenerator()
        self.data_generator.new_data.connect(self.update_displays)
        self.data_generator.start()
        
        # Data storage
        self.data_points = []
        self.max_points = 500
        
        # Initialize displays
        self.init_plots()
        
    def apply_theme(self):
        """Apply the current theme"""
        if self.current_theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2E3440;
                }
                QWidget {
                    color: #D8DEE9;
                    font-family: Segoe UI;
                    font-size: 12px;
                }
                QPushButton {
                    background-color: #3B4252;
                    border: 1px solid #4C566A;
                    border-radius: 4px;
                    padding: 5px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #434C5E;
                }
                QPushButton:pressed {
                    background-color: #4C566A;
                }
                QComboBox, QSpinBox, QSlider, QDoubleSpinBox {
                    background-color: #3B4252;
                    border: 1px solid #4C566A;
                    border-radius: 4px;
                    padding: 3px;
                }
                QTabWidget::pane {
                    border: 1px solid #4C566A;
                    background: #3B4252;
                }
                QTabBar::tab {
                    background: #3B4252;
                    border: 1px solid #4C566A;
                    padding: 8px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background: #434C5E;
                    border-bottom: 2px solid #81A1C1;
                }
                QTableWidget {
                    background-color: #3B4252;
                    gridline-color: #4C566A;
                    border: 1px solid #4C566A;
                }
                QHeaderView::section {
                    background-color: #434C5E;
                    padding: 5px;
                    border: 1px solid #4C566A;
                }
                QGroupBox {
                    border: 1px solid #4C566A;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 15px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                }
            """)
        else:  # light theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #F5F7FA;
                }
                QWidget {
                    color: #2E3440;
                    font-family: Segoe UI;
                    font-size: 12px;
                }
                QPushButton {
                    background-color: #E5E9F0;
                    border: 1px solid #D8DEE9;
                    border-radius: 4px;
                    padding: 5px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #D8DEE9;
                }
                QPushButton:pressed {
                    background-color: #ECEFF4;
                }
                QComboBox, QSpinBox, QSlider, QDoubleSpinBox {
                    background-color: #E5E9F0;
                    border: 1px solid #D8DEE9;
                    border-radius: 4px;
                    padding: 3px;
                }
                QTabWidget::pane {
                    border: 1px solid #D8DEE9;
                    background: #E5E9F0;
                }
                QTabBar::tab {
                    background: #E5E9F0;
                    border: 1px solid #D8DEE9;
                    padding: 8px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background: #ECEFF4;
                    border-bottom: 2px solid #81A1C1;
                }
                QTableWidget {
                    background-color: #E5E9F0;
                    gridline-color: #D8DEE9;
                    border: 1px solid #D8DEE9;
                }
                QHeaderView::section {
                    background-color: #ECEFF4;
                    padding: 5px;
                    border: 1px solid #D8DEE9;
                }
                QGroupBox {
                    border: 1px solid #D8DEE9;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 15px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                }
            """)
        
    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Add toolbar actions
        self.start_button = QPushButton(QIcon.fromTheme('media-playback-start'), "Start")
        self.stop_button = QPushButton(QIcon.fromTheme('media-playback-stop'), "Stop")
        self.export_button = QPushButton(QIcon.fromTheme('document-save'), "Export")
        
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.export_button.clicked.connect(self.export_data)
        
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addWidget(self.export_button)
        
        # Add spacer
        toolbar.addSeparator()
        
        # Theme selector
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        
        toolbar.addWidget(theme_label)
        toolbar.addWidget(self.theme_combo)
        
    def create_control_panel(self):
        """Create the control panel on the left side"""
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Analysis Controls")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        control_layout.addWidget(title)
        
        # Signal parameters group
        params_group = QGroupBox("Signal Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Frequency control
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency:"))
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 2)
        self.freq_slider.setValue(1)
        self.freq_slider.valueChanged.connect(self.update_generator_params)
        freq_layout.addWidget(self.freq_slider)
        self.freq_value = QLabel("5.0")
        self.freq_value.setMinimumWidth(40)
        freq_layout.addWidget(self.freq_value)
        params_layout.addLayout(freq_layout)
        
        # Amplitude control
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("Amplitude:"))
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(1, 100)
        self.amp_slider.setValue(50)
        self.amp_slider.valueChanged.connect(self.update_generator_params)
        amp_layout.addWidget(self.amp_slider)
        self.amp_value = QLabel("50.0")
        self.amp_value.setMinimumWidth(40)
        amp_layout.addWidget(self.amp_value)
        params_layout.addLayout(amp_layout)
        
        # Offset control
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset:"))
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(-100, 100)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(self.update_generator_params)
        offset_layout.addWidget(self.offset_slider)
        self.offset_value = QLabel("0.0")
        self.offset_value.setMinimumWidth(40)
        offset_layout.addWidget(self.offset_value)
        params_layout.addLayout(offset_layout)
        
        # Noise control
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise:"))
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 50)
        self.noise_slider.setValue(10)
        self.noise_slider.valueChanged.connect(self.update_generator_params)
        noise_layout.addWidget(self.noise_slider)
        self.noise_value = QLabel("0.1")
        self.noise_value.setMinimumWidth(40)
        noise_layout.addWidget(self.noise_value)
        params_layout.addLayout(noise_layout)
        
        control_layout.addWidget(params_group)
        
        # Scaling controls group
        scaling_group = QGroupBox("Scaling Controls")
        scaling_layout = QVBoxLayout(scaling_group)
        
        # Auto/manual toggle
        scaling_toggle_layout = QHBoxLayout()
        self.auto_scaling_button = QPushButton("Auto Scaling")
        self.auto_scaling_button.setCheckable(True)
        self.auto_scaling_button.setChecked(True)
        self.auto_scaling_button.clicked.connect(self.toggle_scaling_mode)
        scaling_toggle_layout.addWidget(self.auto_scaling_button)
        
        self.reset_scaling_button = QPushButton("Reset")
        self.reset_scaling_button.clicked.connect(self.reset_scaling)
        scaling_toggle_layout.addWidget(self.reset_scaling_button)
        scaling_layout.addLayout(scaling_toggle_layout)
        
        # Manual scaling controls
        self.manual_scaling_group = QGroupBox("Manual Range")
        self.manual_scaling_group.setEnabled(False)
        manual_layout = QVBoxLayout(self.manual_scaling_group)
        
        # Y-min control
        ymin_layout = QHBoxLayout()
        ymin_layout.addWidget(QLabel("Y Min:"))
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1000, 1000)
        self.ymin_spin.setValue(-2)
        self.ymin_spin.setSingleStep(0.1)
        self.ymin_spin.valueChanged.connect(self.update_manual_range)
        ymin_layout.addWidget(self.ymin_spin)
        manual_layout.addLayout(ymin_layout)
        
        # Y-max control
        ymax_layout = QHBoxLayout()
        ymax_layout.addWidget(QLabel("Y Max:"))
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-1000, 1000)
        self.ymax_spin.setValue(2)
        self.ymax_spin.setSingleStep(0.1)
        self.ymax_spin.valueChanged.connect(self.update_manual_range)
        ymax_layout.addWidget(self.ymax_spin)
        manual_layout.addLayout(ymax_layout)
        
        scaling_layout.addWidget(self.manual_scaling_group)
        control_layout.addWidget(scaling_group)
        
        # Analysis controls group
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Display options
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Raw Data", "Moving Avg", "FFT"])
        self.display_combo.currentTextChanged.connect(self.change_display_mode)
        analysis_layout.addWidget(self.display_combo)
        
        # Sample rate
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Sample Rate:"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 100)
        self.sample_spin.setValue(10)
        self.sample_spin.valueChanged.connect(self.change_sample_rate)
        sample_layout.addWidget(self.sample_spin)
        sample_layout.addWidget(QLabel("Hz"))
        analysis_layout.addLayout(sample_layout)
        
        control_layout.addWidget(analysis_group)
        
        # Stats display group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("No data available")
        self.stats_label.setStyleSheet("font-family: monospace;")
        stats_layout.addWidget(self.stats_label)
        control_layout.addWidget(stats_group)
        
        # Add stretch to push controls to top
        control_layout.addStretch()
        
        self.main_layout.addWidget(control_panel)
        
    def create_main_display(self):
        """Create the main display area with tabs"""
        # Create a splitter for the main area
        splitter = QSplitter(Qt.Vertical)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        
        # Create tabs
        self.create_realtime_tab()
        self.create_spectrum_tab()
        self.create_table_tab()
        
        splitter.addWidget(self.tab_widget)
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        splitter.addWidget(self.status_bar)
        
        self.main_layout.addWidget(splitter)
        
    def create_realtime_tab(self):
        """Create the realtime plotting tab"""
        realtime_tab = QWidget()
        layout = QVBoxLayout(realtime_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create pyqtgraph plot
        self.realtime_plot = pg.PlotWidget()
        self.realtime_plot.setBackground('#3B4252' if self.current_theme == "dark" else '#E5E9F0')
        self.realtime_plot.showGrid(x=True, y=True)
        self.realtime_plot.setLabel('left', 'Value')
        self.realtime_plot.setLabel('bottom', 'Time')
        
        # Add plot to layout
        layout.addWidget(self.realtime_plot)
        
        # Add to tab widget
        self.tab_widget.addTab(realtime_tab, "Realtime")
        
    def create_spectrum_tab(self):
        """Create the spectrum analysis tab"""
        spectrum_tab = QWidget()
        layout = QVBoxLayout(spectrum_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.spectrum_figure = Figure(facecolor='#3B4252' if self.current_theme == "dark" else '#E5E9F0')
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrum_ax = self.spectrum_figure.add_subplot(111)
        self.spectrum_ax.set_facecolor('#3B4252' if self.current_theme == "dark" else '#E5E9F0')
        
        # Style the plot
        text_color = '#D8DEE9' if self.current_theme == "dark" else '#2E3440'
        line_color = '#81A1C1' if self.current_theme == "dark" else '#5E81AC'
        
        for spine in self.spectrum_ax.spines.values():
            spine.set_color(text_color)
        self.spectrum_ax.tick_params(colors=text_color)
        self.spectrum_ax.xaxis.label.set_color(text_color)
        self.spectrum_ax.yaxis.label.set_color(text_color)
        self.spectrum_ax.title.set_color(text_color)
        
        self.spectrum_line, = self.spectrum_ax.plot([], [], color=line_color)
        self.spectrum_ax.set_title("Frequency Spectrum")
        self.spectrum_ax.set_xlabel("Frequency (Hz)")
        self.spectrum_ax.set_ylabel("Magnitude")
        
        layout.addWidget(self.spectrum_canvas)
        self.tab_widget.addTab(spectrum_tab, "Spectrum")
        
    def create_table_tab(self):
        """Create the data table tab"""
        table_tab = QWidget()
        layout = QVBoxLayout(table_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels(["Timestamp", "Value", "Frequency", "Amplitude", "Offset"])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.data_table)
        self.tab_widget.addTab(table_tab, "Data Table")
        
    def init_plots(self):
        """Initialize plot data"""
        # Realtime plot
        line_color = '#81A1C1' if self.current_theme == "dark" else '#5E81AC'
        self.realtime_curve = self.realtime_plot.plot(pen=pg.mkPen(line_color, width=2))
        self.update_plot_range()
        
    def update_generator_params(self):
        """Update data generator parameters based on UI controls"""
        freq = self.freq_slider.value() / 2.0
        amp = self.amp_slider.value() / 10.0
        offset = self.offset_slider.value() / 10.0
        noise = self.noise_slider.value() / 100.0
        
        # Update display values
        self.freq_value.setText(f"{freq:.1f}")
        self.amp_value.setText(f"{amp:.1f}")
        self.offset_value.setText(f"{offset:.1f}")
        self.noise_value.setText(f"{noise:.2f}")
        
        # Update generator
        self.data_generator.update_parameters(freq, amp, offset, noise)
        
    def update_displays(self, data):
        """Update all displays with new data"""
        # Store data
        self.data_points.append(data)
        if len(self.data_points) > self.max_points:
            self.data_points.pop(0)
            
        # Update realtime plot
        x = np.arange(len(self.data_points))
        y = [d['value'] for d in self.data_points]
        self.realtime_curve.setData(x, y)
        
        # Auto-scale if enabled
        if self.auto_scaling and len(y) > 0:
            y_min = min(y)
            y_max = max(y)
            padding = max(0.1, (y_max - y_min) * 0.1)  # 10% padding or 0.1 minimum
            self.realtime_plot.setYRange(y_min - padding, y_max + padding)
        
        # Update spectrum plot
        if len(y) > 10:
            self.update_spectrum(y)
            
        # Update table
        self.update_table()
        
        # Update statistics
        self.update_stats()
        
    def update_spectrum(self, y_data):
        """Update the frequency spectrum display"""
        n = len(y_data)
        if n < 2:
            return
            
        # Compute FFT
        y_fft = np.fft.fft(y_data)
        x_fft = np.fft.fftfreq(n, d=0.1)[:n//2]
        
        # Update plot
        self.spectrum_line.set_data(x_fft, 2/n * np.abs(y_fft[0:n//2]))
        self.spectrum_ax.relim()
        self.spectrum_ax.autoscale_view()
        self.spectrum_canvas.draw()
        
    def update_table(self):
        """Update the data table"""
        self.data_table.setRowCount(min(len(self.data_points), 20))  # Show max 20 points
        
        start_idx = max(0, len(self.data_points) - 20)
        for row, data in enumerate(self.data_points[start_idx:]):
            self.data_table.setItem(row, 0, QTableWidgetItem(data['timestamp'].strftime("%H:%M:%S.%f")))
            self.data_table.setItem(row, 1, QTableWidgetItem(f"{data['value']:.4f}"))
            self.data_table.setItem(row, 2, QTableWidgetItem(f"{data['frequency']:.2f}"))
            self.data_table.setItem(row, 3, QTableWidgetItem(f"{data['amplitude']:.2f}"))
            self.data_table.setItem(row, 4, QTableWidgetItem(f"{data['offset']:.2f}"))
            
        # Scroll to bottom
        self.data_table.scrollToBottom()
        
    def update_stats(self):
        """Update the statistics display"""
        if not self.data_points:
            return
            
        values = [d['value'] for d in self.data_points]
        stats_text = f"""
        Current: {values[-1]:.4f}
        Mean:    {np.mean(values):.4f}
        Std Dev: {np.std(values):.4f}
        Min:     {np.min(values):.4f}
        Max:     {np.max(values):.4f}
        """
        self.stats_label.setText(stats_text)
        
    def toggle_scaling_mode(self):
        """Toggle between auto and manual scaling"""
        self.auto_scaling = self.auto_scaling_button.isChecked()
        
        if self.auto_scaling:
            self.auto_scaling_button.setText("Auto Scaling")
            self.manual_scaling_group.setEnabled(False)
            self.status_bar.showMessage("Auto scaling enabled")
        else:
            self.auto_scaling_button.setText("Manual Scaling")
            self.manual_scaling_group.setEnabled(True)
            self.update_manual_range()
            self.status_bar.showMessage("Manual scaling enabled")
            
        self.update_plot_range()
        
    def reset_scaling(self):
        """Reset to auto-scaling"""
        self.auto_scaling = True
        self.auto_scaling_button.setChecked(True)
        self.toggle_scaling_mode()
        self.status_bar.showMessage("Scaling reset to auto")
        
    def update_manual_range(self):
        """Update the manual scaling range"""
        if not self.auto_scaling:
            self.y_min = self.ymin_spin.value()
            self.y_max = self.ymax_spin.value()
            self.update_plot_range()
            
    def update_plot_range(self):
        """Update the plot's Y-axis range"""
        if self.auto_scaling:
            # Let the plot auto-scale
            self.realtime_plot.enableAutoRange('y', True)
        else:
            # Set manual range
            self.realtime_plot.disableAutoRange('y')
            self.realtime_plot.setYRange(self.y_min, self.y_max)
        
    def start_analysis(self):
        """Start data analysis"""
        self.status_bar.showMessage("Analysis started")
        
    def stop_analysis(self):
        """Stop data analysis"""
        self.status_bar.showMessage("Analysis paused")
        
    def export_data(self):
        """Export data to file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Data", "", 
                                                 "CSV Files (*.csv);;All Files (*)", 
                                                 options=options)
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    f.write("Timestamp,Value,Frequency,Amplitude,Offset\n")
                    for data in self.data_points:
                        f.write(f"{data['timestamp']},{data['value']},{data['frequency']},{data['amplitude']},{data['offset']}\n")
                self.status_bar.showMessage(f"Data exported to {file_name}")
            except Exception as e:
                self.status_bar.showMessage(f"Export failed: {str(e)}")
                
    def change_display_mode(self, mode):
        """Change the display mode"""
        self.status_bar.showMessage(f"Display mode changed to {mode}")
        
    def change_sample_rate(self, rate):
        """Change the sample rate"""
        self.status_bar.showMessage(f"Sample rate changed to {rate}Hz")
        
    def change_theme(self, theme):
        """Change the application theme"""
        self.current_theme = theme.lower()
        self.apply_theme()
        
        # Update plot backgrounds
        plot_bg = '#3B4252' if self.current_theme == "dark" else '#E5E9F0'
        text_color = '#D8DEE9' if self.current_theme == "dark" else '#2E3440'
        line_color = '#81A1C1' if self.current_theme == "dark" else '#5E81AC'
        
        self.realtime_plot.setBackground(plot_bg)
        self.realtime_curve.setPen(pg.mkPen(line_color, width=2))
        
        self.spectrum_figure.set_facecolor(plot_bg)
        self.spectrum_ax.set_facecolor(plot_bg)
        for spine in self.spectrum_ax.spines.values():
            spine.set_color(text_color)
        self.spectrum_ax.tick_params(colors=text_color)
        self.spectrum_ax.xaxis.label.set_color(text_color)
        self.spectrum_ax.yaxis.label.set_color(text_color)
        self.spectrum_ax.title.set_color(text_color)
        self.spectrum_line.set_color(line_color)
        self.spectrum_canvas.draw()
        
        self.status_bar.showMessage(f"Theme changed to {theme}")
            
    def closeEvent(self, event):
        """Clean up on application close"""
        self.data_generator.running = False
        self.data_generator.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    analyzer = RealtimeAnalyzer()
    analyzer.show()
    
    sys.exit(app.exec_())