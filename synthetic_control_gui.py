"""
Synthetic Signal Generator GUI
==============================

This module implements a PyQt5-based GUI application for controlling a synthetic 
signal generator and visualizing the generated data in real-time.

The application provides tools to adjust signal parameters such as frequency, amplitude, 
and noise, as well as phase shifts for multiple axes. It supports theme switching (dark and light) 
and includes controls to start and stop the signal generator.

Dependencies
------------
- PyQt5
- numpy
- main_ui (custom module for the main plotting window)
- data_sources.synthetic_stream (custom module for synthetic data streaming)
- styles (custom module for theme application)
- theme_engine (custom module for theme generation)

Classes
-------
SyntheticControlPanel : QMainWindow
    A control panel for configuring synthetic signal parameters and managing the signal generator.
    
Functions
---------
main()
    The entry point of the application. Initializes the synthetic signal generator, 
    launches the main plotting window, and opens the control panel.

Examples
--------
Run this module to start the application:

>>> python synthetic_control_panel.py
"""
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QPushButton, QGroupBox, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from main_ui import MainWindow
from data_sources.synthetic_stream import SyntheticStream
from styles import apply_theme
from theme_engine import load_theme, generate_stylesheet


class SyntheticControlPanel(QMainWindow):
    """
    A PyQt5 GUI window for controlling synthetic signal generation.

    This class allows users to adjust parameters for a synthetic signal generator, such as:
    - Frequency
    - Amplitude
    - Noise level
    - Phase shifts for multiple axes (X, Y, Z)

    It also provides options to toggle themes (dark and light) and controls to start and stop 
    the signal generator.

    Parameters
    ----------
    stream : SyntheticStream
        An instance of the synthetic signal generator.
    plot_window : MainWindow
        The main plotting window for visualizing the generated signals.

    Attributes
    ----------
    stream : SyntheticStream
        The synthetic signal generator instance.
    plot_window : MainWindow
        The main plotting window.
    current_theme : str
        The current theme of the application ("dark" or "light").
    freq_spin : QDoubleSpinBox
        Spin box for adjusting the frequency of the signal.
    amp_spin : QDoubleSpinBox
        Spin box for adjusting the amplitude of the signal.
    noise_spin : QDoubleSpinBox
        Spin box for adjusting the noise level of the signal.
    phase_x : QDoubleSpinBox
        Spin box for adjusting the phase shift along the X axis.
    phase_y : QDoubleSpinBox
        Spin box for adjusting the phase shift along the Y axis.
    phase_z : QDoubleSpinBox
        Spin box for adjusting the phase shift along the Z axis.
    theme_combo : QComboBox
        Combo box for selecting the application theme.
    start_button : QPushButton
        Button to start the signal generator.
    stop_button : QPushButton
        Button to stop the signal generator.

    Methods
    -------
    init_ui()
        Sets up the user interface components and layouts.
    _add_spin(layout, label, default, minv, maxv)
        Creates a labeled spin box for numeric input.
    update_params()
        Updates the synthetic signal generator with the current parameter values.
    start_stream()
        Starts the synthetic signal generator.
    stop_stream()
        Stops the synthetic signal generator.
    apply_theme(name)
        Applies the specified theme to the application.
    toggle_theme(theme_name)
        Toggles between dark and light themes.
    """
    def __init__(self, stream: SyntheticStream, plot_window: MainWindow):
        """
        Initializes the synthetic control panel.

        Parameters
        ----------
        stream : SyntheticStream
            An instance of the synthetic signal generator.
        plot_window : MainWindow
            The main plotting window for visualizing the generated signals.

        """
        super().__init__()

        self.stream = stream
        self.plot_window = plot_window

        # Window setup
        self.setWindowTitle("Synthetic Signal Controller")
        self.setGeometry(20, 20, 20, 20)
        self.current_theme = "dark"
        self.init_ui()
        self.toggle_theme(self.current_theme)

    def init_ui(self):
        """
        Sets up the user interface components and layouts.

        """
        # Required for QMainWindow
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 🌟 Title
        title = QLabel("Synthetic Signal Generator")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 🎛 Signal settings group
        signal_group = QGroupBox("Signal Parameters")
        group_layout = QVBoxLayout(signal_group)
        group_layout.setSpacing(10)

        self.freq_spin = self._add_spin(group_layout, "Frequency (Hz):", 10, 0.1, 100)
        self.amp_spin = self._add_spin(group_layout, "Amplitude:", 1.0, 0.1, 10)
        self.noise_spin = self._add_spin(group_layout, "Noise (σ):", 0.01, 0.0, 1.0)

        self.phase_x = self._add_spin(group_layout, "Phase X (rad):", 0.0, -np.pi, np.pi)
        self.phase_y = self._add_spin(group_layout, "Phase Y (rad):", 0.2, -np.pi, np.pi)
        self.phase_z = self._add_spin(group_layout, "Phase Z (rad):", -0.3, -np.pi, np.pi)

        layout.addWidget(signal_group)

        # 🌙 Theme selector
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText("Dark")
        self.theme_combo.currentTextChanged.connect(self.toggle_theme)
        theme_layout.addStretch()
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)

        # 🎮 Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Generator")
        self.stop_button = QPushButton("Stop Generator")
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        self.update_params()

    def _add_spin(self, layout, label, default, minv, maxv):
        """
        Creates a labeled spin box for numeric input.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the spin box will be added.
        label : str
            The label text for the spin box.
        default : float
            The default value for the spin box.
        minv : float
            The minimum value for the spin box.
        maxv : float
            The maximum value for the spin box.

        Returns
        -------
        QDoubleSpinBox
            The created spin box.

        """
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(130)
        spin = QDoubleSpinBox()
        spin.setRange(minv, maxv)
        spin.setValue(default)
        spin.setSingleStep(0.1)
        spin.valueChanged.connect(self.update_params)
        row.addWidget(lbl)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def update_params(self):
        """
        Updates the synthetic signal generator with the current parameter values.

        """
        self.stream.freq = self.freq_spin.value()
        self.stream.amplitude = self.amp_spin.value()
        self.stream.noise = self.noise_spin.value()
        self.stream.phase_shift = [
            self.phase_x.value(),
            self.phase_y.value(),
            self.phase_z.value()
        ]

    def start_stream(self):
        """
        Starts the synthetic signal generator.

        """
        self.stream.start()

    def stop_stream(self):
        """
        Stops the synthetic signal generator.

        """
        self.stream.stop()

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

    def toggle_theme(self, theme_name):
        """
        Toggles between dark and light themes.

        Parameters
        ----------
        theme_name : str
            The name of the theme to switch to.

        """
        self.current_theme = theme_name.lower()
        apply_theme(self, self.current_theme)
        self.apply_theme(theme_name)
        self.plot_window.change_theme(theme_name)


def main():
    """
    Entry point of the application.

    Initializes the synthetic signal generator, launches the main plotting window,
    and opens the control panel.

    """
    app = QApplication(sys.argv)

    # 🌊 Create synthetic data stream
    stream = SyntheticStream()

    # 📈 Launch main plot window
    plot_window = MainWindow()
    # Connecting to a stream
    plot_window.set_data_source(stream)
    plot_window.show()

    # 🎛 Launch control panel
    control_panel = SyntheticControlPanel(stream, plot_window)
    control_panel.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
