"""
Synthetic Signal Generator Control Panel
========================================

A PyQt5-based GUI for configuring multi-channel synthetic signals with:
- Multiple waveform components per channel
- Independent parameters per component
- Fade in/out controls
- Theme support
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QPushButton, QGroupBox, QComboBox, QTabWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from main_ui import MainWindow
from data_sources.synthetic_stream import SyntheticStream
from styles import apply_theme
from theme_engine import load_theme, generate_stylesheet

import logging

class SyntheticControlPanel(QMainWindow):
    """
    Main control panel for synthetic signal generator.

    Parameters
    ----------
    stream : SyntheticStream
        Synthetic data stream instance
    plot_window : MainWindow
        Main visualization window

    Attributes
    ----------
    stream : SyntheticStream
        Connected data stream
    plot_window : MainWindow
        Visualization window
    current_theme : str
        Active theme name ('dark' or 'light')
    tabs : QTabWidget
        Channel configuration tabs
    channel_tabs : list
        Channel tab configurations
    fade_in_spin : QDoubleSpinBox
        Fade-in duration control
    fade_out_spin : QDoubleSpinBox
        Fade-out duration control
    """

    def __init__(self, stream: SyntheticStream, plot_window: MainWindow):
        """Initialize control panel with stream and plot window."""
        super().__init__()
        self.stream = stream
        self.plot_window = plot_window

        # Current theme
        self.current_theme = "dark"

        # Initialize component storage
        self.channel_components = [[] for _ in range(4)]

        # Start the user interface
        self._init_ui()

        # Apply initial style
        self.toggle_theme(self.current_theme)

        # Novo código para logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("modal_analyzer.log"),
                logging.StreamHandler()
            ]
        )
        logging.info("Synthetic initialized")

    def _init_ui(self):
        """
        Initialize user interface components.
        """
        # Window setup
        self.setWindowTitle("Multi-Channel Signal Generator")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("4-Channel Synthetic Signal Generator")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Initialize channel tabs FIRST
        self._init_channel_tabs()
        layout.addWidget(self.tabs)

        # Then add fade controls
        fade_group = self._create_fade_controls()
        layout.addWidget(fade_group)

        # Theme selector
        theme_layout = self._create_theme_controls()
        layout.addLayout(theme_layout)

        # Global controls
        global_btn_layout = self._create_global_buttons()
        layout.addLayout(global_btn_layout)
        self.global_start_btn.setEnabled(True)
        self.global_stop_btn.setEnabled(False)

        # Initialize component tracking
        self.channel_components = [[] for _ in range(4)]

        self._update_params()

    def _create_spin(self, layout, label, default, minv, maxv):
        """
        Create labeled spinbox widget.

        Parameters
        ----------
        layout : QLayout
            Parent layout
        label : str
            Spinbox label
        default : float
            Default value
        minv : float
            Minimum value
        maxv : float
            Maximum value

        Returns
        -------
        QDoubleSpinBox
            Configured spinbox
        """
        row = QHBoxLayout()
        lbl = QLabel(label)
        spin = QDoubleSpinBox()
        spin.setRange(minv, maxv)
        spin.setValue(default)
        spin.valueChanged.connect(self._update_params)
        row.addWidget(lbl)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin
    
    def _init_channel_tabs(self):
        """Create tabbed interface for channel configuration"""
        self.tabs = QTabWidget()
        self.channel_tabs = []  # Track layout and components for each tab

        for ch_idx in range(4):
            tab = QWidget()
            tab_layout = QVBoxLayout()
            tab_layout.setContentsMargins(10, 10, 10, 10)

            # Component container
            components_container = QWidget()
            components_layout = QVBoxLayout()
            components_layout.setSpacing(10)
            components_container.setLayout(components_layout)

            # Add Component button
            add_btn = QPushButton("➕ Add Component")
            add_btn.clicked.connect(lambda _, ch=ch_idx: self._add_component(ch))

            # Channel control buttons
            channel_controls = self._create_channel_controls(ch_idx)

            tab_layout.addWidget(components_container)
            tab_layout.addWidget(add_btn)
            tab_layout.addWidget(channel_controls)
            tab.setLayout(tab_layout)

            self.tabs.addTab(tab, f"Channel {ch_idx + 1}")

            # 🔧 Track tab, layout and components properly
            self.channel_tabs.append({
                "widget": tab,
                "layout": components_layout,
                "components": []
            })

    def _apply_channel_config(self, ch_idx):
        self.stream.channels[ch_idx].components = []
        for comp in self.channel_components[ch_idx]:
            self.stream.channels[ch_idx].components.append({
                "waveform": comp["widgets"]["waveform"].currentText(),
                "freq": comp["widgets"]["freq"].value(),
                "amp": comp["widgets"]["amp"].value(),
                "phase": comp["widgets"]["phase"].value()
            })


    def _create_channel_controls(self, ch_idx):
        """Create per-channel action buttons"""
        control_group = QGroupBox("Channel Actions")
        layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(lambda: self._apply_channel_config(ch_idx))
        
        start_btn = QPushButton("Start")
        start_btn.clicked.connect(lambda: self.stream.start_channel(ch_idx))
        
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(lambda: self.stream.stop_channel(ch_idx))
        
        layout.addWidget(apply_btn)
        layout.addWidget(start_btn)
        layout.addWidget(stop_btn)
        control_group.setLayout(layout)
        
        return control_group

    def _create_fade_controls(self):
        """Create fade time controls"""
        group = QGroupBox("Fade Settings")
        layout = QHBoxLayout()
        
        self.fade_in_spin = self._create_labeled_spin(
            layout, "Fade In (s):", 1.0, 0, 10
        )
        self.fade_out_spin = self._create_labeled_spin(
            layout, "Fade Out (s):", 1.0, 0, 10
        )
        
        group.setLayout(layout)
        return group

    def _create_theme_controls(self):
        """Create theme selection controls"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Theme:"))
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self.toggle_theme)
        
        layout.addWidget(self.theme_combo)
        layout.addStretch()
        return layout

    def _create_global_buttons(self):
        """Create global control buttons"""
        layout = QHBoxLayout()
        
        self.global_start_btn = QPushButton("Start All")
        self.global_stop_btn = QPushButton("Stop All")
        
        self.global_start_btn.clicked.connect(self._handle_start_all)
        self.global_stop_btn.clicked.connect(self._handle_stop_all)
        
        layout.addWidget(self.global_start_btn)
        layout.addWidget(self.global_stop_btn)
        return layout

    def _add_component(self, ch_idx):
        """Add new waveform component to specified channel."""
        component = {
            "widgets": {
                "waveform": QComboBox(),
                "freq": QDoubleSpinBox(),
                "amp": QDoubleSpinBox(),
                "phase": QDoubleSpinBox()
            }
        }
        
        # Create component UI
        group = QGroupBox()
        layout = QVBoxLayout()
        
        # Waveform selector
        wave_combo = QComboBox()
        wave_combo.addItems(["sine", "square", "sawtooth", "triangle"])
        
        # Parameter controls
        freq_spin = self._create_spin(layout, "Frequency (Hz):", 1.0, 0.1, 1000)
        amp_spin = self._create_spin(layout, "Amplitude:", 1.0, 0.01, 10)
        phase_spin = self._create_spin(layout, "Phase (rad):", 0.0, -np.pi, np.pi)
        
        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_component(ch_idx, group))
        
        layout.insertWidget(0, wave_combo)
        layout.addWidget(remove_btn)
        group.setLayout(layout)
        
        # Store references
        component["widgets"] = {
            "waveform": wave_combo,
            "freq": freq_spin,
            "amp": amp_spin,
            "phase": phase_spin
        }
        
        # Add to channel
        self.channel_tabs[ch_idx]["components"].append(component)
        self.channel_tabs[ch_idx]["widget"].layout().insertWidget(
            len(self.channel_tabs[ch_idx]["components"])-1,
            group
        )

        # Store reference
        self.channel_components[ch_idx].append(component)

    def _handle_start_all(self):
        self.stream.start_all()
        logging.info("All channels started")
        self.global_start_btn.setEnabled(False)
        self.global_stop_btn.setEnabled(True)

    def _handle_stop_all(self):
        self.stream.stop_all()
        logging.info("All channels stopped")
        self.global_start_btn.setEnabled(True)
        self.global_stop_btn.setEnabled(False)

    def _remove_component(self, ch_idx, widget):
        """Remove component from specified channel."""
        self.channel_components[ch_idx] = [
            c for c in self.channel_components[ch_idx]
            if c["widgets"]["waveform"].parent().parent() != widget
        ]
        widget.deleteLater()
        self._update_params()

    def _create_labeled_spin(self, layout, label, default, minv, maxv):
        """Create a labeled spinbox (renamed from _add_spin)"""
        row = QHBoxLayout()
        lbl = QLabel(label)
        spin = QDoubleSpinBox()
        spin.setRange(minv, maxv)
        spin.setValue(default)
        spin.valueChanged.connect(self._update_params)
        row.addWidget(lbl)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def _update_params(self):
        """Update stream parameters from UI controls."""
        # Fade times
        self.stream.fade_in_time = self.fade_in_spin.value()
        self.stream.fade_out_time = self.fade_out_spin.value()
        
        # Channel components
        for ch_idx in range(4):
            # Clear existing components
            self.stream.channels[ch_idx].components = []

            # Add new components
            for comp in self.channel_components[ch_idx]:
                self.stream.channels[ch_idx].components.append({
                    "waveform": comp["widgets"]["waveform"].currentText(),
                    "freq": comp["widgets"]["freq"].value(),
                    "amp": comp["widgets"]["amp"].value(),
                    "phase": comp["widgets"]["phase"].value()
                })

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

    def toggle_theme(self, theme_name):
        """
        Changes the current theme and updates the application stylesheet.

        Parameters
        ----------
        theme_name : str
            Name of theme to apply ('Dark' or 'Light')
        """
        self.current_theme = theme_name.lower()
        apply_theme(self, self.current_theme)
        self.apply_theme(theme_name)

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    stream = SyntheticStream()
    plot_window = MainWindow()
    plot_window.set_data_source(stream)
    
    control_panel = SyntheticControlPanel(stream, plot_window)
    
    plot_window.show()
    control_panel.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()