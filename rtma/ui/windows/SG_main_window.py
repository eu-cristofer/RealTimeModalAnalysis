"""
SG_main_window.py

Main window for the Signal Generator Station.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QToolBar, QPushButton, QComboBox, QCheckBox,
    QWidget, QVBoxLayout, QTabWidget, QLabel
)
from PyQt6.QtCore import QTimer

from ui.components.chart_widget import ChartWidget
from ui.windows.signal_generator_desk import SignalGeneratorDesk
from utils import theme_manager


class SGMainWindow(QMainWindow):
    """
    The main application window for the Signal Generator.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Generator Station")
        self.resize(792, 900)  # Adjust width to show 3 waveform modules cleanly

        self.plot_enabled = [True, True, True, True]
        self.channel_tabs = []

        self._setup_ui()
        self._setup_timer()
                
        theme_manager.apply_theme("dark")
        self.chart.set_theme("dark")  # make sure chart reflects initial theme

    def _setup_ui(self):
        self._setup_toolbar()

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Chart area at the top
        self.chart = ChartWidget(title="Combined Channel Output")
        main_layout.addWidget(self.chart)

        # Channel tabs at the bottom
        self.tab_widget = QTabWidget()
        for i in range(4):
            tab = SignalGeneratorDesk(channel_id=i)
            tab.plot_checkbox.stateChanged.connect(lambda state, idx=i: self._toggle_plot(idx, state))
            self.channel_tabs.append(tab)
            self.tab_widget.addTab(tab, f"CH{i+1}")

        main_layout.addWidget(self.tab_widget)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        self.start_btn = QPushButton("▶️ Start Signal")
        self.stop_btn = QPushButton("⏹️ Stop Signal")
        self.theme_toggle = QPushButton("☀️🌙 Toggle Light/Dark Mode")
        self.broadcast_btn = QPushButton("📡 Start Broadcast")

        self.output_selector = QComboBox()
        self.output_selector.addItems(["RTMA", "OPC"])

        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addWidget(self.theme_toggle)
        toolbar.addWidget(self.broadcast_btn)
        toolbar.addWidget(QLabel("Output:"))
        toolbar.addWidget(self.output_selector)

        self.start_btn.clicked.connect(self.start_signal)
        self.stop_btn.clicked.connect(self.stop_signal)
        self.theme_toggle.clicked.connect(self._toggle_theme)

    def _toggle_plot(self, channel_index, state):
        self.plot_enabled[channel_index] = bool(state)

    def _toggle_theme(self):
        from utils.theme_manager import toggle_theme, get_current_theme
        toggle_theme()
        self.chart.set_theme(get_current_theme())

    def start_signal(self):
        self.timer.start()

    def stop_signal(self):
        self.timer.stop()

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._update_chart)

    def _update_chart(self):
        """
        Collects signals from each channel and updates the combined chart.
        """
        signal_data = {}

        for i, tab in enumerate(self.channel_tabs):
            if tab.plot_checkbox.isChecked():
                signal = tab.generate_signal()
                signal_data[i] = signal

        self.chart.plot_multiple(signal_data)
