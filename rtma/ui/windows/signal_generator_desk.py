"""
signal_generator_desk.py

Channel configuration panel for signal generation.

Manages waveform components for one channel, and provides UI to add/remove
components and enable/disable plotting.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox
)
from PyQt6.QtCore import Qt

from rtma.ui.components.waveform_control import WaveformControl
from rtma.core.signal_channel import SignalChannel
from rtma.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


class SignalGeneratorDesk(QWidget):
    """
    Per-channel configuration widget for waveform generation.
    """

    def __init__(self, channel_id: int, parent=None):
        """
        Parameters
        ----------
        channel_id : int
            Index of the signal channel (0 to 3).
        parent : QWidget, optional
            Parent widget.
        """
        super().__init__(parent)
        self.channel_id = channel_id
        self.plot_checkbox = QCheckBox("Plot this channel")
        self.plot_checkbox.setChecked(True)

        self.waveform_controls = []
        self.signal_channel = SignalChannel(sample_rate=1000, duration=1.0)

        self._init_ui()

    def _init_ui(self):
        """
        Set up the user interface for this channel tab.
        """
        logger.info(f"Initializing UI for channel {self.channel_id + 1}")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Header row: label + plot checkbox + add button
        header = QHBoxLayout()
        header.addWidget(QLabel(f"Configure Channel {self.channel_id + 1}"))

        header.addStretch()
        header.addWidget(self.plot_checkbox)

        self.add_button = QPushButton("+ Add Component")
        self.add_button.clicked.connect(self.add_waveform_control)
        header.addWidget(self.add_button)

        layout.addLayout(header)

        # Waveform controls row
        self.wave_container = QWidget()
        self.wave_layout = QHBoxLayout()
        self.wave_container.setLayout(self.wave_layout)
        self.add_waveform_control()
        layout.addWidget(self.wave_container)

        # # "+ Add Component" row (right-aligned, compact)
        # add_row = QHBoxLayout()
        # add_row.addStretch()
        # self.add_button = QPushButton("+ Add Component")
        # self.add_button.clicked.connect(self.add_waveform_control)
        # add_row.addWidget(self.add_button)
        # layout.addLayout(add_row)

        self.setLayout(layout)

    def add_waveform_control(self):
        """
        Add a new waveform configuration control to the layout.
        Limits to 3 components per channel.
        """
        if len(self.waveform_controls) >= 3:
            return  # Max limit reached
        wc = WaveformControl()
        wc.removed.connect(self.remove_waveform_control)
        self.waveform_controls.append(wc)
        self.wave_layout.addWidget(wc)

    def remove_waveform_control(self, control):
        """
        Remove a waveform control from the layout and UI.

        Parameters
        ----------
        control : WaveformControl
            The widget to remove.
        """
        if control in self.waveform_controls:
            self.waveform_controls.remove(control)
            control.setParent(None)

    def generate_signal(self) -> np.ndarray:
        """
        Generate signal from all active waveform components.

        Returns
        -------
        np.ndarray
            Composite signal for this channel.
        """
        self.signal_channel.components.clear()
        for wc in self.waveform_controls:
            self.signal_channel.add_component(wc.to_component())
        return self.signal_channel.generate_signal()
