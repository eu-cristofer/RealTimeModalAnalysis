"""
waveform_control.py

UI control for configuring a single waveform component in a DAW-style layout.

This widget allows users to select a waveform type and adjust its parameters
(interactively or manually) including amplitude, frequency, phase, and offset.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QSlider, QDial, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from rtma.core.waveforms import (
    SineWave, SquareWave, TriangleWave, SawtoothWave,
    NoiseWave, PinkNoiseWave, WaveformComponent
)


class WaveformControl(QWidget):
    """
    UI widget for editing and configuring a waveform component.

    This panel includes:
        - A dropdown to select waveform type
        - Sliders for amplitude and frequency (with editable fields)
        - Dials for phase and offset (with live value labels)
        - A remove button

    Signals
    -------
    changed : pyqtSignal
        Emitted when any control is modified.
    removed : pyqtSignal
        Emitted when the "Remove" button is clicked.
    """

    changed = pyqtSignal()
    removed = pyqtSignal(object)

    def __init__(self, parent=None):
        """
        Initialize the waveform control UI.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setFixedWidth(240)
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self._build_groupbox(), alignment=Qt.AlignmentFlag.AlignLeft)
        self.setLayout(outer_layout)

    def _build_groupbox(self) -> QGroupBox:
        """
        Build the full waveform UI, wrapped in a framed QGroupBox.

        Returns
        -------
        QGroupBox
            The group box containing the entire waveform control module.
        """
        box = QGroupBox("Waveform Component")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Waveform Type Dropdown ---
        self.waveform_type = QComboBox()
        self.waveform_type.setFixedWidth(200)
        self.waveform_type.addItems([
            "Sine", "Square", "Triangle", "Sawtooth",
            "White Noise", "Pink Noise"
        ])
        self.waveform_type.currentIndexChanged.connect(self.changed.emit)

        layout.addWidget(QLabel("Waveform Type"))
        layout.addWidget(self.waveform_type)

        # --- Amplitude & Frequency Sliders ---
        top_controls = QHBoxLayout()
        self._init_amp_slider(top_controls)
        self._init_freq_slider(top_controls)
        layout.addLayout(top_controls)

        # --- Phase & Offset Dials ---
        dial_controls = QHBoxLayout()
        self._init_phase_dial(dial_controls)
        self._init_offset_dial(dial_controls)
        layout.addLayout(dial_controls)

        # --- Remove Button ---
        self.remove_btn = QPushButton("❌ Remove")
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.remove_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        box.setLayout(layout)
        return box


    def _init_amp_slider(self, parent_layout):
        """
        Initialize amplitude control slider and text input.

        Parameters
        ----------
        parent_layout : QHBoxLayout
            The layout to which the control is added.
        """
        self.amp_slider = QSlider(Qt.Orientation.Vertical)
        self.amp_slider.setMinimumHeight(40)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.setValue(10)
        self.amp_slider.setTickInterval(10)

        self.amp_value_edit = QLineEdit("1.0")
        self.amp_value_edit.setFixedWidth(50)
        self.amp_value_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.amp_slider.valueChanged.connect(self.update_amp_from_slider)
        self.amp_value_edit.editingFinished.connect(self.update_amp_from_edit)
        self.amp_slider.valueChanged.connect(self.changed.emit)

        amp_layout = QVBoxLayout()
        amp_layout.addWidget(QLabel("Amplitude"))
        amp_layout.addWidget(self.amp_slider)
        amp_layout.addWidget(self.amp_value_edit)
        parent_layout.addLayout(amp_layout)

    def _init_freq_slider(self, parent_layout):
        """
        Initialize frequency control slider and text input.

        Parameters
        ----------
        parent_layout : QHBoxLayout
            The layout to which the control is added.
        """
        self.freq_slider = QSlider(Qt.Orientation.Vertical)
        self.freq_slider.setRange(1, 500)
        self.freq_slider.setValue(10)
        self.freq_slider.setTickInterval(50)

        self.freq_value_edit = QLineEdit("10")
        self.freq_value_edit.setFixedWidth(50)
        self.freq_value_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.freq_slider.valueChanged.connect(self.update_freq_from_slider)
        self.freq_value_edit.editingFinished.connect(self.update_freq_from_edit)
        self.freq_slider.valueChanged.connect(self.changed.emit)

        freq_layout = QVBoxLayout()
        freq_layout.addWidget(QLabel("Frequency"))
        freq_layout.addWidget(self.freq_slider)
        freq_layout.addWidget(self.freq_value_edit)
        parent_layout.addLayout(freq_layout)

    def _init_phase_dial(self, parent_layout):
        """
        Initialize the phase dial.

        Parameters
        ----------
        parent_layout : QHBoxLayout
            The layout to which the control is added.
        """
        self.phase_dial = QDial()
        self.phase_dial.setRange(-180, 180)
        self.phase_dial.setValue(0)
        self.phase_dial.setNotchesVisible(True)

        self.phase_value_label = QLabel("0°")
        self.phase_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.phase_dial.valueChanged.connect(
            lambda v: self.phase_value_label.setText(f"{v}°")
        )
        self.phase_dial.valueChanged.connect(self.changed.emit)

        phase_layout = QVBoxLayout()
        phase_layout.addWidget(QLabel("Phase"))
        phase_layout.addWidget(self.phase_dial)
        phase_layout.addWidget(self.phase_value_label)
        parent_layout.addLayout(phase_layout)

    def _init_offset_dial(self, parent_layout):
        """
        Initialize the offset dial.

        Parameters
        ----------
        parent_layout : QHBoxLayout
            The layout to which the control is added.
        """
        self.offset_dial = QDial()
        self.offset_dial.setRange(-100, 100)
        self.offset_dial.setValue(0)
        self.offset_dial.setNotchesVisible(True)

        self.offset_value_label = QLabel("0.0")
        self.offset_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.offset_dial.valueChanged.connect(
            lambda v: self.offset_value_label.setText(f"{v / 10.0:.1f}")
        )
        self.offset_dial.valueChanged.connect(self.changed.emit)

        offset_layout = QVBoxLayout()
        offset_layout.addWidget(QLabel("Offset"))
        offset_layout.addWidget(self.offset_dial)
        offset_layout.addWidget(self.offset_value_label)
        parent_layout.addLayout(offset_layout)

    def update_amp_from_slider(self, v):
        """Update amplitude text box when slider is moved."""
        self.amp_value_edit.setText(f"{v / 10.0:.1f}")

    def update_amp_from_edit(self):
        """Update amplitude slider when value is typed in."""
        try:
            val = float(self.amp_value_edit.text())
            self.amp_slider.setValue(int(val * 10))
        except ValueError:
            pass

    def update_freq_from_slider(self, v):
        """Update frequency text box when slider is moved."""
        self.freq_value_edit.setText(str(v))

    def update_freq_from_edit(self):
        """Update frequency slider when value is typed in."""
        try:
            val = int(self.freq_value_edit.text())
            self.freq_slider.setValue(val)
        except ValueError:
            pass

    def to_component(self) -> WaveformComponent:
        """
        Construct and return a waveform component from UI state.

        Returns
        -------
        WaveformComponent
            A configured instance of the selected waveform.
        """
        wave_type = self.waveform_type.currentText()
        amplitude = self.amp_slider.value() / 10.0
        frequency = self.freq_slider.value()
        phase_rad = self.phase_dial.value() * 3.14159 / 180
        offset = self.offset_dial.value() / 10.0

        if wave_type == "Sine":
            return SineWave(amplitude, frequency, phase_rad, offset)
        elif wave_type == "Square":
            return SquareWave(amplitude, frequency, phase_rad, offset)
        elif wave_type == "Triangle":
            return TriangleWave(amplitude, frequency, phase_rad, offset)
        elif wave_type == "Sawtooth":
            return SawtoothWave(amplitude, frequency, phase_rad, offset)
        elif wave_type == "Pink Noise":
            return PinkNoiseWave(amplitude, offset=offset)
        else:
            return NoiseWave(amplitude, offset=offset)
