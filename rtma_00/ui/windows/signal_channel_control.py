from PyQt6.QtWidgets import (
    QWidget, QLabel, QDial, QVBoxLayout, QHBoxLayout, QSlider, QComboBox,
    QGroupBox, QGridLayout, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import numpy as np

class SignalChannelControl(QWidget):
    def __init__(self, channel_name="Channel 1"):
        super().__init__()
        self.channel_name = channel_name
        self.init_params()
        self.build_ui()
        self.build_timer()
        QTimer.singleShot(0, self._apply_initial_values)

    def init_params(self):
        self.freq = 10
        self.amp = 1.0
        self.phase = 0
        self.offset = 0.0
        self.noise_level = 0.0
        self.dc_bias = 0.0
        self.waveform = "sine"
        self.modulation_type = "None"
        self.running = False
        self.t = 0
        self.fade_factor = 1.0

    def build_ui(self):
        self.group = QGroupBox()
        self.group.setStyleSheet("QGroupBox { border: none; }")
        header_layout = QHBoxLayout()
        self.status_light = QFrame()
        self.status_light.setFixedSize(14, 14)
        self.status_light.setStyleSheet("background-color: red; border-radius: 7px;")
        header_label = QLabel(self.channel_name)
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.status_light)
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        grid = QGridLayout()
        font = QFont("Segoe UI", 10)

        self.freq_dial = QDial()
        self.freq_dial.setRange(1, 200)
        self.freq_dial.setValue(10)
        self.freq_dial.valueChanged.connect(self.update_freq)
        self.freq_label = QLabel("Freq: 10 Hz")
        self.freq_label.setFont(font)

        self.amp_dial = QDial()
        self.amp_dial.setRange(1, 100)
        self.amp_dial.setValue(10)
        self.amp_dial.valueChanged.connect(self.update_amp)
        self.amp_label = QLabel("Amp: 1.0")
        self.amp_label.setFont(font)

        self.phase_slider = QSlider(Qt.Orientation.Horizontal)
        self.phase_slider.setRange(-180, 180)
        self.phase_slider.setValue(0)
        self.phase_slider.valueChanged.connect(self.update_phase)
        self.phase_label = QLabel("Phase: 0°")
        self.phase_label.setFont(font)

        self.offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.offset_slider.setRange(-100, 100)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(self.update_offset)
        self.offset_label = QLabel("Offset: 0.0")
        self.offset_label.setFont(font)

        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(0)
        self.noise_slider.valueChanged.connect(self.update_noise)
        self.noise_label = QLabel("Noise: 0%")
        self.noise_label.setFont(font)

        self.dc_slider = QSlider(Qt.Orientation.Horizontal)
        self.dc_slider.setRange(-100, 100)
        self.dc_slider.setValue(0)
        self.dc_slider.valueChanged.connect(self.update_dc_bias)
        self.dc_label = QLabel("DC Bias: 0.0")
        self.dc_label.setFont(font)

        self.waveform_selector = QComboBox()
        self.waveform_selector.addItems(["sine", "square", "triangle", "sawtooth", "noise", "chirp"])
        self.waveform_selector.currentTextChanged.connect(self.update_waveform)

        self.mod_selector = QComboBox()
        self.mod_selector.addItems(["None", "AM", "FM"])
        self.mod_selector.currentTextChanged.connect(self.update_modulation)

        self.start_btn = QPushButton("▶ Start")
        self.stop_btn = QPushButton("■ Stop")
        self.start_btn.clicked.connect(self.start_channel)
        self.stop_btn.clicked.connect(self.stop_channel)

        grid.addWidget(self.freq_label, 0, 0)
        grid.addWidget(self.freq_dial, 1, 0)
        grid.addWidget(self.amp_label, 0, 1)
        grid.addWidget(self.amp_dial, 1, 1)
        grid.addWidget(self.phase_label, 2, 0)
        grid.addWidget(self.phase_slider, 3, 0, 1, 2)
        grid.addWidget(self.offset_label, 4, 0)
        grid.addWidget(self.offset_slider, 5, 0, 1, 2)
        grid.addWidget(self.noise_label, 6, 0)
        grid.addWidget(self.noise_slider, 7, 0, 1, 2)
        grid.addWidget(self.dc_label, 8, 0)
        grid.addWidget(self.dc_slider, 9, 0, 1, 2)
        grid.addWidget(QLabel("Waveform"), 10, 0)
        grid.addWidget(self.waveform_selector, 10, 1)
        grid.addWidget(QLabel("Modulation"), 11, 0)
        grid.addWidget(self.mod_selector, 11, 1)
        grid.addWidget(self.start_btn, 12, 0)
        grid.addWidget(self.stop_btn, 12, 1)

        self.group.setLayout(grid)
        layout = QVBoxLayout()
        layout.addLayout(header_layout)
        layout.addWidget(self.group)
        self.setLayout(layout)

    def build_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.generate_signal)

    def start_channel(self):
        self.running = True
        self.fade_factor = 1.0
        self.status_light.setStyleSheet("background-color: lightgreen; border-radius: 6px;")
        self.timer.start(50)

    def stop_channel(self):
        self.running = False
        self.fade_factor = 1.0
        self.status_light.setStyleSheet("background-color: red; border-radius: 6px;")

    def generate_signal(self):
        t = self.t
        A = self.amp * self.fade_factor
        val = self.get_waveform_value(t, A)
        self.t += 0.05
        if not self.running:
            self.fade_factor *= 0.9
            if self.fade_factor < 0.01:
                self.fade_factor = 0

    def get_waveform_value(self, t, A):
        f = self.freq
        phi = np.radians(self.phase)
        offset = self.offset
        dc = self.dc_bias
        noise = np.random.normal(0, A * self.noise_level)

        if self.waveform == "sine":
            val = A * np.sin(2 * np.pi * f * t + phi)
        elif self.waveform == "square":
            val = A * np.sign(np.sin(2 * np.pi * f * t + phi))
        elif self.waveform == "triangle":
            val = A * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f * t + phi))
        elif self.waveform == "sawtooth":
            val = A * (2 * (t * f - np.floor(t * f + 0.5)))
        elif self.waveform == "noise":
            val = A * np.random.normal(0, 1)
        elif self.waveform == "chirp":
            val = A * np.sin(2 * np.pi * f * t * t + phi)
        else:
            val = 0

        return val + offset + noise + dc

    def update_freq(self): self.freq = self.freq_dial.value(); self.freq_label.setText(f"Freq: {self.freq} Hz")
    def update_amp(self): self.amp = self.amp_dial.value() / 10.0; self.amp_label.setText(f"Amp: {self.amp:.1f}")
    def update_phase(self): self.phase = self.phase_slider.value(); self.phase_label.setText(f"Phase: {self.phase}°")
    def update_offset(self): self.offset = self.offset_slider.value() / 10.0; self.offset_label.setText(f"Offset: {self.offset:.1f}")
    def update_noise(self): self.noise_level = self.noise_slider.value() / 100.0; self.noise_label.setText(f"Noise: {int(self.noise_level * 100)}%")
    def update_dc_bias(self): self.dc_bias = self.dc_slider.value() / 10.0; self.dc_label.setText(f"DC Bias: {self.dc_bias:.1f}")
    def update_waveform(self, waveform): self.waveform = waveform
    def update_modulation(self, mod): self.modulation_type = mod

    def _apply_initial_values(self):
        self.update_freq()
        self.update_amp()
        self.update_phase()
        self.update_offset()
        self.update_noise()
        self.update_dc_bias()
        self.update_waveform(self.waveform_selector.currentText())
        self.update_modulation(self.mod_selector.currentText())
        self.start_channel()
