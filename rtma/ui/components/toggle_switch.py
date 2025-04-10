from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtCore import Qt

class ToggleSwitch(QCheckBox):
    def __init__(self, label_on="🌙 Dark Mode", label_off="☀️ Light Mode", parent=None):
        super().__init__(label_on, parent)
        self.label_on = label_on
        self.label_off = label_off
        self.setChecked(True)
        self.setTristate(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._switch_style())
        self.toggled.connect(self._update_label)

    def _update_label(self, checked):
        self.setText(self.label_on if checked else self.label_off)

    def _switch_style(self):
        return """
        QCheckBox::indicator { width: 40px; height: 20px; }
        QCheckBox::indicator:unchecked {
            image: url(none);
            border: 1px solid #aaa;
            border-radius: 10px;
            background-color: #ccc;
        }
        QCheckBox::indicator:checked {
            image: url(none);
            border: 1px solid #666;
            border-radius: 10px;
            background-color: #4caf50;
        }
        """
