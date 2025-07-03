from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtCore import Qt
from rtma.utils.theme_manager import theme_manager


class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.label_on = "🌙 Dark Mode"
        self.label_off = "☀️ Light Mode"

        # Setup visuals
        self.setTristate(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._switch_style())

        # Sync state with theme
        self.setChecked(theme_manager.get_current_theme() == "dark")
        self._update_label(self.isChecked())

        # Connect signals
        self.toggled.connect(self._on_toggle)
        theme_manager.theme_changed.connect(self._on_theme_change)

    def _on_toggle(self):
        '''
        When the user toggles the switch, _on_toggle is called.

        NOTE:
        =====
        It delegates theme toggling to theme_manager.
        '''
        theme_manager.toggle_theme()

    def _on_theme_change(self, theme):
        '''
        When the theme changes (from somewhere else), _on_theme_change updates the UI accordingly
        '''
        self.blockSignals(True)
        self.setChecked(theme == "dark")
        self.blockSignals(False)
        self._update_label(theme == "dark")

    def _update_label(self, is_dark):
        self.setText(self.label_on if is_dark else self.label_off)

    def _switch_style(self):
        return """
        QCheckBox::indicator { width: 36px; height: 18px; }
        QCheckBox::indicator:unchecked {
            border: 1px solid #aaa;
            border-radius: 9px;
            background-color: #ccc;
        }
        QCheckBox::indicator:checked {
            border: 1px solid #666;
            border-radius: 9px;
            background-color: #4caf50;
        }
        """
