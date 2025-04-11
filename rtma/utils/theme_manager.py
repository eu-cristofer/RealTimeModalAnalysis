"""
theme_manager.py

This module provides a utility class for managing the application's global theme 
with dynamic live updates. The goal of the ThemeManager is to centralize theme 
handling, enabling the entire application to switch between visual styles 
(such as dark and light modes) easily and consistently.

Philosophy:
-----------

Modern applications often allow users to toggle between different visual themes, 
like light and dark modes. Rather than scattering theme logic throughout the codebase, 
ThemeManager encapsulates this functionality in a centralized, reusable component. 
This design ensures:

1. **Single Source of Truth** - The active theme is maintained in one place.
2. **Signal-Based Communication** - The `theme_changed` signal allows other parts of 
   the application to react to theme changes dynamically.
3. **Simplicity and Robustness** - Theme files (.qss) are loaded from a fixed 
   directory. If a theme file is missing, the app falls back to an empty style safely.

This manager assumes themes are defined as `.qss` (Qt Style Sheets) in the `ui/themes/` 
directory relative to the project root.

Usage:
------
    from theme_manager import theme_manager

    theme_manager.apply_theme("dark")
    theme_manager.toggle_theme()
    current = theme_manager.get_current_theme()

Classes:
--------
    ThemeManager - A QObject-based singleton managing theme application and signaling.

Signals:
--------
    theme_changed (str) - Emitted whenever a new theme is applied.
"""

import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal


class ThemeManager(QObject):
    theme_changed = pyqtSignal(str)
    """
    Signal emitted when the theme changes.

    Parameters:
        name (str): The name of the newly applied theme.
    """

    def __init__(self):
        super().__init__()
        self._current_theme = "dark"
        self._last_applied = None
        self._theme_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui", "themes"))

    def apply_theme(self, name: str):
        """
        Applies the given theme by loading the corresponding QSS file.
        If the theme has already been applied, this function does nothing.

        Args:
            name (str): The name of the theme to apply.
        """
        if name == self._last_applied:
            return

        self._current_theme = name
        self._last_applied = name

        app = QApplication.instance()
        if app:
            style_path = os.path.join(self._theme_dir, f"{name}.qss")
            try:
                with open(style_path, "r") as f:
                    app.setStyleSheet(f.read())
            except FileNotFoundError:
                self._safe_print(f"[WARNING] Theme file not found: {style_path}")
                app.setStyleSheet("")

        self.theme_changed.emit(name)

    def _safe_print(self, msg: str):
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="ignore").decode())

    def toggle_theme(self):
        """
        Toggles between 'light' and 'dark' themes.
        """
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self.apply_theme(new_theme)

    def get_current_theme(self):
        """
        Returns the name of the currently active theme.

        Returns:
            str: The current theme name.
        """
        return self._current_theme


# Singleton instance of ThemeManager for application-wide use
theme_manager = ThemeManager()
