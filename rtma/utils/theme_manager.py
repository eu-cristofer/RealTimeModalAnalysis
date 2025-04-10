"""
theme_manager.py

Utility for managing global application theme.
"""

import os
from PyQt6.QtWidgets import QApplication

_current_theme = "dark"  # default
_base_dir = os.path.dirname(os.path.abspath(__file__))
_theme_dir = os.path.abspath(os.path.join(_base_dir, "..", "ui", "themes"))

def apply_theme(theme_name: str):
    """
    Apply the specified theme to the entire application.

    Parameters
    ----------
    theme_name : str
        Either "dark" or "light"
    """
    global _current_theme
    _current_theme = theme_name

    app = QApplication.instance()
    if not app:
        return

    style_path = os.path.join(_theme_dir, f"{theme_name}.qss")
    try:
        with open(style_path, "r") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print(f"Theme file not found: {style_path}")
        app.setStyleSheet("")


def toggle_theme():
    """
    Toggle between light and dark themes.
    """
    global _current_theme
    new_theme = "light" if _current_theme == "dark" else "dark"
    apply_theme(new_theme)


def get_current_theme():
    return _current_theme
