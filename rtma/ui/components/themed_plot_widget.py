"""
themed_plot_widget.py

A QWidget subclass that integrates with the global ThemeManager to dynamically
adjust plot appearance based on the active application theme.

This widget assumes it contains an attribute `plot_widget` that supports methods
like `setBackground`, `getAxis`, and `showGrid`, such as those from PyQtGraph.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSlot
from utils.theme_manager import theme_manager


class ThemedPlotWidget(QWidget):
    def __init__(self, parent=None):
        """
        Initializes the themed plot widget and connects it to the global theme change signal.
        """
        super().__init__(parent)
        self._theme_applied = None
        theme_manager.theme_changed.connect(self.set_theme)

    @pyqtSlot(str)
    def set_theme(self, theme: str):
        """
        Slot that applies a visual style to the plot based on the selected theme.

        Args:
            theme (str): The name of the theme ('light' or 'dark').
        """
        if theme == self._theme_applied:
            return  # 🚫 prevent recursion
        self._theme_applied = theme

        if not hasattr(self, 'plot_widget'):
            return  # No plot to theme yet

        try:
            if theme == "light":
                self.plot_widget.setBackground("w")
                self.plot_widget.getAxis("left").setPen("black")
                self.plot_widget.getAxis("bottom").setPen("black")
                self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
            else:
                self.plot_widget.setBackground("k")
                self.plot_widget.getAxis("left").setPen("white")
                self.plot_widget.getAxis("bottom").setPen("white")
                self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        except Exception as e:
            print("❌ Theme error in set_theme():", e)
