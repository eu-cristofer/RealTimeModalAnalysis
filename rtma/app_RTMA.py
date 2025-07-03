"""
app_RTMA.py

Launches the Real-Time Modal Analyzer (RTMA) with stream dock interface.
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui.windows.RTMA_main_window import RTMAMainWindow
from utils.theme_manager import apply_theme

def main():
    app = QApplication(sys.argv)
    apply_theme("dark")  # or "light"
    win = RTMAMainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
