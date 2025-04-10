"""
app_RTMA.py

Entry point to start the Real-Time Modal Analyzer.
"""

import sys
from PyQt6.QtWidgets import QApplication
from app.main import run_main_window, logger


def main():
    """
    Initialize and run the main PyQt6 application loop.
    """
    logger.info("Launching application...")
    app = QApplication(sys.argv)
    app.setApplicationName("RTMA - Real-Time Modal Analysis")
    

    main_window = run_main_window()
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

