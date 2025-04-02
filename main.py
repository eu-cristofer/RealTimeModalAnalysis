"""
Base GUI Module
===============

This module is the entry point of a PyQt5-based GUI application for real-time modal analysis.

It initializes the application, sets up the main window, and starts the event loop. 
The application provides interactive tools for visualizing signals in the time domain, 
frequency domain, and cross-correlation plots, along with controls for adjusting FFT parameters 
and themes.

Dependencies
------------
- PyQt5
- main_ui (custom module for the main application window)

Functions
---------
main()
    Initializes the PyQt5 application, sets up the main window, and starts the event loop.

Examples
--------
Run this module to start the application:

>>> python base_gui.py
"""
import sys
from PyQt5.QtWidgets import QApplication
from main_ui import MainWindow


def main():
    """
    Entry point of the application.

    This function initializes the PyQt5 application, creates the main window,
    and starts the event loop.

    Notes
    -----
    - This function uses QApplication to manage the application lifecycle.
    - The `MainWindow` class is imported from the `main_ui` module.

    """
    app = QApplication(sys.argv)  # Create the application instance
    window = MainWindow()         # Instantiate the main window
    window.show()                 # Display the main window
    sys.exit(app.exec_())         # Start the event loop and exit on close


if __name__ == "__main__":
    main()
