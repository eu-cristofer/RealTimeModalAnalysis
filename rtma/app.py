"""
app.py

Single entry point for launching the RTMA application suite
with CLI and UI support
"""

import sys
import argparse
from logging import DEBUG, INFO
from PyQt6.QtWidgets import QApplication, QSplashScreen, QMainWindow
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
from rtma.utils.logger import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Launch RTMA Suite")
    parser.add_argument(
        "--mode",
        choices=["sg", "rtma", "both"],
        help="Launch in a specific mode: sg (Signal Generator); rtma (Analyzer); or both (both apps)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Default is disabled",
    )

    # Parse the args
    args = parser.parse_args()

    # --- Setup Logging Before Anything Else ---
    setup_logging(
        level= DEBUG if args.debug else INFO,
        log_to_file=args.debug,
        log_file="logs/cocoa_engine_ui.log"
    )

    log = get_logger(__name__)
    log.info("Application starting...")

    # Now import modules that may use logging
    log.debug("Importing modules that uses logger.")
    from rtma.window_selector import run_main_window
    from rtma.ui.windows.launcher_window import LauncherWindow


    # For any GUI application using Qt, there is precisely one QApplication
    # object, no matter whether the application has 0, 1, 2 or more windows
    # at any given time.
    app = QApplication(sys.argv)

    # --- Splash screen setup ---
    splash_pix = QPixmap("rtma/ui/splash/splash_01.png")
    splash = QSplashScreen(splash_pix,
                           Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Loading RTMA Suite...",
                       alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                       color=Qt.GlobalColor.white)

    # Hold references to all windows to prevent garbage collection
    windows = {}

    def launch():
        if args.mode == "sg":
            win = run_main_window("sg")
            win.show()
            windows["sg"] = win
            splash.finish(win)

        elif args.mode == "rtma":
            win = run_main_window("rtma")
            win.show()
            windows["rtma"] = win
            splash.finish(win)

        elif args.mode == "both":
            win1 = run_main_window("sg")
            win2 = run_main_window("rtma")
            win1.show()
            win2.show()
            windows["sg"] = win1
            windows["rtma"] = win2
            splash.finish(win1)  # Pick one to anchor the splash to

        else:
            launcher = LauncherWindow()
            launcher.show()
            windows["launcher"] = launcher
            splash.finish(launcher)
    
    # Simulate a loading delay before launching the main window
    QTimer.singleShot(500, launch)

    # Running the main event loop
    sys.exit(app.exec())
    '''
    NOTE:
    =====
    The call to .exec() is wrapped in a call to sys.exit(), which allows you to
    cleanly exit Python and release memory resources when the application terminates.
    '''

if __name__ == "__main__":
    main()

