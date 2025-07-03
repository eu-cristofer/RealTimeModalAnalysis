"""
app_RTMS.py

Single entry point for launching the RTMA application suite
with CLI and UI support
"""

import sys
import argparse
from PyQt6.QtWidgets import QApplication
from ui.windows.launcher_window import LauncherWindow
from app.main import run_main_window  # ✅ Required to resolve the function


def main():
    parser = argparse.ArgumentParser(description="Launch RTMA Suite")
    parser.add_argument(
        "--mode", choices=["sg", "rtma", "both"],
        help="Launch in a specific mode: sg (Signal Generator); rtma (Analyzer); or both (both apps)"
    )
    args = parser.parse_args()

    # For any GUI application using Qt, there is precisely one QApplication
    # object, no matter whether the application has 0, 1, 2 or more windows
    # at any given time.
    app = QApplication(sys.argv)

    # Creating and showing the application GUI
    if args.mode == "sg":
        win = run_main_window("sg")
        # The call to .show() schedules a paint event, which is a request to
        # paint the widgets that compose a GUI. This event is then added to
        # the application’s event queue
        win.show()
    elif args.mode == "rtma":
        win = run_main_window("rtma")
        win.show()
    elif args.mode == "both":
        win1 = run_main_window("sg")
        win2 = run_main_window("rtma")
        win1.show()
        win2.show()
    else:
        launcher = LauncherWindow()
        launcher.show()

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

