"""
app_RTMS.py

Single entry point for launching the RTMA application suite
with CLI and UI support
"""

import sys
import argparse
from PyQt6.QtWidgets import QApplication
from ui.windows.launcher_window import LauncherWindow


def main():
    parser = argparse.ArgumentParser(description="Launch RTMA Suite")
    parser.add_argument(
        "--mode", choices=["sg", "rtma", "both"],
        help="Launch in a specific mode: sg (Signal Generator), rtma (Analyzer), both (both apps)"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    if args.mode == "sg":
        win = run_main_window("sg")
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

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

