from PyQt5.QtWidgets import QApplication
from main_ui import MainWindow
from data_sources.synthetic_stream import SyntheticStream
import sys

def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    stream = SyntheticStream()
    window.set_data_source(stream)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
