"""
frameless_window.py

A reusable frameless window with custom title bar and drag behavior.
"""

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QMouseEvent


class FramelessWindow(QMainWindow):
    def __init__(self, title="App", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._drag_position = QPoint()

        # Central widget + layout
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Add custom title bar
        self._add_title_bar(title)

        self.setCentralWidget(self.central_widget)

    def _add_title_bar(self, title: str):
        title_bar = QWidget()
        title_bar.setObjectName("TitleBar")
        title_bar.setStyleSheet("""
            QWidget#TitleBar {
                background-color: #2c2c2e;
                padding: 6px;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #444;
                border-radius: 4px;
            }
        """)

        layout = QHBoxLayout()
        layout.setContentsMargins(10, 2, 10, 2)

        self.title_label = QLabel(title)
        btn_min = QPushButton("–")
        btn_close = QPushButton("✕")

        btn_min.clicked.connect(self.showMinimized)
        btn_close.clicked.connect(self.close)

        layout.addWidget(self.title_label)
        layout.addStretch()
        layout.addWidget(btn_min)
        layout.addWidget(btn_close)

        title_bar.setLayout(layout)
        self.main_layout.addWidget(title_bar)

    def set_content_widget(self, widget: QWidget):
        self.main_layout.addWidget(widget)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_position
            self.move(self.pos() + delta)
            self._drag_position = event.globalPosition().toPoint()
