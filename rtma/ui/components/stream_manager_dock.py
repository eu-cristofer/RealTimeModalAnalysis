"""
stream_manager_dock.py

Dockable UI panel to manage and add new stream sources.
"""

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QListWidget, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt


class StreamManagerDock(QDockWidget):
    """
    Dockable widget for managing signal streams in RTMA.
    """

    add_sg_stream = pyqtSignal()
    add_ni_stream = pyqtSignal()
    remove_stream = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Stream Manager", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.stream_list = QListWidget()
        self._init_ui()

    def _init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Active Streams:"))
        layout.addWidget(self.stream_list)

        btn_add_sg = QPushButton("➕ Add Signal Generator")
        btn_add_ni = QPushButton("➕ Add NI Stream")
        btn_remove = QPushButton("❌ Remove Selected")

        btn_add_sg.clicked.connect(self.add_sg_stream.emit)
        btn_add_ni.clicked.connect(self.add_ni_stream.emit)
        btn_remove.clicked.connect(self._remove_selected_stream)

        layout.addWidget(btn_add_sg)
        layout.addWidget(btn_add_ni)
        layout.addWidget(btn_remove)

        widget.setLayout(layout)
        self.setWidget(widget)

    def add_stream_name(self, name: str):
        self.stream_list.addItem(name)

    def _remove_selected_stream(self):
        selected = self.stream_list.currentItem()
        if selected:
            name = selected.text()
            self.stream_list.takeItem(self.stream_list.row(selected))
            self.remove_stream.emit(name)
