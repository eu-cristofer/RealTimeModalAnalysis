"""
sg_stream_widget.py

A stream viewer for Signal Generator data via shared memory.
"""

from PyQt6.QtWidgets import QWidget
from ui.components.stream_widget_base import IStreamSource, BaseStreamWidget
from core.shared_buffer import attach_shared_buffer
import numpy as np


class SGStreamSource(IStreamSource):
    """
    Reads from shared memory buffer created by SG app.
    """

    def __init__(self, channel_index: int = 0):
        self.shm, self.buffer = attach_shared_buffer()
        self.channel_index = channel_index
        self.read_index = 0

    def read_chunk(self, length: int) -> np.ndarray:
        """
        Return a chunk of samples from all channels.
        """
        output = np.zeros((4, length), dtype=np.float32)
        for ch in range(4):
            buf = self.buffer[ch]
            end = self.read_index + length
            if end <= buf.shape[0]:
                output[ch] = buf[self.read_index:end]
            else:
                split = buf.shape[0] - self.read_index
                output[ch, :split] = buf[self.read_index:]
                output[ch, split:] = buf[:length - split]
        
        self.read_index = (self.read_index + length) % self.buffer.shape[1]
        return output

    
    def get_channel_count(self) -> int:
        return 4  # SG always streams 4 channels

    def get_channel_labels(self) -> list[str]:
        return ["Ref", "X", "Y", "Z"]



class SGStreamWidget(BaseStreamWidget):
    """
    A stream widget for displaying SG shared memory data.
    """

    def __init__(self, channel_index: int = 0, parent: QWidget = None):
        source = SGStreamSource(channel_index)
        super().__init__(source, title=f"SG Channel {channel_index+1}", parent=parent)
