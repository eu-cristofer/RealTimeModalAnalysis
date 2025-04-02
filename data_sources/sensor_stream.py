import numpy as np
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, AccelUnits, ExcitationSource
from .base_stream import BaseStream
from config import CHANNELS

class SensorStream(BaseStream):
    def __init__(self, sample_rate=1000, chunk_size=256):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.task = None
        self.reader = None
        self.buffer = np.zeros((4, self.chunk_size))

    def start(self):
        self.task = nidaqmx.Task()
        for ch in CHANNELS:
            self.task.ai_channels.add_ai_accel_chan(
                physical_channel=ch,
                sensitivity=100.0,
                terminal_config=TerminalConfiguration.DEFAULT,
                min_val=-50.0,
                max_val=50.0,
                units=AccelUnits.G,
                current_excit_source=ExcitationSource.INTERNAL,
                current_excit_val=0.002
            )

        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.chunk_size * 10
        )

        self.reader = AnalogMultiChannelReader(self.task.in_stream)
        self.task.start()

    def stop(self):
        if self.task:
            self.task.stop()
            self.task.close()

    def read_chunk(self):
        try:
            self.reader.read_many_sample(self.buffer, number_of_samples_per_channel=self.chunk_size, timeout=1.0)
            return self.buffer.copy()
        except Exception as e:
            print(f"[SensorStream] Read error: {e}")
            return np.zeros((4, self.chunk_size))
