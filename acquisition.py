"""
Módulo de aquisição de dados via NI cDAQ para análise modal experimental.
"""

import nidaqmx
import numpy as np
from collections import deque
from nidaqmx.constants import TerminalConfiguration, AcquisitionType, AccelUnits, ExcitationSource
from nidaqmx.stream_readers import AnalogMultiChannelReader
from PyQt5.QtCore import QThread, pyqtSignal
import time
import config


class DataCollector(QThread):
    data_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.buffer = deque(maxlen=config.BUFFER_SIZE)
        self.running = False

    def run(self):
        with nidaqmx.Task() as task:
            for ch in config.CHANNELS:
                task.ai_channels.add_ai_accel_chan(
                    physical_channel=ch,
                    sensitivity=config.SENSITIVITY,
                    terminal_config=TerminalConfiguration.DEFAULT,
                    min_val=-50.0,
                    max_val=50.0,
                    units=AccelUnits.G,
                    current_excit_source=ExcitationSource.INTERNAL,
                    current_excit_val=config.CURRENT_EXCITATION
                )
            task.timing.cfg_samp_clk_timing(
                rate=config.SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS
            )

            reader = AnalogMultiChannelReader(task.in_stream)
            data_chunk = np.zeros((len(config..CHANNELS), 1000))

            task.start()
            start_time = time.time()
            self.running = True
            while self.running and (time.time() - start_time < 30):
                reader.read_many_sample(data_chunk, number_of_samples_per_channel=1000)
                self.buffer.extend(data_chunk.T)
                self.data_signal.emit(np.array(self.buffer).T)
            task.stop()

    def stop(self):
        self.running = False