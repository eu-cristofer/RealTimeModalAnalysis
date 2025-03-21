# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 21:05:34 2025

@author: EMKA
"""
from nidaqmx.constants import (
    AcquisitionType, VoltageUnits, TerminalConfiguration
)
from nidaqmx.stream_readers import AnalogMultiChannelReader
from config import CHANNELS
channels = CHANNELS

import nidaqmx
import numpy as np


samples_per_read = 1000
sample_rate = 10 # Sampling rate in Hz (adjust based on analysis needs)

with nidaqmx.Task() as task:
    for ch in channels:
        # Configure the accelerometer input with IEPE enabled
        task.ai_channels.add_ai_accel_chan( # Enables IEPE mode (constant current excitation).
            physical_channel=ch,
            sensitivity=100.0,  # Sensitivity in mV/g (check sensor calibration sheet)
            terminal_config=TerminalConfiguration.DEFAULT,
            min_val=-50.0,  # Set based on expected acceleration range
            max_val=50.0,
            units=nidaqmx.constants.AccelUnits.G,
            current_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL,
            current_excit_val=0.002  # 2 mA excitation (PCB 320C33 spec)
        )
    task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples_per_read)
    
    data = np.zeros((4, samples_per_read))
    reader = AnalogMultiChannelReader(task.in_stream)
    task.start()
    reader.read_many_sample(data, number_of_samples_per_channel=samples_per_read, timeout=10.0)
    print(data)

