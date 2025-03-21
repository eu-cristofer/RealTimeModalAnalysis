import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, AccelUnits, ExcitationSource
from nidaqmx.stream_readers import AnalogMultiChannelReader
import numpy as np
from config import CHANNELS

channels = CHANNELS
samples_per_read = 1024 * 2
sample_rate = 500  # Collect signals with bandwidth up to ~400-500 Hz

with nidaqmx.Task() as task:
    for ch in channels:
        task.ai_channels.add_ai_accel_chan(
            physical_channel=ch,
            sensitivity=100.0,  # mV/g
            terminal_config=TerminalConfiguration.DEFAULT,
            min_val=-50.0,
            max_val=50.0,
            units=AccelUnits.G,
            current_excit_source=ExcitationSource.INTERNAL,
            current_excit_val=0.002  # 2 mA excitation
        )
    # Sets up the timing for finite sampling (not continuous) with the specified rate and number of samples.
    task.timing.cfg_samp_clk_timing(rate=sample_rate,
                                    sample_mode=AcquisitionType.FINITE,
                                    samps_per_chan=samples_per_read)
    task.start()
    # Pre allocate numpy array
    data = np.zeros((4, samples_per_read))
    '''The reader will fill this array with acquired data.
    Remarks:
    ========
    task.in_stream gives access to the input data stream.
    AnalogMultiChannelReader is a special object optimized for efficiently reading from multiple channels at once.
    It works directly with the NumPy array to avoid slow data transfers.
    '''
    reader = AnalogMultiChannelReader(task.in_stream)
    reader.read_many_sample(data,
                            number_of_samples_per_channel=samples_per_read,
                            timeout=10.0) # Wait this long (in seconds) before giving up if data isn’t available.
    print(data)

    # Optional: plot quickly
    import matplotlib.pyplot as plt
    t = np.linspace(0, samples_per_read / sample_rate, samples_per_read)
    for i, axis in enumerate(["Ref", "X", "Y", "Z"]):
        plt.figure()
        plt.plot(t, data[i])
        plt.title(f"{axis} Axis Time Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [g]")
        plt.grid()
        plt.show()
