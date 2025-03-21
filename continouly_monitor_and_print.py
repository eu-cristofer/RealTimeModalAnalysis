import nidaqmx
from nidaqmx.constants import (
    AcquisitionType, VoltageUnits, TerminalConfiguration
)

# Replace with your actual device name
DEVICE_NAME = "cDAQ9181-1FC6921Mod1"  # Or use the IP address
CHANNEL_NAME = f"{DEVICE_NAME}/ai0"  # Example for an analog input channel

try:
    with nidaqmx.Task() as task:
        # Configure the accelerometer input with IEPE enabled
        task.ai_channels.add_ai_accel_chan( # Enables IEPE mode (constant current excitation).
            physical_channel=CHANNEL_NAME,
            sensitivity=100.0,  # Sensitivity in mV/g (check sensor calibration sheet)
            terminal_config=TerminalConfiguration.DEFAULT,
            min_val=-50.0,  # Set based on expected acceleration range
            max_val=50.0,
            units=nidaqmx.constants.AccelUnits.G,
            current_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL,
            current_excit_val=0.002  # 2 mA excitation (PCB 320C33 spec)
        )
        # Set sample rate and continuous acquisition mode
        task.timing.cfg_samp_clk_timing(
            rate=1000,  # Sampling rate in Hz (adjust based on analysis needs)
            sample_mode=AcquisitionType.CONTINUOUS
        )
        print("Reading accelerometer data... Press Ctrl+C to stop.")
    
        while True:
            accel_value = task.read(number_of_samples_per_channel=1)
            print(f"Acceleration: {accel_value[0]:.6f} g")
            
except nidaqmx.errors.DaqError as e:
    print(f"DAQ Error: {e}")
    
except KeyboardInterrupt:
    print("Acquisition stopped by user.")