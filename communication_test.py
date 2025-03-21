import nidaqmx

# List available NI DAQ devices
system = nidaqmx.system.System.local()
devices = system.devices

if devices:
    print("Detected NI Devices:")
    for device in devices:
        print(f"Device Name: {device.name}")
        print(f"  Product Type: {device.product_type}")
        print(f"  Serial Number: {device.serial_num}")
        for channel in device.ai_physical_chans:
            print(f"    Available Channels: {channel.name}")
        print("-" * 40)
else:
    print("No NI devices detected. Check your connection and NI MAX configuration.")
