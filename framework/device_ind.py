import tvm
from tvm import rpc
import os

# Connect to the RPC tracker
tracker_host = "127.0.0.1"
tracker_port = 9190
tracker = rpc.connect_tracker(tracker_host, tracker_port)

# Request a remote session using the "android" key
device_key = "android"
remote = tracker.request(device_key)

# Get the list of available devices by running a command on the remote device
devices_output = remote.system("cat /proc/net/unix | grep 'socket' | awk '{print $9}'")
devices = devices_output.split('\n')[:-1]  # Remove the last empty element

# Replace 'TARGET_DEVICE_SERIAL' with the serial number of your Android device
target_device_serial = 'R3CT107HNVD'

# Find the target device index
device_index = None
for device_id, device_serial in enumerate(devices):
    if device_serial == target_device_serial:
        device_index = device_id
        break

if device_index is not None:
    print(f"Found target device at index {device_index}")
else:
    print("Target device not found")
