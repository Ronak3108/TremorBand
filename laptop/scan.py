import asyncio
from bleak import BleakScanner

async def main():
    device_found = False

    name = "TremorBand"
    devices = await BleakScanner.discover(timeout=5.0, return_adv=True)

    devices_info = []
    for address, (device, adv_data) in devices.items():
        devices_info.append((adv_data.local_name or "Unknown", address, adv_data.rssi))

    devices_info = sorted(devices_info, key=lambda x: x[2], reverse=True)

    # Print the Header Row
    print(f"{'Device Name':<25} {'MAC Address':<20} {'RSSI':>8}")
    print("-" * 55)  # Separator line

    # Print the Data Rows
    for name, address, rssi in devices_info:
    # Using string representation for RSSI to add "dBm" text seamlessly
        rssi_str = f"{rssi} dBm"
        print(f"{name:<25} {address:<20} {rssi_str:>8}")

        if(name == "TremorBand"):
            device_found = True

    if(device_found):
        print("TremorBand Detected...")
    else:
        print("TremorBand Not Found...")


asyncio.run(main())