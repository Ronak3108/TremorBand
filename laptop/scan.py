"""List nearby BLE devices and say whether the TremorBand is among them."""
import asyncio

from bleak import BleakScanner

DEVICE_NAME = "TremorBand"
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"


async def main():
    devices = await BleakScanner.discover(timeout=5.0, return_adv=True)

    rows, found = [], False
    for address, (device, adv) in devices.items():
        name = adv.local_name or device.name or "Unknown"
        uuids = {u.lower() for u in (adv.service_uuids or [])}
        is_band = name == DEVICE_NAME or SERVICE_UUID in uuids
        found = found or is_band
        rows.append((name, address, adv.rssi if adv.rssi is not None else -999, is_band))

    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"{'Device Name':<25} {'MAC Address':<20} {'RSSI':>8}")
    print("-" * 55)
    for name, address, rssi, is_band in rows:
        print(f"{name:<25} {address:<20} {f'{rssi} dBm':>8}{'   <-- TremorBand' if is_band else ''}")

    print("TremorBand Detected..." if found else "TremorBand Not Found...")


if __name__ == "__main__":
    asyncio.run(main())
