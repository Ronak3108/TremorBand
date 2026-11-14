import argparse
import asyncio
import queue
import threading
import time

from bleak import BleakClient, BleakScanner

from .protocol import PACKET_SIZE, SAMPLES_PER_PACKET, SEQ_MOD, parse_packet

DEVICE_NAME = "TremorBand"
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

STATUS_SCANNING = f"Scanning for {DEVICE_NAME}..."
STATUS_CONNECTING = "Connecting..."
STATUS_CONNECTED = "Connected"
STATUS_RECONNECTING = "Connection lost - reconnecting..."


class BleReceiver:
    def __init__(self, name=DEVICE_NAME, scan_timeout=3.0):
        self.name = name
        self.scan_timeout = scan_timeout
        self.status = "Starting..."

        # counters (plain ints, read by other threads for display)
        self.notifications = 0            # every notification received, before cleaning up
        self.packets_delivered = 0        # packets handed to the pipeline
        self.delivered_this_connection = 0
        self.duplicates_collapsed = 0     # earlier copies of a seq that were replaced by a later one
        self.packets_lost = 0             # from gaps in seq
        self.resyncs = 0                  # seq went backwards / sideways (band restarted)
        self.bad_packets = 0              # too short to be a packet
        self.connections = 0

        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._pending = None              # newest packet seen, held until a different seq arrives
        self._pending_seq = None
        self._last_seq = None

    # ---------------------------------------------------------------- public
    def start(self):
        self._thread = threading.Thread(target=self._thread_main, daemon=True, name="ble")
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get(self, timeout=0.1):
        """Next packet, or None if none arrived within `timeout` seconds."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def packets(self):
        """Endless packet source for reader(); ends after close()."""
        while True:
            packet = self.get(0.1)
            if packet is not None:
                yield packet
            elif self._stop.is_set():
                return

    @property
    def connected(self):
        return self.status == STATUS_CONNECTED

    def stats(self):
        total = self.packets_delivered + self.packets_lost
        return {
            "connections": self.connections,
            "notifications": self.notifications,
            "delivered": self.packets_delivered,
            "duplicates_collapsed": self.duplicates_collapsed,
            "lost_packets": self.packets_lost,
            "lost_percent": 100 * self.packets_lost / total if total else 0.0,
            "resyncs": self.resyncs,
            "bad_packets": self.bad_packets,
        }

    # ------------------------------------------------- background asyncio thread
    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except Exception as e:  # nothing here should escape, but never die silently
            self.status = f"Bluetooth error: {e}"

    async def _run(self):
        while not self._stop.is_set():
            try:
                self.status = STATUS_SCANNING
                device = await self._find_device()
                if device is None:
                    continue                      # not found in this scan: scan again
                self.status = STATUS_CONNECTING
                lost_link = asyncio.Event()
                async with BleakClient(device, disconnected_callback=lambda _c: lost_link.set()) as client:
                    self._new_connection()
                    self.status = STATUS_CONNECTED
                    await client.start_notify(CHARACTERISTIC_UUID, self._on_notify)
                    while not lost_link.is_set() and not self._stop.is_set():
                        try:
                            await asyncio.wait_for(lost_link.wait(), timeout=0.2)
                        except asyncio.TimeoutError:
                            pass
                if not self._stop.is_set():
                    self.status = STATUS_RECONNECTING
            except Exception as e:
                self.status = f"Bluetooth error: {e}"
                await self._sleep(2.0)            # e.g. adapter off; try again shortly

    async def _sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._stop.is_set():
            await asyncio.sleep(0.1)

    async def _find_device(self):
        found = await BleakScanner.discover(timeout=self.scan_timeout, return_adv=True)
        best = None
        for _address, (device, adv) in found.items():
            names = {n for n in (adv.local_name, device.name) if n}
            uuids = {u.lower() for u in (adv.service_uuids or [])}
            if self.name in names or SERVICE_UUID in uuids:     # match the name OR the service UUID
                if best is None or (adv.rssi or -999) > (best[1].rssi or -999):
                    best = (device, adv)
        return best[0] if best else None

    def _new_connection(self):
        self.connections += 1
        self.delivered_this_connection = 0
        self._pending = None          # a half-received packet from the old link is useless
        self._last_seq = None         # the band keeps counting while we are away: no "loss" to report

    def _on_notify(self, _characteristic, data):
        self.notifications += 1
        data = bytes(data)
        if len(data) < PACKET_SIZE:
            self.bad_packets += 1
            return
        packet = data[:PACKET_SIZE]   # the band sends 20 bytes; the last 2 are padding
        seq = int.from_bytes(packet[:2], "little")

        if self._pending is not None:
            if seq != self._pending_seq:
                self._deliver(self._pending_seq, self._pending)   # a new seq: the held one was the final copy
            else:
                self.duplicates_collapsed += 1                    # same seq again: newer copy replaces it
        self._pending, self._pending_seq = packet, seq

    def _deliver(self, seq, packet):
        if self._last_seq is not None:
            step = (seq - self._last_seq) % SEQ_MOD
            if step == SAMPLES_PER_PACKET:
                pass
            elif step % SAMPLES_PER_PACKET == 0 and 0 < step < SEQ_MOD // 2:
                self.packets_lost += step // SAMPLES_PER_PACKET - 1
            else:
                self.resyncs += 1
        self._last_seq = seq
        self.packets_delivered += 1
        self.delivered_this_connection += 1
        self._queue.put(packet)


# ---------------------------------------------------------------- command line
def main():
    parser = argparse.ArgumentParser(description="Receive from the TremorBand and print values / link stats")
    parser.add_argument("-s", "--seconds", type=float, default=None, help="stop after this long (default: Ctrl-C)")
    parser.add_argument("--name", default=DEVICE_NAME)
    args = parser.parse_args()

    rx = BleReceiver(name=args.name)
    rx.start()
    t0 = time.monotonic()
    next_print = t0 + 1.0
    last_sample, last_delivered, last_status = None, 0, None
    try:
        while args.seconds is None or time.monotonic() - t0 < args.seconds:
            packet = rx.get(0.1)
            if packet is not None:
                last_sample = parse_packet(packet)[1][-1]
            if rx.status != last_status:
                last_status = rx.status
                print(f"[{time.monotonic() - t0:6.1f} s] {rx.status}")
            now = time.monotonic()
            if now >= next_print:
                if last_sample and rx.connected:
                    rate = rx.packets_delivered - last_delivered
                    print(f"[{now - t0:6.1f} s] accel {last_sample[0]:.3f} g | gyro {last_sample[1]:6.1f} deg/s"
                          f" | {rate:2d} packets/s | lost {rx.packets_lost}")
                last_delivered = rx.packets_delivered
                next_print += 1.0
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()

    elapsed = time.monotonic() - t0
    s = rx.stats()
    print("|-----SUMMARY-----|")
    print(f"Run: {elapsed:.1f} s | connections: {s['connections']}")
    print(f"Packets delivered: {s['delivered']} ({s['delivered'] / elapsed:.1f}/s; 25/s when nothing is lost)")
    print(f"Packets lost (seq gaps): {s['lost_packets']}  ({s['lost_percent']:.2f}% = {s['lost_packets'] * SAMPLES_PER_PACKET} samples)")
    print(f"Restarts of the band's seq: {s['resyncs']} | too-short notifications: {s['bad_packets']}")
    if s["delivered"]:
        per_packet = s["notifications"] / s["delivered"]
        print(f"Notifications received: {s['notifications']} ({per_packet:.1f} per delivered packet)")
        if per_packet > 1.5:
            print("  Note: the band is notifying more than once per packet (extra half-updated copies);")
            print("  the receiver keeps only the last, complete copy of each seq.")


if __name__ == "__main__":
    main()