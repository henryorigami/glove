"""
Read / set the channel of a Vive Lighthouse 2.0 base station via BLE.

LH 2.0 base stations all default to channel 1 from the factory. Running two
on the same channel causes IR sweep collisions — the trackers see two
overlapping sweeps and can't disambiguate them, so OOTX never decodes and
no poses are produced. Fix: put each lighthouse on a different channel.

Channel values are a single byte 0x00 ... 0x0F (16 channels total). The
value is encoded as (channel_number - 1). So channel 1 = 0x00, channel 2 = 0x01,
... channel 16 = 0x0F. Channel settings persist across power cycles.

Usage:
    python -m pipeline.set_channel --list
    python -m pipeline.set_channel LHB-89218ED6 1       # set to channel 1
    python -m pipeline.set_channel LHB-E4C06FAE 2       # set to channel 2
    python -m pipeline.set_channel LHB-XXX --identify   # flash LED 5s
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from bleak import BleakClient, BleakScanner

VALVE_SERVICE = "00001523-1212-efde-1523-785feabcd124"
CHAR_CHANNEL = "00001524-1212-efde-1523-785feabcd124"
CHAR_POWER = "00001525-1212-efde-1523-785feabcd124"
CHAR_IDENTIFY = "00008421-1212-efde-1523-785feabcd124"


async def find_basestations(timeout: float = 8.0):
    devices = await BleakScanner.discover(timeout=timeout)
    return [d for d in devices if d.name and d.name.startswith("LHB-")]


async def get_channel(client: BleakClient) -> int:
    raw = await client.read_gatt_char(CHAR_CHANNEL)
    return raw[0] + 1  # encoded value + 1 = channel number


async def set_channel(client: BleakClient, channel: int) -> None:
    if not (1 <= channel <= 16):
        raise ValueError("channel must be 1-16")
    await client.write_gatt_char(CHAR_CHANNEL,
                                 bytes([channel - 1]),
                                 response=True)


async def identify(client: BleakClient) -> None:
    # Writing 1 to identify char flashes the LED.
    await client.write_gatt_char(CHAR_IDENTIFY, bytes([0x01]), response=True)


async def list_channels(timeout: float):
    stations = await find_basestations(timeout)
    if not stations:
        print("No base stations visible.")
        return
    for s in stations:
        try:
            async with BleakClient(s) as c:
                ch = await get_channel(c)
            print(f"  {s.name}  channel = {ch}")
        except Exception as e:
            print(f"  {s.name}  read failed: {e}")


async def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", nargs="?", help="Base station name, e.g. LHB-89218ED6")
    ap.add_argument("channel", nargs="?", type=int,
                    help="Channel 1-16 to write")
    ap.add_argument("--list", action="store_true",
                    help="Show channel of every visible LH")
    ap.add_argument("--identify", action="store_true",
                    help="Flash LED on target LH")
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()

    if args.list:
        print(f"Scanning ({args.timeout:.0f}s)...")
        await list_channels(args.timeout)
        return 0

    if not args.name:
        ap.error("name (or --list) required")

    print(f"Scanning for {args.name}...")
    stations = await find_basestations(args.timeout)
    target = next((s for s in stations if s.name == args.name), None)
    if target is None:
        print(f"Could not find {args.name}. Visible: "
              f"{[s.name for s in stations]}")
        return 1

    async with BleakClient(target) as client:
        if args.identify:
            await identify(client)
            print(f"Sent identify to {args.name} — its LED should flash.")
            return 0
        if args.channel is None:
            ap.error("provide a channel 1-16, or --identify")
        current = await get_channel(client)
        print(f"  current channel: {current}")
        if current == args.channel:
            print("  already on that channel, nothing to do.")
            return 0
        await set_channel(client, args.channel)
        # Read back to confirm
        new = await get_channel(client)
        print(f"  set to channel {new}")
        if new != args.channel:
            print(f"  WARNING: readback shows {new}, expected {args.channel}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
