"""
Wake / standby Vive Lighthouse v2 base stations directly over BLE.

Usage:
    python -m pipeline.wake_basestations          # wake
    python -m pipeline.wake_basestations off      # standby
    python -m pipeline.wake_basestations --timeout 12 on
"""

import argparse
import asyncio
import sys

from bleak import BleakClient, BleakScanner

POWER_CHAR_UUID = "00001525-1212-efde-1523-785feabcd124"
WAKE = bytearray([0x01])
STANDBY = bytearray([0x00])


async def find_basestations(timeout: float = 8.0):
    devices = await BleakScanner.discover(timeout=timeout)
    return [d for d in devices if d.name and d.name.startswith("LHB-")]


async def set_power(device, value: bytearray):
    async with BleakClient(device) as client:
        await client.write_gatt_char(POWER_CHAR_UUID, value, response=True)


async def wake_all(timeout: float = 8.0) -> list[str]:
    """Wake all visible base stations. Returns list of names successfully woken."""
    stations = await find_basestations(timeout)
    succeeded = []
    for s in stations:
        try:
            await set_power(s, WAKE)
            succeeded.append(s.name)
        except Exception:
            pass
    return succeeded


async def standby_all(timeout: float = 8.0) -> list[str]:
    stations = await find_basestations(timeout)
    succeeded = []
    for s in stations:
        try:
            await set_power(s, STANDBY)
            succeeded.append(s.name)
        except Exception:
            pass
    return succeeded


async def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["on", "off"], nargs="?", default="on")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    print(f"Scanning ({args.timeout:.0f}s)...", flush=True)
    if args.action == "on":
        woken = await wake_all(args.timeout)
    else:
        woken = await standby_all(args.timeout)

    if not woken:
        print("No base stations responded.", flush=True)
        sys.exit(1)
    for name in woken:
        print(f"  {name} -> {args.action.upper()}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
