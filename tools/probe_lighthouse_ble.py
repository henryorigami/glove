"""
One-off probe: enumerate GATT services + characteristics on each Vive
Lighthouse base station so we can confirm the channel characteristic UUID
before writing a setter.
"""

import asyncio
from bleak import BleakClient, BleakScanner


async def main():
    print("Scanning for LHB-* (10s)...")
    devices = await BleakScanner.discover(timeout=10.0)
    lhs = [d for d in devices if d.name and d.name.startswith("LHB-")]
    print(f"Found {len(lhs)}: {[d.name for d in lhs]}")
    for d in lhs:
        print(f"\n=== {d.name} ({d.address}) ===")
        try:
            async with BleakClient(d) as client:
                for svc in client.services:
                    print(f"  service {svc.uuid}")
                    for ch in svc.characteristics:
                        props = ",".join(ch.properties)
                        print(f"    char {ch.uuid}  [{props}]")
                        if "read" in ch.properties:
                            try:
                                val = await client.read_gatt_char(ch.uuid)
                                hexv = val.hex()
                                ascii_v = val.decode("ascii", errors="replace")
                                print(f"      = 0x{hexv}  ({ascii_v!r})")
                            except Exception as e:
                                print(f"      read failed: {e}")
        except Exception as e:
            print(f"  connect failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
