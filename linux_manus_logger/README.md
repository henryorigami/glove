# Linux MANUS Integrated Logger

This is the WSL-native MANUS logger path. It uses `libManusSDK_Integrated.so`,
so it does **not** require MANUS Core on Windows.

`record_session.py` uses this logger by default. Set
`HAND_CAPTURE_MANUS_MODE=windows` to fall back to the old Windows
`manus_logger.exe` path.

## Build inside WSL

```bash
cd /mnt/c/Users/henry/Desktop/hand_capture/linux_manus_logger
make
```

## Run

```bash
./manus_integrated_logger --session-dir /mnt/c/Users/henry/Desktop/hand_capture/recordings/test_linux_manus --duration 20
```

The MANUS dongle/gloves must be visible inside WSL via `usbipd-win`. From an
elevated PowerShell:

```powershell
tools\share_manus_usb.ps1
```

If the logger starts but CSVs only contain headers, WSL has the dongle but the
gloves are not publishing frames yet. Check glove power / pairing / battery and
rerun; the dashboard shows MANUS USB separately from skeleton frame count.
