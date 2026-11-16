Deterministic failure tests for the RAM encoder's send/receive boundary, run
from the same **x64 MSVC developer PowerShell** fixture as the VRAM tests
(`VCPKG_ROOT` must be set):

```powershell
./tests/vram/repeat_failures.ps1
```

The script compiles `tests/ram/send_eagain.cpp`, which includes the
production `cpp/ffmpeg_ram/ffmpeg_ram_encode.cpp` with only
`avcodec_send_frame` and `avcodec_receive_packet` substituted, and drives the
public `ffmpeg_ram_encode()` entry with a real registered frame.

It covers the EAGAIN flow-control contract for screen encoding:

- a normal encode followed by a drained EAGAIN,
- warm-up (frame accepted, no packet ready) on the receive side,
- input queue full (`EAGAIN` from `avcodec_send_frame`) with pending packets
  that must be delivered by the fall-through drain instead of failing,
- `EAGAIN` on both sides, and several queued packets draining in one call,
- real `EIO` failures on either side still propagating as encode errors,
- and the empty-packet guard.

VideoToolbox pipelines on Intel macs routinely return `EAGAIN` from both
sides while the first packets are still buffering; that burst must not be
counted as an encoder failure by callers such as RustDesk, which would
otherwise disable hardware encoding for the session.
