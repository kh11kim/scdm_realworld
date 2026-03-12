# rs415

Minimal RealSense D415 toolkit with:
- a `Camera` wrapper for device connection, intrinsics, and frame capture
- ArUco GridBoard generation and live calibration commands
- a Tyro CLI (`info`, `connect`, `generate-board`, `calibrate`)
- a shared-memory publisher mode for local process integration

## Quick Start

```bash
uv sync
uv run rs415 list
uv run rs415 info
uv run rs415 connect --mode window
uv run rs415 connect --mode server
uv run rs415 generate-board --output aruco_gridboard.png
uv run rs415 calibrate --output calibration.json
```

## Calibration

1. Generate and print a board:

```bash
uv run rs415 generate-board \
  --output aruco_gridboard.png \
  --markers-x 5 \
  --markers-y 7 \
  --marker-length-m 0.04 \
  --marker-separation-m 0.01
```

2. Run live calibration:

```bash
uv run rs415 calibrate --output calibration.json
```

Controls in the calibration window:
- `SPACE`: capture current board detection as one sample
- `ENTER`: solve calibration early once at least 3 samples are captured
- `q` or `ESC`: cancel

## Shared Memory Mode

```bash
uv run rs415 connect --mode server
```

When server mode starts, it creates these segments:
- `rs415_{serial}_meta`
- `rs415_{serial}_rgb`
- `rs415_{serial}_depth`

`meta` stores UTF-8 JSON prefixed by a 4-byte little-endian payload length.
`rgb` stores `uint8` RGB data with shape `(480, 640, 3)`.
`depth` stores `uint16` depth data in millimeters with shape `(480, 640)`, aligned to the RGB camera geometry.
`intrinsics` in `meta` are the RGB camera intrinsics used for the aligned RGB/depth pair.

Reader example:

```python
from rs415.shm_io import RS415SharedMemoryReader

reader = RS415SharedMemoryReader(serial="123456789")
try:
    bundle = reader.wait_for_frame(timeout_sec=5.0)
    print(bundle.meta["intrinsics"])
    rgb = bundle.rgb
    depth_mm = bundle.depth
finally:
    reader.close()
```
