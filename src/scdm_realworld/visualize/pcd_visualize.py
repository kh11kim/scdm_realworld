from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import tyro
import viser

from rs415.shm_io import RS415SharedMemoryReader, list_available_serials
from scdm_realworld.utils.geometry import (
    compute_camera_pose,
    compute_frustum_params,
    project_depth_to_world,
)

@dataclass
class Args:
    serial: str | None = None
    calib: Path | None = None
    dt: float = 0.1
    host: str = "0.0.0.0"
    port: int = 8080
    frustum_scale: float = 0.12
    pcd: int | None = None


def _resolve_serial(serial: str | None) -> str:
    if serial is not None:
        return serial
    available = list_available_serials()
    if not available:
        raise RuntimeError("No RS415 shared memory publishers found.")
    if len(available) > 1:
        raise RuntimeError(
            f"Multiple RS415 publishers found: {', '.join(available)}. Pass --serial."
        )
    return available[0]


def _resolve_calib_path(serial: str, calib_path: Path | None) -> Path:
    path = calib_path if calib_path is not None else Path("calib") / f"{serial}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Calibration YAML not found: {path}")
    return path


def _load_calibration(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, object] = {}
    section: str | None = None
    subsection: str | None = None

    for raw_line in lines:
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1]
            subsection = None
            if section not in result:
                result[section] = {}
            continue

        if indent == 0 and ":" in stripped:
            key, value = stripped.split(":", 1)
            result[key.strip()] = _parse_scalar(value.strip())
            section = None
            subsection = None
            continue

        if indent == 2 and stripped.endswith(":"):
            subsection = stripped[:-1]
            parent = _require_dict(result, section)
            parent[subsection] = []
            continue

        if indent == 2 and ":" in stripped:
            key, value = stripped.split(":", 1)
            parent = _require_dict(result, section)
            parent[key.strip()] = _parse_scalar(value.strip())
            subsection = None
            continue

        if indent == 4 and stripped.startswith("-"):
            parent = _require_list(result, section, subsection)
            parent.append(_parse_scalar(stripped[1:].strip()))
            continue

        raise ValueError(f"Unsupported calibration YAML structure in {path}: {raw_line}")

    return result


def _require_dict(root: dict[str, object], key: str | None) -> dict[str, object]:
    if key is None or not isinstance(root.get(key), dict):
        raise ValueError(f"Expected mapping section: {key}")
    return root[key]  # type: ignore[return-value]


def _require_list(
    root: dict[str, object], section: str | None, subsection: str | None
) -> list[float | str]:
    parent = _require_dict(root, section)
    if subsection is None or not isinstance(parent.get(subsection), list):
        raise ValueError(f"Expected list section: {section}.{subsection}")
    return parent[subsection]  # type: ignore[return-value]


def _parse_scalar(value: str) -> float | str:
    if value == "":
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def _depth_to_rgb(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm > 0
    if not np.any(valid):
        return np.zeros(depth_mm.shape + (3,), dtype=np.uint8)

    lo = float(np.percentile(depth_mm[valid], 5))
    hi = float(np.percentile(depth_mm[valid], 95))
    hi = max(hi, lo + 1.0)
    normalized = np.clip((depth_mm.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    ramp = (normalized * 255.0).astype(np.uint8)
    ramp_i16 = ramp.astype(np.int16)
    green = np.clip(255 - 2 * np.abs(ramp_i16 - 128), 0, 255).astype(np.uint8)
    depth_rgb = np.stack((ramp, green, 255 - ramp), axis=-1)
    depth_rgb[~valid] = 0
    return depth_rgb


def main() -> int:
    args = tyro.cli(Args)
    serial = _resolve_serial(args.serial)
    calib_path = _resolve_calib_path(serial, args.calib)
    calib = _load_calibration(calib_path)
    fov, aspect, position, wxyz = compute_frustum_params(calib)
    rotation_wc, translation_wc = compute_camera_pose(calib)
    intrinsic = calib["intrinsic"]
    if not isinstance(intrinsic, dict):
        raise ValueError("Calibration YAML is missing intrinsic section.")
    if args.pcd is not None and args.pcd <= 0:
        raise ValueError("--pcd must be a positive integer when provided.")

    server = viser.ViserServer(host=args.host, port=args.port)
    reader = RS415SharedMemoryReader(serial=serial)

    with server.gui.add_folder("Image"):
        mode_handle = server.gui.add_dropdown(
            "Mode",
            ("rgb", "depth"),
            initial_value="rgb",
        )

    server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.01)
    server.scene.add_frame(
        f"/camera/{serial}",
        axes_length=0.15,
        axes_radius=0.008,
        position=position,
        wxyz=wxyz,
    )
    server.scene.add_label(
        f"/camera/{serial}/label",
        text=f"RS415 {serial}",
        position=(0.0, 0.0, 0.05),
    )

    initial_bundle = reader.read(copy=True)
    initial_image = initial_bundle.rgb
    frustum = server.scene.add_camera_frustum(
        f"/camera/{serial}/frustum",
        fov=fov,
        aspect=aspect,
        scale=args.frustum_scale,
        color=(40, 160, 255),
        image=initial_image,
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    point_cloud = None
    if args.pcd is not None:
        initial_points, initial_colors = project_depth_to_world(
            initial_bundle.depth,
            initial_bundle.rgb,
            intrinsic,
            rotation_wc,
            translation_wc,
            _pcd_step_from_count(initial_bundle.depth, args.pcd),
        )
        point_cloud = server.scene.add_point_cloud(
            f"/pcd/{serial}",
            points=initial_points,
            colors=initial_colors,
            point_size=0.003,
            point_shape="circle",
        )

    print(f"Attaching to RS415 shared memory for serial={serial}")
    print(f"Calibration: {calib_path}")
    print(f"Refresh period: {args.dt:.3f}s")

    last_frame_id: int | None = None
    current_mode = str(mode_handle.value)

    @mode_handle.on_update
    def _on_mode_update(_) -> None:
        nonlocal current_mode
        current_mode = str(mode_handle.value)

    try:
        while True:
            loop_start = time.monotonic()
            bundle = reader.read(copy=True)
            frame_id = int(bundle.meta["frame_id"])
            if last_frame_id != frame_id:
                frustum.image = (
                    bundle.rgb if current_mode == "rgb" else _depth_to_rgb(bundle.depth)
                )
                if point_cloud is not None:
                    points, colors = project_depth_to_world(
                        bundle.depth,
                        bundle.rgb,
                        intrinsic,
                        rotation_wc,
                        translation_wc,
                        _pcd_step_from_count(bundle.depth, args.pcd),
                    )
                    point_cloud.remove()
                    point_cloud = server.scene.add_point_cloud(
                        f"/pcd/{serial}",
                        points=points,
                        colors=colors,
                        point_size=0.003,
                        point_shape="circle",
                    )
                last_frame_id = frame_id

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, args.dt - elapsed))
    except KeyboardInterrupt:
        return 0
    finally:
        reader.close()
        try:
            server.stop()
        except RuntimeError:
            pass


def _pcd_step_from_count(depth_mm: np.ndarray, count: int | None) -> int:
    if count is None:
        return 1
    valid_count = int(np.count_nonzero(depth_mm > 0))
    if valid_count <= count:
        return 1
    return max(1, valid_count // count)


if __name__ == "__main__":
    raise SystemExit(main())
