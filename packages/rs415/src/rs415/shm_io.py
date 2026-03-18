from __future__ import annotations

"""Shared memory IO helpers for RS415 frame publishing and reading."""

from dataclasses import dataclass
import json
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
import os
import struct
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .rs415 import Camera


META_SIZE = 16 * 1024
META_LENGTH_STRUCT = struct.Struct("<I")
RGB_DTYPE = np.dtype(np.uint8)
DEPTH_DTYPE = np.dtype(np.uint16)


@dataclass(frozen=True)
class SharedMemoryNames:
    meta: str
    rgb: str
    depth: str


@dataclass(frozen=True)
class FrameBundle:
    meta: dict[str, Any]
    rgb: np.ndarray
    depth: np.ndarray


def list_available_serials() -> list[str]:
    try:
        entries = os.listdir("/dev/shm")
    except FileNotFoundError:
        return []

    serials: set[str] = set()
    for entry in entries:
        if not entry.startswith("rs415_") or not entry.endswith("_meta"):
            continue
        serial = entry[len("rs415_") : -len("_meta")]
        if serial:
            serials.add(serial)
    return sorted(serials)


def make_shm_names(serial: str) -> SharedMemoryNames:
    return SharedMemoryNames(
        meta=f"rs415_{serial}_meta",
        rgb=f"rs415_{serial}_rgb",
        depth=f"rs415_{serial}_depth",
    )


def build_camera_meta(camera: Camera) -> dict[str, Any]:
    if camera.connected_serial is None or camera.connected_name is None:
        raise RuntimeError("Camera is not connected. Call connect() first.")

    intr = camera.get_color_intrinsics()
    intrinsics_payload = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "coeffs": list(intr.coeffs[:5]),
    }

    return {
        "version": 1,
        "connected": True,
        "serial": camera.connected_serial,
        "camera_name": camera.connected_name,
        "depth_scale": camera.get_depth_scale(),
        "rgb_shape": [480, 640, 3],
        "rgb_dtype": "uint8",
        "depth_shape": [480, 640],
        "depth_dtype": "uint16",
        "intrinsics": intrinsics_payload,
        "frame_id": 0,
        "timestamp_ns": 0,
    }


def depth_to_mm_uint16(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    if abs(depth_scale - 0.001) < 1e-12 and depth_raw.dtype == DEPTH_DTYPE:
        return depth_raw
    depth_mm_f = depth_raw.astype(np.float32, copy=False) * depth_scale * 1000.0
    return np.clip(np.rint(depth_mm_f), 0, 65535).astype(DEPTH_DTYPE)


class RS415SharedMemoryWriter:
    def __init__(
        self,
        *,
        serial: str,
        rgb_shape: tuple[int, int, int],
        depth_shape: tuple[int, int],
        meta: dict[str, Any],
    ) -> None:
        self.names = make_shm_names(serial)
        self._frame_id = int(meta.get("frame_id", 0))
        self._meta_shm = _create_or_replace_shared_memory(
            name=self.names.meta,
            size=META_SIZE,
        )
        self._rgb_shm = _create_or_replace_shared_memory(
            name=self.names.rgb,
            size=int(np.prod(rgb_shape)) * RGB_DTYPE.itemsize,
        )
        self._depth_shm = _create_or_replace_shared_memory(
            name=self.names.depth,
            size=int(np.prod(depth_shape)) * DEPTH_DTYPE.itemsize,
        )
        self._rgb = np.ndarray(rgb_shape, dtype=RGB_DTYPE, buffer=self._rgb_shm.buf)
        self._depth = np.ndarray(depth_shape, dtype=DEPTH_DTYPE, buffer=self._depth_shm.buf)
        self._meta = dict(meta)
        self._write_meta(self._meta)

    @classmethod
    def from_camera(cls, camera: Camera) -> RS415SharedMemoryWriter:
        meta = build_camera_meta(camera)
        return cls(
            serial=meta["serial"],
            rgb_shape=tuple(meta["rgb_shape"]),
            depth_shape=tuple(meta["depth_shape"]),
            meta=meta,
        )

    def write(self, rgb: np.ndarray, depth_mm: np.ndarray, *, timestamp_ns: int | None = None) -> dict[str, Any]:
        if rgb.shape != self._rgb.shape or rgb.dtype != RGB_DTYPE:
            raise ValueError(f"rgb must have shape {self._rgb.shape} and dtype uint8")
        if depth_mm.shape != self._depth.shape or depth_mm.dtype != DEPTH_DTYPE:
            raise ValueError(
                f"depth_mm must have shape {self._depth.shape} and dtype uint16"
            )

        np.copyto(self._rgb, rgb)
        np.copyto(self._depth, depth_mm)
        self._frame_id += 1
        self._meta["frame_id"] = self._frame_id
        self._meta["timestamp_ns"] = time.time_ns() if timestamp_ns is None else timestamp_ns
        self._meta["connected"] = True
        self._write_meta(self._meta)
        return dict(self._meta)

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def mark_disconnected(self) -> None:
        self._meta["connected"] = False
        self._meta["timestamp_ns"] = time.time_ns()
        self._write_meta(self._meta)

    def close(self) -> None:
        self._meta_shm.close()
        self._rgb_shm.close()
        self._depth_shm.close()

    def unlink(self) -> None:
        for shm in (self._meta_shm, self._rgb_shm, self._depth_shm):
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def _write_meta(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        total_size = META_LENGTH_STRUCT.size + len(encoded)
        if total_size > META_SIZE:
            raise ValueError("metadata payload exceeds shared memory segment size")
        self._meta_shm.buf[: META_LENGTH_STRUCT.size] = META_LENGTH_STRUCT.pack(0)
        self._meta_shm.buf[META_LENGTH_STRUCT.size : total_size] = encoded
        self._meta_shm.buf[: META_LENGTH_STRUCT.size] = META_LENGTH_STRUCT.pack(len(encoded))


class RS415SharedMemoryReader:
    def __init__(self, serial: str) -> None:
        self.names = make_shm_names(serial)
        self._meta_shm = shared_memory.SharedMemory(name=self.names.meta)
        initial_meta = self.read_meta()
        rgb_shape = tuple(initial_meta["rgb_shape"])
        depth_shape = tuple(initial_meta["depth_shape"])
        self._rgb_shm = shared_memory.SharedMemory(name=self.names.rgb)
        self._depth_shm = shared_memory.SharedMemory(name=self.names.depth)
        _unregister_shared_memory(self._meta_shm)
        _unregister_shared_memory(self._rgb_shm)
        _unregister_shared_memory(self._depth_shm)
        self._rgb = np.ndarray(rgb_shape, dtype=RGB_DTYPE, buffer=self._rgb_shm.buf)
        self._depth = np.ndarray(depth_shape, dtype=DEPTH_DTYPE, buffer=self._depth_shm.buf)

    def read_meta(self) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(10):
            raw_length_before = bytes(self._meta_shm.buf[: META_LENGTH_STRUCT.size])
            payload_length = META_LENGTH_STRUCT.unpack(raw_length_before)[0]
            if payload_length == 0:
                last_error = RuntimeError("shared memory metadata is empty")
                time.sleep(0.001)
                continue
            if payload_length > META_SIZE - META_LENGTH_STRUCT.size:
                last_error = RuntimeError("shared memory metadata length is invalid")
                time.sleep(0.001)
                continue
            start = META_LENGTH_STRUCT.size
            stop = start + payload_length
            try:
                payload = bytes(self._meta_shm.buf[start:stop]).decode("utf-8")
                raw_length_after = bytes(self._meta_shm.buf[: META_LENGTH_STRUCT.size])
                if raw_length_before != raw_length_after:
                    last_error = RuntimeError("shared memory metadata changed during read")
                    time.sleep(0.001)
                    continue
                return json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(0.001)
        if last_error is None:
            raise RuntimeError("failed to read shared memory metadata")
        raise RuntimeError("failed to read a consistent shared memory metadata payload") from last_error

    def read(self, *, copy: bool = True) -> FrameBundle:
        meta = self.read_meta()
        rgb = self._rgb.copy() if copy else self._rgb
        depth = self._depth.copy() if copy else self._depth
        return FrameBundle(meta=meta, rgb=rgb, depth=depth)

    def wait_for_frame(
        self,
        *,
        last_frame_id: int | None = None,
        timeout_sec: float | None = None,
        poll_interval_sec: float = 0.01,
        copy: bool = True,
    ) -> FrameBundle:
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while True:
            bundle = self.read(copy=copy)
            frame_id = int(bundle.meta["frame_id"])
            if last_frame_id is None or frame_id > last_frame_id:
                return bundle
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for a new frame")
            time.sleep(poll_interval_sec)

    def close(self) -> None:
        self._meta_shm.close()
        self._rgb_shm.close()
        self._depth_shm.close()


def run_shm_service(camera: Camera) -> int:
    if not camera.is_connected:
        raise RuntimeError("Camera must be connected before starting service loop.")

    writer = RS415SharedMemoryWriter.from_camera(camera)
    meta = writer.meta
    print(f"Shared memory publisher ready for serial={meta['serial']}")
    print(f"- meta: {writer.names.meta}")
    print(f"- rgb: {writer.names.rgb}")
    print(f"- depth: {writer.names.depth}")
    print("Publisher loop running. Press Ctrl+C to stop.")

    warned_frame_timeout = False
    try:
        while True:
            try:
                frames = camera.get_aligned_frames(types=("rgb", "depth"), timeout_ms=5000)
            except RuntimeError as exc:
                if "Frame didn't arrive within" not in str(exc):
                    raise
                if not warned_frame_timeout:
                    print("Frame delivery is delayed; keeping the publisher loop alive.")
                    warned_frame_timeout = True
                time.sleep(0.05)
                continue

            warned_frame_timeout = False
            depth_mm = depth_to_mm_uint16(frames["depth"], meta["depth_scale"])
            writer.write(frames["rgb"], depth_mm)
    except KeyboardInterrupt:
        print("Stopping connect loop.")
        return 0
    finally:
        writer.mark_disconnected()
        writer.close()
        writer.unlink()


def _unregister_shared_memory(shm: shared_memory.SharedMemory) -> None:
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


def _create_or_replace_shared_memory(*, name: str, size: int) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        stale = shared_memory.SharedMemory(name=name, create=False)
        stale.close()
        stale.unlink()
        return shared_memory.SharedMemory(name=name, create=True, size=size)
