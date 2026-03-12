from __future__ import annotations
"""Core camera wrapper for RealSense D415 device access and frame capture."""

from dataclasses import dataclass
import time
from typing import Any, Sequence

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    serial: str


@dataclass(frozen=True)
class IntrinsicInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str
    coeffs: tuple[float, ...]


class Camera:
    def __init__(self, window: bool = False, window_name: str = "RS415") -> None:
        """Initialize camera context and runtime state."""
        self._context = rs.context()
        self._pipeline: rs.pipeline | None = None
        self._profile: rs.pipeline_profile | None = None
        self._align_to_color = rs.align(rs.stream.color)
        self._connected_serial: str | None = None
        self._connected_name: str | None = None
        self._window = window
        self._window_name = window_name
        self._frame_timeout_ms = 5000

    def list_devices(self) -> list[DeviceInfo]:
        """Return all currently connected RealSense devices."""
        devices = self._context.query_devices()
        result: list[DeviceInfo] = []
        for device in devices:
            result.append(
                DeviceInfo(
                    name=device.get_info(rs.camera_info.name),
                    serial=device.get_info(rs.camera_info.serial_number),
                )
            )
        return result

    def connect(self, serial: str | None = None) -> DeviceInfo:
        """Connect the pipeline to a target serial or the first available device."""
        devices = self.list_devices()
        if not devices:
            raise RuntimeError("No RealSense devices detected.")

        if serial is None:
            target = devices[0]
        else:
            target = next((device for device in devices if device.serial == serial), None)
            if target is None:
                raise RuntimeError(f"RealSense device not found: serial={serial}")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(target.serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)

        self._pipeline = pipeline
        self._profile = profile
        self._connected_serial = target.serial
        self._connected_name = target.name
        return target

    def disconnect(self) -> None:
        """Stop the active pipeline and clear connection state."""
        if self._pipeline is None:
            return
        self._pipeline.stop()
        self._pipeline = None
        self._profile = None
        self._connected_serial = None
        self._connected_name = None

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    @property
    def connected_serial(self) -> str | None:
        return self._connected_serial

    @property
    def connected_name(self) -> str | None:
        return self._connected_name

    def get_intrinsics(self) -> dict[str, IntrinsicInfo]:
        """Return per-stream intrinsic values keyed by lowercase stream name."""
        if self._profile is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")

        intrinsics_by_stream: dict[str, IntrinsicInfo] = {}
        for stream in self._profile.get_streams():
            vsp: Any = stream.as_video_stream_profile()
            intr = vsp.get_intrinsics()
            stream_key = stream.stream_name().lower()
            intrinsics_by_stream[stream_key] = IntrinsicInfo(
                width=intr.width,
                height=intr.height,
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.ppx,
                cy=intr.ppy,
                model=str(intr.model),
                coeffs=tuple(float(c) for c in intr.coeffs),
            )

        return intrinsics_by_stream

    def get_color_intrinsics(self) -> IntrinsicInfo:
        """Return the color stream intrinsics for aligned RGB/depth consumers."""
        if self._profile is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")

        color_stream = next(
            (
                stream
                for stream in self._profile.get_streams()
                if stream.stream_type() == rs.stream.color
            ),
            None,
        )
        if color_stream is None:
            raise RuntimeError("Color stream is not available.")

        vsp: Any = color_stream.as_video_stream_profile()
        intr = vsp.get_intrinsics()
        return IntrinsicInfo(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            model=str(intr.model),
            coeffs=tuple(float(c) for c in intr.coeffs),
        )

    def get_frames(
        self, types: Sequence[str], timeout_ms: int = 5000
    ) -> dict[str, np.ndarray]:
        """Capture requested frame types (rgb/depth) in a single wait cycle."""
        if self._pipeline is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")
        if not types:
            raise ValueError("types must contain at least one frame type.")
        requested = set(types)
        unknown = requested - {"rgb", "depth"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown frame types: {names}")

        frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        result: dict[str, np.ndarray] = {}

        if "rgb" in requested:
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Color frame is not available.")
            color = np.asanyarray(color_frame.get_data())
            color_format = color_frame.profile.format()
            if color_format == rs.format.bgr8:
                result["rgb"] = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            else:
                result["rgb"] = color

        if "depth" in requested:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError("Depth frame is not available.")
            depth_raw = np.asanyarray(depth_frame.get_data())
            if depth_raw.dtype != np.uint16:
                depth_raw = depth_raw.astype(np.uint16, copy=False)
            result["depth"] = depth_raw

        return result

    def get_aligned_frames(
        self, types: Sequence[str], timeout_ms: int = 5000
    ) -> dict[str, np.ndarray]:
        """Capture RGB/depth after aligning depth into the color camera geometry."""
        if self._pipeline is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")
        if not types:
            raise ValueError("types must contain at least one frame type.")
        requested = set(types)
        unknown = requested - {"rgb", "depth"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown frame types: {names}")

        frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned_frames = self._align_to_color.process(frames)
        result: dict[str, np.ndarray] = {}

        if "rgb" in requested:
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Aligned color frame is not available.")
            color = np.asanyarray(color_frame.get_data())
            color_format = color_frame.profile.format()
            if color_format == rs.format.bgr8:
                result["rgb"] = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            else:
                result["rgb"] = color

        if "depth" in requested:
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError("Aligned depth frame is not available.")
            depth_raw = np.asanyarray(depth_frame.get_data())
            if depth_raw.dtype != np.uint16:
                depth_raw = depth_raw.astype(np.uint16, copy=False)
            result["depth"] = depth_raw

        return result

    def get_depth_scale(self) -> float:
        """Return the depth scale used to convert raw depth values to meters."""
        if self._profile is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")
        device = self._profile.get_device()
        for sensor in device.sensors:
            if sensor.is_depth_sensor():
                return float(sensor.as_depth_sensor().get_depth_scale())
        raise RuntimeError("Depth sensor is not available.")

    def spin(self) -> int:
        """Run the main camera loop with optional RGB+Depth window rendering."""
        if self._pipeline is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")

        if self._window:
            cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
            print("Showing OpenCV window. Press q or ESC to stop.")
        else:
            print("Entering idle loop. Press Ctrl+C to stop.")

        warned_frame_timeout = False
        try:
            while True:
                if self._window:
                    try:
                        frames = self.get_frames(
                            types=("rgb", "depth"),
                            timeout_ms=self._frame_timeout_ms,
                        )
                    except RuntimeError as exc:
                        if "Frame didn't arrive within" not in str(exc):
                            raise
                        if not warned_frame_timeout:
                            print(
                                "Frame delivery is delayed; keeping the window loop alive."
                            )
                            warned_frame_timeout = True
                        time.sleep(0.05)
                        continue

                    warned_frame_timeout = False
                    rgb = frames["rgb"]
                    depth = frames["depth"]
                    depth_vis = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
                    )
                    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    combined = np.hstack((rgb_bgr, depth_vis))
                    cv2.imshow(self._window_name, combined)
                    key = cv2.waitKey(1)
                    if key in (27, ord("q"), ord("Q")):
                        print("Stopping connect loop.")
                        return 0
                else:
                    time.sleep(1.0)
        finally:
            if self._window:
                cv2.destroyAllWindows()
