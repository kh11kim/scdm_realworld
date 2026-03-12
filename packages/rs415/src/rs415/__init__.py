"""RealSense D415 helper package."""

__all__ = [
    "DeviceInfo",
    "IntrinsicInfo",
    "Camera",
    "BoardSpec",
    "CalibrationResult",
    "FrameBundle",
    "SharedMemoryNames",
    "RS415SharedMemoryWriter",
    "RS415SharedMemoryReader",
    "build_camera_meta",
    "depth_to_mm_uint16",
    "list_available_serials",
    "make_shm_names",
]


def __getattr__(name: str):
    if name in {"Camera", "DeviceInfo", "IntrinsicInfo"}:
        from .rs415 import Camera, DeviceInfo, IntrinsicInfo

        return {
            "Camera": Camera,
            "DeviceInfo": DeviceInfo,
            "IntrinsicInfo": IntrinsicInfo,
        }[name]
    if name in {"BoardSpec", "CalibrationResult"}:
        from .calibration import BoardSpec, CalibrationResult

        return {
            "BoardSpec": BoardSpec,
            "CalibrationResult": CalibrationResult,
        }[name]
    if name in {
        "FrameBundle",
        "SharedMemoryNames",
        "RS415SharedMemoryWriter",
        "RS415SharedMemoryReader",
        "build_camera_meta",
        "depth_to_mm_uint16",
        "list_available_serials",
        "make_shm_names",
    }:
        from .shm_io import (
            FrameBundle,
            RS415SharedMemoryReader,
            RS415SharedMemoryWriter,
            SharedMemoryNames,
            build_camera_meta,
            depth_to_mm_uint16,
            list_available_serials,
            make_shm_names,
        )

        return {
            "FrameBundle": FrameBundle,
            "SharedMemoryNames": SharedMemoryNames,
            "RS415SharedMemoryWriter": RS415SharedMemoryWriter,
            "RS415SharedMemoryReader": RS415SharedMemoryReader,
            "build_camera_meta": build_camera_meta,
            "depth_to_mm_uint16": depth_to_mm_uint16,
            "list_available_serials": list_available_serials,
            "make_shm_names": make_shm_names,
        }[name]
    raise AttributeError(name)
