from __future__ import annotations

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)), dtype=np.float64)
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)), dtype=np.float64)
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
        return (w, x, y, z)

    diag = np.diag(rotation)
    index = int(np.argmax(diag))
    if index == 0:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif index == 1:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


def compute_frustum_params(
    calib: dict[str, object],
) -> tuple[float, float, np.ndarray, tuple[float, float, float, float]]:
    cam_pose = calib["cam_pose"]
    intrinsic = calib["intrinsic"]
    if not isinstance(cam_pose, dict) or not isinstance(intrinsic, dict):
        raise ValueError("Calibration YAML is missing cam_pose or intrinsic sections.")

    trans = np.asarray(cam_pose["trans"], dtype=np.float64)
    rpy = np.asarray(cam_pose["rpy"], dtype=np.float64)
    width = float(intrinsic["width"])
    height = float(intrinsic["height"])
    fy = float(intrinsic["fy"])

    aspect = width / height
    fov = 2.0 * np.arctan2(height / 2.0, fy)
    rotation = rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    wxyz = matrix_to_wxyz(rotation)
    return float(fov), float(aspect), trans, wxyz


def compute_camera_pose(calib: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    cam_pose = calib["cam_pose"]
    if not isinstance(cam_pose, dict):
        raise ValueError("Calibration YAML is missing cam_pose section.")
    trans = np.asarray(cam_pose["trans"], dtype=np.float64)
    rpy = np.asarray(cam_pose["rpy"], dtype=np.float64)
    rotation = rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return rotation, trans


def project_depth_to_world(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    intrinsic: dict[str, object],
    rotation_wc: np.ndarray,
    translation_wc: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    if step <= 0:
        raise ValueError("pcd step must be positive")

    depth_m = depth_mm.astype(np.float32) * 0.001
    valid_mask = depth_m > 0.0
    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    if valid_indices.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    sampled = valid_indices[::step]
    _, width = depth_mm.shape
    ys, xs = np.divmod(sampled, width)
    z = depth_m[ys, xs]

    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])

    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    camera_points = np.stack((x, y, z), axis=1)
    world_points = (rotation_wc @ camera_points.T).T + translation_wc.reshape(1, 3)
    colors = rgb[ys, xs]
    return world_points.astype(np.float32), colors.astype(np.uint8)
