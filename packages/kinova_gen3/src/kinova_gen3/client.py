from __future__ import annotations

from multiprocessing.connection import Client

import numpy as np
from scipy.interpolate import CubicSpline

from kinova_gen3.config import KINOVA_SOCKET_PATH
from kinova_gen3.config import KINOVA_JOINT_POSITION_LIMITS as JOINT_POSITION_LIMITS
from kinova_gen3.protocol import (
    ExecuteJointTrajectoryRequest,
    ExecuteJointTrajectoryResponse,
    GetKinematicLimitsRequest,
    GetMeasuredJointsRequest,
    KinematicLimitsResponse,
    MeasuredJointsResponse,
)


def _compute_waypoint_durations(
    waypoints_rad: list[list[float]],
    total_time_s: float,
) -> list[float]:
    if len(waypoints_rad) == 0:
        return []
    if len(waypoints_rad) == 1:
        return [float(total_time_s)]

    qs = [np.asarray(q, dtype=np.float64) for q in waypoints_rad]
    segment_lengths = [
        float(np.max(np.abs(q_next - q_curr)))
        for q_curr, q_next in zip(qs[:-1], qs[1:], strict=True)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-9:
        per_waypoint = float(total_time_s) / len(waypoints_rad)
        return [per_waypoint for _ in waypoints_rad]

    segment_durations = [float(total_time_s) * length / total_length for length in segment_lengths]
    min_duration = 0.1
    durations = [max(min_duration, segment_durations[0])] + [
        max(min_duration, value) for value in segment_durations
    ]
    scale = float(total_time_s) / sum(durations)
    return [value * scale for value in durations]


def _resample_trajectory_for_execution(
    trajectory: list[list[float]],
    *,
    current_q: list[float],
    num_waypoints: int = 30,
) -> list[list[float]]:
    if len(trajectory) == 0:
        return []

    qs = np.asarray(trajectory, dtype=np.float64)
    current = np.asarray(current_q, dtype=np.float64).reshape(1, -1)
    if qs.ndim != 2:
        raise ValueError(f"trajectory must have shape (N, dof), got {qs.shape}")
    if qs.shape[1] != current.shape[1]:
        raise ValueError(
            f"trajectory dof {qs.shape[1]} does not match current dof {current.shape[1]}"
        )

    qs = qs.copy()
    qs[0] = current[0]
    if len(qs) == 1:
        return qs.tolist()

    sample_count = max(num_waypoints, len(qs))
    s_orig = np.linspace(0.0, 1.0, len(qs))
    t = np.linspace(0.0, 1.0, sample_count)
    u = 3.0 * t**2 - 2.0 * t**3

    resampled = np.empty((sample_count, qs.shape[1]), dtype=np.float64)
    for joint_idx in range(qs.shape[1]):
        spline = CubicSpline(s_orig, qs[:, joint_idx], bc_type="clamped")
        resampled[:, joint_idx] = spline(u)
    resampled[0] = current[0]
    resampled[-1] = qs[-1]
    return resampled.tolist()


def _call(request, *, socket_path: str = KINOVA_SOCKET_PATH):
    if isinstance(request, ExecuteJointTrajectoryRequest):
        print(
            "[kinova_gen3.client] sending ExecuteJointTrajectoryRequest "
            f"waypoints={len(request.waypoints_rad)} wait={request.wait}",
            flush=True,
        )
    conn = Client(socket_path, family="AF_UNIX")
    try:
        conn.send(request)
        response = conn.recv()
        if isinstance(request, ExecuteJointTrajectoryRequest):
            print(
                "[kinova_gen3.client] received "
                f"{type(response).__name__}",
                flush=True,
            )
        return response
    finally:
        conn.close()


def get_measured_joints(*, socket_path: str = KINOVA_SOCKET_PATH) -> MeasuredJointsResponse:
    response = _call(GetMeasuredJointsRequest(), socket_path=socket_path)
    if not isinstance(response, MeasuredJointsResponse):
        raise TypeError(f"Unexpected response type: {type(response).__name__}")
    return response


def get_joints(*, socket_path: str = KINOVA_SOCKET_PATH) -> list[float]:
    response = get_measured_joints(socket_path=socket_path)
    if response.is_error():
        raise RuntimeError(response.error)
    return response.joints_rad


def get_kinematic_limits(
    *,
    control_mode: int = 4,
    socket_path: str = KINOVA_SOCKET_PATH,
) -> KinematicLimitsResponse:
    response = _call(
        GetKinematicLimitsRequest(control_mode=control_mode),
        socket_path=socket_path,
    )
    if not isinstance(response, KinematicLimitsResponse):
        raise TypeError(f"Unexpected response type: {type(response).__name__}")
    return response


def execute_joint_trajectory(
    waypoints_rad: list[list[float]],
    durations_s: list[float] | None = None,
    *,
    total_time_s: float = 5.0,
    wait: bool = True,
    socket_path: str = KINOVA_SOCKET_PATH,
) -> ExecuteJointTrajectoryResponse:
    resolved_durations = (
        _compute_waypoint_durations(waypoints_rad, total_time_s)
        if durations_s is None
        else durations_s
    )
    response = _call(
        ExecuteJointTrajectoryRequest(
            waypoints_rad=waypoints_rad,
            durations_s=resolved_durations,
            wait=wait,
        ),
        socket_path=socket_path,
    )
    if not isinstance(response, ExecuteJointTrajectoryResponse):
        raise TypeError(f"Unexpected response type: {type(response).__name__}")
    return response


def send_trajectory(
    waypoints_rad: list[list[float]],
    durations_s: list[float] | None = None,
    *,
    total_time_s: float = 5.0,
    wait: bool = True,
    socket_path: str = KINOVA_SOCKET_PATH,
) -> None:
    response = execute_joint_trajectory(
        waypoints_rad,
        durations_s,
        total_time_s=total_time_s,
        wait=wait,
        socket_path=socket_path,
    )
    if response.is_error():
        raise RuntimeError(response.error)


def execute_trajectory(
    trajectory: list[list[float]],
    total_time_s: float,
) -> None:
    if len(trajectory) == 0:
        return
    current_q = get_joints()
    execution_trajectory = _resample_trajectory_for_execution(
        trajectory,
        current_q=current_q,
    )
    send_trajectory(
        execution_trajectory,
        total_time_s=total_time_s,
        wait=True,
    )


__all__ = [
    "execute_trajectory",
    "execute_joint_trajectory",
    "get_joints",
    "get_kinematic_limits",
    "get_measured_joints",
    "send_trajectory",
]
