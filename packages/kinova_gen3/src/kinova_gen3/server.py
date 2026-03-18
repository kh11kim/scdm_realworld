from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path
import threading

import numpy as np

from kortex_api.Exceptions.KServerException import KServerException
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Base_pb2, ControlConfig_pb2

from kinova_gen3.device import DeviceConnection
from kinova_gen3.config import (
    KINOVA_JOINT_POSITION_LIMITS,
    KINOVA_SOFT_JOINT_ACCELERATION_LIMITS,
    KINOVA_SOFT_JOINT_SPEED_LIMITS,
)
from kinova_gen3.protocol import (
    ExecuteJointTrajectoryRequest,
    ExecuteJointTrajectoryResponse,
    GetKinematicLimitsRequest,
    GetMeasuredJointsRequest,
    KinematicLimitsResponse,
    MeasuredJointsResponse,
)


TIMEOUT_DURATION = 100.0


@dataclass
class ServerConfig:
    ip: str
    username: str
    password: str
    socket_path: str
    dt: float = 0.05
    echo_joints: bool = False


def _log(message: str) -> None:
    print(f"[kinova_gen3.server] {message}", flush=True)


def _check_for_end_or_abort(event: threading.Event):
    def _callback(notification, event: threading.Event = event):
        if notification.action_event in (
            Base_pb2.ACTION_END,
            Base_pb2.ACTION_ABORT,
        ):
            event.set()

    return _callback


def _measured_joints_rad(base: BaseClient) -> list[float]:
    joint_angles = base.GetMeasuredJointAngles()
    joints_rad = [float(np.deg2rad(joint_angle.value)) for joint_angle in joint_angles.joint_angles]
    wrapped: list[float] = []
    two_pi = 2.0 * np.pi
    for index, value in enumerate(joints_rad):
        limit = None
        if index < len(KINOVA_JOINT_POSITION_LIMITS):
            limit = KINOVA_JOINT_POSITION_LIMITS[index]
        if limit is None:
            wrapped.append(float((value + np.pi) % two_pi - np.pi))
            continue
        lower, upper = float(limit[0]), float(limit[1])
        candidates = [value + k * two_pi for k in range(-2, 3)]
        feasible = [candidate for candidate in candidates if lower <= candidate <= upper]
        if feasible:
            wrapped.append(float(min(feasible, key=lambda candidate: abs(candidate - value))))
            continue
        if abs((upper - lower) - two_pi) < 1e-3:
            wrapped.append(float((value - lower) % two_pi + lower))
            continue
        wrapped.append(float(value))
    return wrapped


def _execute_joint_trajectory(
    base: BaseClient,
    request: ExecuteJointTrajectoryRequest,
) -> ExecuteJointTrajectoryResponse:
    _log(
        "received ExecuteJointTrajectoryRequest "
        f"waypoints={len(request.waypoints_rad)} wait={request.wait}"
    )
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = False

    durations = request.durations_s
    if durations is not None and len(durations) != len(request.waypoints_rad):
        _log("rejecting trajectory: durations_s length does not match waypoints_rad length")
        return ExecuteJointTrajectoryResponse(
            error="durations_s length must match waypoints_rad length"
        )

    for index, joint_pose_rad in enumerate(request.waypoints_rad):
        waypoint = waypoints.waypoints.add()
        waypoint.name = f"waypoint_{index}"
        angular_waypoint = Base_pb2.AngularWaypoint()
        angular_waypoint.angles.extend([float(np.rad2deg(value)) for value in joint_pose_rad])
        angular_waypoint.duration = (
            float(durations[index]) if durations is not None else 1.0
        )
        waypoint.angular_waypoint.CopyFrom(angular_waypoint)

    validation = base.ValidateWaypointList(waypoints)
    errors = validation.trajectory_error_report.trajectory_error_elements
    if len(errors) > 0:
        _log(f"trajectory validation failed: {validation.trajectory_error_report}")
        return ExecuteJointTrajectoryResponse(error=str(validation.trajectory_error_report))

    if not request.wait:
        _log("sending trajectory without waiting for completion")
        base.ExecuteWaypointTrajectory(waypoints)
        _log("trajectory sent")
        return ExecuteJointTrajectoryResponse()

    event = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        _check_for_end_or_abort(event),
        Base_pb2.NotificationOptions(),
    )
    try:
        _log("sending trajectory and waiting for completion")
        base.ExecuteWaypointTrajectory(waypoints)
        finished = event.wait(TIMEOUT_DURATION)
    finally:
        base.Unsubscribe(notification_handle)

    if not finished:
        _log("trajectory wait timed out")
        return ExecuteJointTrajectoryResponse(error="timeout while waiting for trajectory")
    _log("trajectory completed")
    return ExecuteJointTrajectoryResponse()


def _get_kinematic_limits(
    control_config: ControlConfigClient,
    request: GetKinematicLimitsRequest,
) -> KinematicLimitsResponse:
    control_mode_info = ControlConfig_pb2.ControlModeInformation()
    control_mode_info.control_mode = int(request.control_mode)
    hard_limits = control_config.GetKinematicHardLimits()
    soft_limits = control_config.GetKinematicSoftLimits(control_mode_info)
    return KinematicLimitsResponse(
        control_mode=int(request.control_mode),
        hard_joint_speed_limits=[float(value) for value in hard_limits.joint_speed_limits],
        hard_joint_acceleration_limits=[
            float(value) for value in hard_limits.joint_acceleration_limits
        ],
        soft_joint_speed_limits=[float(value) for value in soft_limits.joint_speed_limits],
        soft_joint_acceleration_limits=[
            float(value) for value in soft_limits.joint_acceleration_limits
        ],
    )


def _apply_soft_limits(control_config: ControlConfigClient) -> None:
    if KINOVA_SOFT_JOINT_SPEED_LIMITS:
        speed_limits = ControlConfig_pb2.JointSpeedSoftLimits()
        speed_limits.control_mode = ControlConfig_pb2.ANGULAR_TRAJECTORY
        speed_limits.joint_speed_soft_limits.extend(KINOVA_SOFT_JOINT_SPEED_LIMITS)
        control_config.SetJointSpeedSoftLimits(speed_limits)
        _log(
            "applied soft joint speed limits: "
            f"{[round(value, 4) for value in KINOVA_SOFT_JOINT_SPEED_LIMITS]}"
        )
    if KINOVA_SOFT_JOINT_ACCELERATION_LIMITS:
        acceleration_limits = ControlConfig_pb2.JointAccelerationSoftLimits()
        acceleration_limits.control_mode = ControlConfig_pb2.ANGULAR_TRAJECTORY
        acceleration_limits.joint_acceleration_soft_limits.extend(
            KINOVA_SOFT_JOINT_ACCELERATION_LIMITS
        )
        control_config.SetJointAccelerationSoftLimits(acceleration_limits)
        _log(
            "applied soft joint acceleration limits: "
            f"{[round(value, 4) for value in KINOVA_SOFT_JOINT_ACCELERATION_LIMITS]}"
        )


def _handle_request(base: BaseClient, control_config: ControlConfigClient, request: object):
    def _error_response(message: str):
        if isinstance(request, GetMeasuredJointsRequest):
            return MeasuredJointsResponse(joints_rad=[], error=message)
        if isinstance(request, GetKinematicLimitsRequest):
            return KinematicLimitsResponse(
                control_mode=int(request.control_mode),
                hard_joint_speed_limits=[],
                hard_joint_acceleration_limits=[],
                soft_joint_speed_limits=[],
                soft_joint_acceleration_limits=[],
                error=message,
            )
        return ExecuteJointTrajectoryResponse(error=message)

    try:
        if isinstance(request, GetMeasuredJointsRequest):
            return MeasuredJointsResponse(joints_rad=_measured_joints_rad(base))
        if isinstance(request, GetKinematicLimitsRequest):
            return _get_kinematic_limits(control_config, request)
        if isinstance(request, ExecuteJointTrajectoryRequest):
            return _execute_joint_trajectory(base, request)
        _log(f"received unsupported request type: {type(request).__name__}")
        return _error_response(f"unsupported request type: {type(request).__name__}")
    except KServerException as exc:
        _log(
            "Kortex API error: "
            f"error_code={exc.get_error_code()} "
            f"sub_error_code={exc.get_error_sub_code()} {exc}"
        )
        return _error_response(
            (
                f"Kortex API error: error_code={exc.get_error_code()} "
                f"sub_error_code={exc.get_error_sub_code()} {exc}"
            )
        )
    except Exception as exc:
        _log(f"unexpected server error: {exc}")
        return _error_response(str(exc))


def _echo_loop(base: BaseClient, dt: float, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            joints_rad = _measured_joints_rad(base)
            print("Measured joints (rad):", [round(value, 6) for value in joints_rad])
        except Exception as exc:
            print(f"Joint echo error: {exc}")
        stop_event.wait(dt)


def serve_forever(config: ServerConfig) -> int:
    socket_path = Path(config.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    with DeviceConnection(
        ip=config.ip,
        username=config.username,
        password=config.password,
    ) as router:
        base = BaseClient(router)
        control_config = ControlConfigClient(router)
        _apply_soft_limits(control_config)
        listener = Listener(str(socket_path), family="AF_UNIX")
        stop_event = threading.Event()
        echo_thread: threading.Thread | None = None
        if config.echo_joints:
            echo_thread = threading.Thread(
                target=_echo_loop,
                args=(base, config.dt, stop_event),
                daemon=True,
            )
            echo_thread.start()

        print(f"Connected to Kinova Gen3 at {config.ip}")
        print(f"Listening on {socket_path}")
        print("Server loop running. Press Ctrl+C to stop.")
        try:
            while True:
                conn = listener.accept()
                try:
                    request = conn.recv()
                    response = _handle_request(base, control_config, request)
                    conn.send(response)
                    if isinstance(request, ExecuteJointTrajectoryRequest):
                        _log(f"response sent: {type(response).__name__}")
                finally:
                    conn.close()
        except KeyboardInterrupt:
            return 0
        finally:
            stop_event.set()
            if echo_thread is not None:
                echo_thread.join(timeout=1.0)
            listener.close()
            if socket_path.exists():
                socket_path.unlink()


__all__ = ["ServerConfig", "serve_forever"]
