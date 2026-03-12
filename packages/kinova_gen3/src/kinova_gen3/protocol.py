from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GetMeasuredJointsRequest:
    pass


@dataclass(frozen=True)
class GetKinematicLimitsRequest:
    control_mode: int = 4


@dataclass(frozen=True)
class ExecuteJointTrajectoryRequest:
    waypoints_rad: list[list[float]]
    durations_s: list[float] | None = None
    wait: bool = True


@dataclass(frozen=True)
class MeasuredJointsResponse:
    joints_rad: list[float]
    error: str = ""

    def is_error(self) -> bool:
        return self.error != ""


@dataclass(frozen=True)
class KinematicLimitsResponse:
    control_mode: int
    hard_joint_speed_limits: list[float]
    hard_joint_acceleration_limits: list[float]
    soft_joint_speed_limits: list[float]
    soft_joint_acceleration_limits: list[float]
    error: str = ""

    def is_error(self) -> bool:
        return self.error != ""


@dataclass(frozen=True)
class ExecuteJointTrajectoryResponse:
    error: str = ""

    def is_error(self) -> bool:
        return self.error != ""


__all__ = [
    "GetMeasuredJointsRequest",
    "GetKinematicLimitsRequest",
    "ExecuteJointTrajectoryRequest",
    "MeasuredJointsResponse",
    "KinematicLimitsResponse",
    "ExecuteJointTrajectoryResponse",
]
