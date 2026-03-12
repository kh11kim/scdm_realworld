from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GetStateRequest:
    pass


@dataclass(frozen=True)
class SetDesiredPositionRequest:
    desired_position: list[float]


@dataclass
class StateResponse:
    frame: int = 0
    motion: str = ""
    position: list[float] | None = None
    torque: list[float] | None = None
    tactile: list[int] | None = None
    temperature: list[int] | None = None
    imu_rpy: list[float] | None = None
    error: str = ""

    def is_error(self) -> bool:
        return self.error != ""


@dataclass
class AckResponse:
    ok: bool = True
    error: str = ""

    def is_error(self) -> bool:
        return self.error != ""
