from __future__ import annotations

from pathlib import Path
from typing import Iterable
from typing import Callable, Sequence

import numpy as np

from scdm_realworld.robot_model import DEFAULT_URDF_PATH, RobotModel


class RobotReal(RobotModel):
    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        *,
        get_joints_fn: Callable[[], list[float]],
        execute_trajectory_fn: Callable[[list[list[float]], float], None],
        joint_position_limits: Sequence[Sequence[float] | None] | None = None,
    ) -> RobotReal:
        robot = super().from_urdf(urdf_path)
        robot.__class__ = cls
        robot._get_joints_fn = get_joints_fn
        robot._execute_trajectory_fn = execute_trajectory_fn
        robot._joint_position_limits = joint_position_limits
        return robot

    def get_joints(self) -> np.ndarray:
        return np.asarray(self._get_joints_fn(), dtype=np.float64)

    def sync_from_real(self) -> np.ndarray:
        q = self.get_joints()
        self.set_joint_positions(q=q)
        return q

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bounds, upper_bounds = super().get_joint_limits()
        if self._joint_position_limits is None:
            return lower_bounds, upper_bounds
        for index, limit in enumerate(self._joint_position_limits):
            if limit is None:
                continue
            lower_bounds[index] = float(limit[0])
            upper_bounds[index] = float(limit[1])
        return lower_bounds, upper_bounds

    def execute_trajectory(
        self,
        trajectory: Iterable[Iterable[float]],
        durations_s: list[float] | None = None,
        *,
        total_time_s: float = 5.0,
        wait: bool = True,
    ) -> None:
        if not wait:
            raise ValueError("RobotReal.execute_trajectory only supports wait=True")
        waypoints_rad = [list(np.asarray(q, dtype=np.float64)) for q in trajectory]
        if durations_s is not None:
            raise ValueError("RobotReal.execute_trajectory does not support explicit durations_s")
        self._execute_trajectory_fn(waypoints_rad, float(total_time_s))


__all__ = ["RobotReal"]
