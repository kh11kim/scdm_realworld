from __future__ import annotations

from typing import Callable

import numpy as np

from scdm_realworld.rrt_connect import rrt_connect
from scdm_realworld.smoothing import smooth_trajectory


def plan(
    q0: np.ndarray,
    qg: np.ndarray,
    col_fn: Callable[[np.ndarray], bool],
    joint_limits: tuple[np.ndarray, np.ndarray],
    rrt_max_time: float,
    smooth_max_time: float,
) -> list[np.ndarray] | None:
    trajectory = rrt_connect(
        q0,
        qg,
        col_fn,
        joint_limits=joint_limits,
        max_time=rrt_max_time,
    )
    if trajectory is None:
        return None
    return smooth_trajectory(
        trajectory,
        col_fn,
        max_time=smooth_max_time,
    )
