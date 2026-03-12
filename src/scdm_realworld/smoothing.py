from __future__ import annotations

import time
from typing import Callable

import numpy as np


def _interpolate(q1: np.ndarray, q2: np.ndarray, step_size: float) -> list[np.ndarray]:
    delta = q2 - q1
    distance = np.linalg.norm(delta)
    if distance == 0.0:
        return []
    steps = max(1, int(np.ceil(distance / step_size)))
    return [q1 + (i / steps) * delta for i in range(1, steps)]


def _path_cost(trajectory: list[np.ndarray]) -> float:
    if len(trajectory) < 2:
        return 0.0
    return float(
        sum(np.linalg.norm(q2 - q1) for q1, q2 in zip(trajectory[:-1], trajectory[1:], strict=True))
    )


def smooth_trajectory(
    trajectory: list[np.ndarray] | None,
    col_fn: Callable[[np.ndarray], bool],
    *,
    step_size: float = 0.05,
    max_iterations: int = 200,
    max_time: float = 1.0,
) -> list[np.ndarray] | None:
    if trajectory is None or len(trajectory) <= 2:
        return trajectory

    path = [np.asarray(q, dtype=np.float64).copy() for q in trajectory]
    start_time = time.time()

    for _ in range(max_iterations):
        if time.time() - start_time > max_time:
            break
        if len(path) <= 2:
            break

        i = np.random.randint(0, len(path) - 1)
        j = np.random.randint(i + 1, len(path))
        if j <= i + 1:
            continue

        q1, q2 = path[i], path[j]
        shortcut = _interpolate(q1, q2, step_size)
        if any(col_fn(q) for q in shortcut):
            continue

        candidate = path[: i + 1] + shortcut + path[j:]
        if _path_cost(candidate) < _path_cost(path):
            path = candidate

    return path
