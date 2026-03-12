from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np


@dataclass
class _Node:
    q: np.ndarray
    parent: _Node | None = None


def _distance(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(np.linalg.norm(q1 - q2))


def _nearest(tree: list[_Node], q: np.ndarray) -> _Node:
    return min(tree, key=lambda node: _distance(node.q, q))


def _interpolate(q1: np.ndarray, q2: np.ndarray, step_size: float) -> list[np.ndarray]:
    delta = q2 - q1
    distance = np.linalg.norm(delta)
    if distance == 0.0:
        return []
    steps = max(1, int(np.ceil(distance / step_size)))
    return [q1 + (i / steps) * delta for i in range(1, steps + 1)]


def _extend(
    tree: list[_Node],
    target: np.ndarray,
    col_fn: Callable[[np.ndarray], bool],
    step_size: float,
) -> tuple[_Node, bool]:
    nearest = _nearest(tree, target)
    last = nearest
    reached = False
    for q in _interpolate(nearest.q, target, step_size):
        q = np.asarray(q, dtype=np.float64)
        if col_fn(q):
            break
        last = _Node(q=q, parent=last)
        tree.append(last)
        if np.allclose(q, target):
            reached = True
    return last, reached


def _is_straight_line_feasible(
    q0: np.ndarray,
    qg: np.ndarray,
    col_fn: Callable[[np.ndarray], bool],
    step_size: float,
) -> bool:
    for q in _interpolate(q0, qg, step_size):
        if col_fn(np.asarray(q, dtype=np.float64)):
            return False
    return True


def _trace_path(node: _Node) -> list[np.ndarray]:
    path: list[np.ndarray] = []
    while node is not None:
        path.append(node.q.copy())
        node = node.parent
    path.reverse()
    return path


def _connect_paths(a: _Node, b: _Node) -> list[np.ndarray]:
    path_a = _trace_path(a)
    path_b = _trace_path(b)
    return path_a + list(reversed(path_b[:-1]))


def rrt_connect(
    q0: np.ndarray,
    qg: np.ndarray,
    col_fn: Callable[[np.ndarray], bool],
    *,
    joint_limits: tuple[np.ndarray, np.ndarray],
    step_size: float = 0.1,
    goal_tolerance: float = 0.05,
    max_iterations: int = 2000,
    max_time: float = 5.0,
    goal_bias: float = 0.2,
) -> list[np.ndarray] | None:
    q0 = np.asarray(q0, dtype=np.float64)
    qg = np.asarray(qg, dtype=np.float64)
    q_min = np.asarray(joint_limits[0], dtype=np.float64)
    q_max = np.asarray(joint_limits[1], dtype=np.float64)

    if q0.shape != qg.shape:
        raise ValueError(f"q0 and qg must have the same shape, got {q0.shape} and {qg.shape}")
    if q_min.shape != q0.shape or q_max.shape != q0.shape:
        raise ValueError("joint_limits must have same shape as q0/qg")
    if col_fn(q0) or col_fn(qg):
        return None
    if _is_straight_line_feasible(q0, qg, col_fn, step_size):
        return [q0.copy(), qg.copy()]

    tree_a = [_Node(q=q0.copy())]
    tree_b = [_Node(q=qg.copy())]
    start_time = time.time()

    for iteration in range(max_iterations):
        if time.time() - start_time > max_time:
            return None

        if np.random.random() < goal_bias:
            q_rand = tree_b[-1].q.copy()
        else:
            q_rand = np.random.uniform(q_min, q_max)

        node_a, _ = _extend(tree_a, q_rand, col_fn, step_size)
        node_b, _ = _extend(tree_b, node_a.q, col_fn, step_size)

        if _distance(node_a.q, node_b.q) <= goal_tolerance:
            return _connect_paths(node_a, node_b) if iteration % 2 == 0 else _connect_paths(node_b, node_a)

        tree_a, tree_b = tree_b, tree_a

    return None
