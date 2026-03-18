from __future__ import annotations

from multiprocessing.connection import Client
import time

import numpy as np

from allegro_v5.protocol import (
    AckResponse,
    GetStateRequest,
    SetDesiredPositionRequest,
    StateResponse,
)


DEFAULT_SOCKET_PATH = "/tmp/allegro_v5.sock"


def _call(request, *, socket_path: str = DEFAULT_SOCKET_PATH, timeout_s: float | None = 1.0):
    conn = Client(socket_path, family="AF_UNIX")
    try:
        should_log = not isinstance(request, GetStateRequest)
        if should_log:
            print(f"[allegro_v5.client] send {type(request).__name__}", flush=True)
        conn.send(request)
        if timeout_s is not None and not conn.poll(timeout_s):
            raise TimeoutError(f"Timed out waiting for response on {socket_path}")
        response = conn.recv()
        if should_log:
            print(f"[allegro_v5.client] recv {type(response).__name__}", flush=True)
        return response
    finally:
        conn.close()


def get_state(*, socket_path: str = DEFAULT_SOCKET_PATH, timeout_s: float | None = 1.0) -> StateResponse:
    response = _call(GetStateRequest(), socket_path=socket_path, timeout_s=timeout_s)
    if not isinstance(response, StateResponse):
        raise TypeError(f"Unexpected response type: {type(response).__name__}")
    return response


def get_joints(*, socket_path: str = DEFAULT_SOCKET_PATH, timeout_s: float | None = 1.0) -> list[float]:
    state = get_state(socket_path=socket_path, timeout_s=timeout_s)
    if state.is_error():
        raise RuntimeError(state.error)
    if state.position is None:
        raise RuntimeError("State response does not include joint positions")
    return list(state.position)


def set_desired_positions(
    desired_position: list[float],
    *,
    socket_path: str = DEFAULT_SOCKET_PATH,
    timeout_s: float | None = 1.0,
) -> None:
    response = _call(
        SetDesiredPositionRequest(desired_position=list(desired_position)),
        socket_path=socket_path,
        timeout_s=timeout_s,
    )
    if not isinstance(response, AckResponse):
        raise TypeError(f"Unexpected response type: {type(response).__name__}")
    if response.is_error():
        raise RuntimeError(response.error)


def set_joints(
    desired_position: list[float],
    *,
    socket_path: str = DEFAULT_SOCKET_PATH,
    timeout_s: float | None = 1.0,
) -> None:
    set_desired_positions(desired_position, socket_path=socket_path, timeout_s=timeout_s)


def goto_joints(
    q: list[float],
    *,
    target_q_vel: float = 0.5,
    socket_path: str = DEFAULT_SOCKET_PATH,
    timeout_s: float | None = 1.0,
) -> None:
    dt = 0.02
    current = np.asarray(get_joints(socket_path=socket_path, timeout_s=timeout_s), dtype=np.float64)
    desired = np.asarray(q, dtype=np.float64)
    if current.shape != desired.shape:
        raise ValueError(f"q must have shape {current.shape}, got {desired.shape}")
    target_q_vel = max(float(target_q_vel), 1e-3)
    duration_s = max(float(np.max(np.abs(desired - current)) / target_q_vel), dt)
    num_steps = max(2, int(np.ceil(duration_s / dt)))
    for ratio in np.linspace(0.0, 1.0, num_steps):
        q = current + ratio * (desired - current)
        set_joints(q.tolist(), socket_path=socket_path, timeout_s=timeout_s)
        time.sleep(dt)
