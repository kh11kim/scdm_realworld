"""
Lightweight polling client for Allegro Hand state.

Externally this package now talks to the local multiprocessing server rather
than directly to ZMQ. The class name is kept for compatibility with existing
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from allegro_v5.client import DEFAULT_SOCKET_PATH, get_state


@dataclass
class TelemetryConfig:
    host: str = "localhost"
    port: int = 5556
    subscribe: str = ""
    rcv_hwm: int = 1
    conflate: bool = True
    timeout_ms: Optional[int] = None
    socket_path: str = DEFAULT_SOCKET_PATH


class ZmqTelemetryClient:
    def __init__(self, config: TelemetryConfig):
        self._config = config

    def recv_latest(self) -> dict:
        state = get_state(socket_path=self._config.socket_path)
        if state.is_error():
            raise RuntimeError(state.error)
        return {
            "frame": state.frame,
            "motion": state.motion,
            "position": [] if state.position is None else state.position,
            "torque": [] if state.torque is None else state.torque,
            "tactile": [] if state.tactile is None else state.tactile,
            "temperature": [] if state.temperature is None else state.temperature,
            "imu_rpy": [] if state.imu_rpy is None else state.imu_rpy,
        }

    def close(self) -> None:
        return None

    def __enter__(self) -> "ZmqTelemetryClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
