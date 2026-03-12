from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid kinova_gen3 config: {CONFIG_PATH}")
    return payload


_CONFIG = load_config()

KINOVA_IP = _CONFIG["ip"]
KINOVA_USERNAME = _CONFIG.get("username", "admin")
KINOVA_PASSWORD = _CONFIG.get("password", "admin")
KINOVA_SOCKET_PATH = _CONFIG.get("socket_path", "/tmp/kinova_gen3.sock")
KINOVA_HOME_Q = [float(value) for value in _CONFIG.get("home_q_rad", [])]
KINOVA_JOINT_POSITION_LIMITS = [
    None if value is None else (float(value[0]), float(value[1]))
    for value in _CONFIG.get("joint_position_limits_rad", [])
]
KINOVA_SOFT_JOINT_SPEED_LIMITS = [
    float(value) for value in _CONFIG.get("soft_joint_speed_limits_deg_s", [])
]
KINOVA_SOFT_JOINT_ACCELERATION_LIMITS = [
    float(value) for value in _CONFIG.get("soft_joint_acceleration_limits_deg_s2", [])
]


__all__ = [
    "CONFIG_PATH",
    "KINOVA_IP",
    "KINOVA_USERNAME",
    "KINOVA_PASSWORD",
    "KINOVA_SOCKET_PATH",
    "KINOVA_HOME_Q",
    "KINOVA_JOINT_POSITION_LIMITS",
    "KINOVA_SOFT_JOINT_SPEED_LIMITS",
    "KINOVA_SOFT_JOINT_ACCELERATION_LIMITS",
    "load_config",
]
