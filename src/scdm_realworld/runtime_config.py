from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import yaml


DEFAULT_APP_CONFIG_PATH = Path("assets/config.yaml")
DEFAULT_URDF_PATH = Path("assets/gen3_allegro/gen3_allegro.urdf")
DEFAULT_ARM_API_MODULE = "kinova_gen3.client"
DEFAULT_HAND_API_MODULE = "allegro_v5.client"


@dataclass(frozen=True)
class ArmApi:
    module: str
    get_joints: Callable[[], list[float]]
    execute_trajectory: Callable[[list[list[float]], float], None]
    joint_position_limits: list[list[float] | tuple[float, float] | None] | tuple[Any, ...] | None


@dataclass(frozen=True)
class HandApi:
    module: str
    get_joints: Callable[[], list[float]]
    set_joints: Callable[[list[float]], None]
    goto_joints: Callable[[list[float], float], None]


def _resolve_q_presets(
    config: dict[str, object],
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    q_preset_cfg = config.get("q_preset", {})
    if not isinstance(q_preset_cfg, dict):
        return {}
    resolved: dict[str, np.ndarray] = {}
    for key, values in q_preset_cfg.items():
        if not isinstance(key, str) or not key.startswith(prefix) or not key.endswith("_q"):
            continue
        if not isinstance(values, list):
            continue
        name = key.removeprefix(prefix).removesuffix("_q")
        resolved[name] = np.asarray(values, dtype=np.float64)
    return resolved


def load_runtime_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid runtime config YAML: {path}")
    return payload


def resolve_robot_urdf(config: dict[str, object], *, fallback: Path = DEFAULT_URDF_PATH) -> Path:
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, dict):
        return fallback
    urdf = robot_cfg.get("urdf")
    if not isinstance(urdf, str) or not urdf:
        return fallback
    return Path(urdf)


def resolve_arm_home_q(config: dict[str, object]) -> np.ndarray | None:
    q_preset_cfg = config.get("q_preset", {})
    if not isinstance(q_preset_cfg, dict):
        return None
    values = q_preset_cfg.get("arm_home_q")
    if not isinstance(values, list):
        return None
    return np.asarray(values, dtype=np.float64)


def resolve_hand_home_q(config: dict[str, object]) -> np.ndarray | None:
    q_preset_cfg = config.get("q_preset", {})
    if not isinstance(q_preset_cfg, dict):
        return None
    values = q_preset_cfg.get("hand_home_q")
    if not isinstance(values, list):
        return None
    return np.asarray(values, dtype=np.float64)


def resolve_arm_presets(config: dict[str, object]) -> dict[str, np.ndarray]:
    return _resolve_q_presets(config, prefix="arm_")


def resolve_hand_presets(config: dict[str, object]) -> dict[str, np.ndarray]:
    return _resolve_q_presets(config, prefix="hand_")


def _resolve_api_module_name(
    config: dict[str, object],
    *,
    kind: str,
    fallback: str,
) -> str:
    api_cfg = config.get("api", {})
    if not isinstance(api_cfg, dict):
        return fallback
    kind_cfg = api_cfg.get(kind, {})
    if not isinstance(kind_cfg, dict):
        return fallback
    module_name = kind_cfg.get("module")
    if not isinstance(module_name, str) or not module_name:
        return fallback
    return module_name


def _import_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _get_callable(module: ModuleType, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise AttributeError(f"{module.__name__}.{name} is required")
    return value


def resolve_arm_api(config: dict[str, object]) -> ArmApi:
    module_name = _resolve_api_module_name(
        config,
        kind="arm",
        fallback=DEFAULT_ARM_API_MODULE,
    )
    module = _import_module(module_name)
    return ArmApi(
        module=module_name,
        get_joints=_get_callable(module, "get_joints"),
        execute_trajectory=_get_callable(module, "execute_trajectory"),
        joint_position_limits=getattr(module, "JOINT_POSITION_LIMITS", None),
    )


def resolve_hand_api(config: dict[str, object]) -> HandApi:
    module_name = _resolve_api_module_name(
        config,
        kind="hand",
        fallback=DEFAULT_HAND_API_MODULE,
    )
    module = _import_module(module_name)
    return HandApi(
        module=module_name,
        get_joints=_get_callable(module, "get_joints"),
        set_joints=_get_callable(module, "set_joints"),
        goto_joints=_get_callable(module, "goto_joints"),
    )
