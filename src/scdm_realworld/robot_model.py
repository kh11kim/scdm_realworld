from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yourdfpy

from scdm_realworld.collision import (
    DEFAULT_SPHERE_SET_PATH,
    compute_world_spheres,
    has_collision,
    load_link_spheres,
)
from scdm_realworld.environment import BoxEnvironment
from scdm_realworld.planning import plan


DEFAULT_URDF_PATH = Path("assets/gen3_allegro/gen3_allegro.urdf")


@dataclass(frozen=True)
class LinkPose:
    link_name: str
    transform: np.ndarray


class RobotModel:
    def __init__(
        self,
        urdf: yourdfpy.URDF,
        urdf_path: Path,
        link_spheres: dict[str, list] | None = None,
    ) -> None:
        self._urdf = urdf
        self._urdf_path = urdf_path
        self._joint_names = tuple(self._urdf.actuated_joint_names)
        self._link_names = tuple(link.name for link in self._urdf.link_map.values())
        self._joint_index = {name: idx for idx, name in enumerate(self._joint_names)}
        self._arm_joint_names = tuple(
            joint_name for joint_name in self._joint_names if joint_name.startswith("gen3_joint_")
        )
        self._default_configuration = self._make_default_configuration()
        self._configuration = self._default_configuration.copy()
        self._link_spheres = (
            load_link_spheres(DEFAULT_SPHERE_SET_PATH)
            if link_spheres is None
            else link_spheres
        )
        self._urdf.update_cfg(self._configuration)

    @classmethod
    def from_urdf(cls, urdf_path: str | Path = DEFAULT_URDF_PATH) -> RobotModel:
        path = Path(urdf_path).resolve()
        urdf = yourdfpy.URDF.load(
            path,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
        )
        return cls(urdf=urdf, urdf_path=path)

    @property
    def urdf_path(self) -> Path:
        return self._urdf_path

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def link_names(self) -> tuple[str, ...]:
        return self._link_names

    @property
    def configuration(self) -> np.ndarray:
        return self._configuration[: len(self._arm_joint_names)].copy()

    @property
    def visual_configuration(self) -> np.ndarray:
        return self._configuration.copy()

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return self._arm_joint_names

    def _make_default_configuration(self) -> np.ndarray:
        values: list[float] = []
        for joint_name in self._joint_names:
            joint = self._urdf.joint_map[joint_name]
            lower = -np.pi if joint.limit is None or joint.limit.lower is None else float(joint.limit.lower)
            upper = np.pi if joint.limit is None or joint.limit.upper is None else float(joint.limit.upper)
            values.append(0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0)
        return np.asarray(values, dtype=np.float64)

    def _arm_to_full_q(self, q: Iterable[float]) -> np.ndarray:
        q_array = np.asarray(tuple(q), dtype=np.float64)
        if q_array.shape != (len(self._arm_joint_names),):
            raise ValueError(
                f"q must have shape ({len(self._arm_joint_names)},), got {q_array.shape}"
            )
        full_q = self._default_configuration.copy()
        full_q[: len(self._arm_joint_names)] = q_array
        return full_q

    def _set_full_configuration(self, q_full: np.ndarray) -> None:
        self._configuration = np.asarray(q_full, dtype=np.float64).copy()
        self._urdf.update_cfg(self._configuration)

    def set_joint_positions(
        self,
        joint_positions: dict[str, float] | None = None,
        *,
        q: Iterable[float] | None = None,
    ) -> np.ndarray:
        if joint_positions is not None and q is not None:
            raise ValueError("Provide either joint_positions or q, not both.")

        if joint_positions is not None:
            configuration = self._configuration.copy()
            unknown = sorted(set(joint_positions) - set(self._joint_index))
            if unknown:
                raise ValueError(f"Unknown joint names: {', '.join(unknown)}")
            for name, value in joint_positions.items():
                configuration[self._joint_index[name]] = float(value)
        elif q is not None:
            configuration = self._arm_to_full_q(q)
        else:
            configuration = self._configuration.copy()

        self._set_full_configuration(configuration)
        return self.configuration

    def get_link_pose(self, link_name: str, *, frame_from: str | None = "world") -> np.ndarray:
        if link_name not in self._urdf.link_map:
            raise ValueError(f"Unknown link name: {link_name}")
        transform = self._urdf.get_transform(link_name, frame_from=frame_from)
        return np.asarray(transform, dtype=np.float64)

    def get_all_link_poses(
        self, *, frame_from: str | None = "world"
    ) -> dict[str, np.ndarray]:
        return {
            link_name: self.get_link_pose(link_name, frame_from=frame_from)
            for link_name in self._link_names
        }

    def get_link_poses(self, link_names: Iterable[str], *, frame_from: str | None = "world") -> dict[str, np.ndarray]:
        return {
            link_name: self.get_link_pose(link_name, frame_from=frame_from)
            for link_name in link_names
        }

    def is_collision(self, q: Iterable[float], box_env: BoxEnvironment) -> bool:
        self.set_joint_positions(q=q)
        link_poses = self.get_link_poses(self._link_spheres.keys())
        world_spheres = compute_world_spheres(self._link_spheres, link_poses)
        return has_collision(world_spheres, list(box_env.boxes))

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bounds = []
        upper_bounds = []
        for joint_name in self._arm_joint_names:
            joint = self._urdf.joint_map[joint_name]
            if joint.limit is None:
                lower_bounds.append(-2.0 * np.pi)
                upper_bounds.append(2.0 * np.pi)
            else:
                lower = -2.0 * np.pi if joint.limit.lower is None else float(joint.limit.lower)
                upper = 2.0 * np.pi if joint.limit.upper is None else float(joint.limit.upper)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
        return np.asarray(lower_bounds, dtype=np.float64), np.asarray(upper_bounds, dtype=np.float64)

    def get_plan_fn(self, box_env: BoxEnvironment):
        col_fn = partial(self.is_collision, box_env=box_env)
        joint_limits = self.get_joint_limits()

        def _plan_fn(
            q0: np.ndarray,
            qg: np.ndarray,
            *,
            rrt_max_time: float,
            smooth_max_time: float,
        ) -> list[np.ndarray] | None:
            return plan(
                q0,
                qg,
                col_fn,
                joint_limits,
                rrt_max_time,
                smooth_max_time,
            )

        return _plan_fn

    def solve_arm_ik(
        self,
        target_pose: np.ndarray,
        q0: Iterable[float],
        link_name: str = "gen3_end_effector_link",
        link_offset: np.ndarray | None = None,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.2,
        regularization_weight: float = 0.01,
        max_nfev: int = 100,
    ) -> np.ndarray:
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (4, 4):
            raise ValueError(f"target_pose must have shape (4, 4), got {target_pose.shape}")
        if link_name not in self._link_names:
            raise ValueError(f"Unknown link name: {link_name}")
        if len(self._arm_joint_names) != 7:
            raise ValueError(
                f"Expected 7 arm joints, found {len(self._arm_joint_names)}: {self._arm_joint_names}"
            )
        offset_transform = np.eye(4, dtype=np.float64) if link_offset is None else np.asarray(link_offset, dtype=np.float64)
        if offset_transform.shape != (4, 4):
            raise ValueError(
                f"link_offset must have shape (4, 4), got {offset_transform.shape}"
            )

        arm_indices = [self._joint_index[name] for name in self._arm_joint_names]
        lower_bounds, upper_bounds = self.get_joint_limits()

        initial_full_q = self.visual_configuration
        q_arm0 = np.asarray(tuple(q0), dtype=np.float64)
        if q_arm0.shape != (len(arm_indices),):
            raise ValueError(
                f"q0 must have shape ({len(arm_indices)},), got {q_arm0.shape}"
            )

        reference_pose = self.get_link_pose("gen3_end_effector_link")
        target_link_pose = self.get_link_pose(link_name)
        ee_to_target = np.linalg.inv(reference_pose) @ target_link_pose @ offset_transform
        ee_target_pose = target_pose @ np.linalg.inv(ee_to_target)

        target_position = ee_target_pose[:3, 3]
        target_rotation = ee_target_pose[:3, :3]

        def _residual(q_arm: np.ndarray) -> np.ndarray:
            q_full = initial_full_q.copy()
            q_full[arm_indices] = q_arm
            self._set_full_configuration(q_full)
            current_pose = self.get_link_pose("gen3_end_effector_link")

            position_error = target_position - current_pose[:3, 3]
            rotation_error = Rotation.from_matrix(
                target_rotation @ current_pose[:3, :3].T
            ).as_rotvec()
            return np.concatenate(
                (
                    position_weight * position_error,
                    orientation_weight * rotation_error,
                    regularization_weight * (q_arm - q_arm0),
                )
            )

        result = least_squares(
            _residual,
            x0=q_arm0,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=max_nfev,
        )
        solved_q = initial_full_q.copy()
        solved_q[arm_indices] = result.x
        self._set_full_configuration(solved_q)
        return result.x.copy()

    def solve_ik(
        self,
        target_pose: np.ndarray,
        *,
        q0: Iterable[float] | None = None,
        link_name: str = "gen3_end_effector_link",
        link_offset: np.ndarray | None = None,
        position_weight: float = 1.0,
        orientation_weight: float = 0.2,
        regularization_weight: float = 0.01,
        max_nfev: int = 100,
    ) -> np.ndarray:
        initial_q0 = self.configuration[: len(self._arm_joint_names)] if q0 is None else q0
        return self.solve_arm_ik(
            target_pose,
            initial_q0,
            link_name=link_name,
            link_offset=link_offset,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            regularization_weight=regularization_weight,
            max_nfev=max_nfev,
        )

    def solve(
        self,
        target_pose: np.ndarray,
        q0: Iterable[float] | None = None,
        *,
        target: str = "ee",
        position_weight: float = 1.0,
        orientation_weight: float = 0.2,
        regularization_weight: float = 0.01,
        max_nfev: int = 100,
    ) -> np.ndarray:
        target_map = {
            "ee": "ee",
            "palm": "palm",
            "camera": "camera",
        }
        if target not in target_map:
            raise ValueError(
                f"Unknown target '{target}'. Expected one of: {', '.join(target_map)}"
            )
        return self.solve_ik(
            target_pose,
            q0=q0,
            link_name=target_map[target],
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            regularization_weight=regularization_weight,
            max_nfev=max_nfev,
        )
