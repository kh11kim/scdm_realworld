from __future__ import annotations

from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf


class RobotScene:
    def __init__(
        self,
        server: viser.ViserServer,
        urdf_path: Path,
        *,
        scale: float,
        full_joint_names: tuple[str, ...],
        arm_joint_names: tuple[str, ...],
        default_full_q: np.ndarray,
    ) -> None:
        self._server = server
        self._full_joint_names = full_joint_names
        self._arm_joint_names = arm_joint_names
        self._arm_count = len(arm_joint_names)
        self._hand_joint_names = tuple(
            name for name in full_joint_names if not name.startswith("gen3_joint_")
        )
        self._default_full_q = np.asarray(default_full_q, dtype=np.float64).copy()
        self._real_q = self._default_full_q.copy()
        self._desired_q = self._default_full_q.copy()

        self._real = ViserUrdf(
            server,
            urdf_or_path=urdf_path,
            scale=scale,
            root_node_name="/real_robot",
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._desired = ViserUrdf(
            server,
            urdf_or_path=urdf_path,
            scale=scale,
            root_node_name="/desired_robot",
            mesh_color_override=(0.2, 0.6, 1.0, 0.22),
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._real.update_cfg(self._real_q)
        self._desired.update_cfg(self._desired_q)

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return self._arm_joint_names

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return self._hand_joint_names

    @property
    def default_hand_q(self) -> np.ndarray:
        return self._default_full_q[self._arm_count :].copy()

    @property
    def real_q(self) -> np.ndarray:
        return self._real_q.copy()

    @property
    def desired_q(self) -> np.ndarray:
        return self._desired_q.copy()

    def set_real_q(self, arm_q: np.ndarray, hand_q: np.ndarray) -> None:
        self._real_q = self._compose_full_q(arm_q, hand_q)
        self._real.update_cfg(self._real_q)

    def set_desired_q(self, arm_q: np.ndarray, hand_q: np.ndarray) -> None:
        self._desired_q = self._compose_full_q(arm_q, hand_q)
        self._desired.update_cfg(self._desired_q)

    def set_desired_arm(self, arm_q: np.ndarray) -> None:
        self.set_desired_q(arm_q, self._desired_q[self._arm_count :])

    def set_desired_hand(self, hand_q: np.ndarray) -> None:
        self.set_desired_q(self._desired_q[: self._arm_count], hand_q)

    def _compose_full_q(self, arm_q: np.ndarray, hand_q: np.ndarray) -> np.ndarray:
        arm = np.asarray(arm_q, dtype=np.float64)
        hand = np.asarray(hand_q, dtype=np.float64)
        if arm.shape != (self._arm_count,):
            raise ValueError(f"arm_q must have shape ({self._arm_count},), got {arm.shape}")
        if hand.shape != (len(self._hand_joint_names),):
            raise ValueError(
                f"hand_q must have shape ({len(self._hand_joint_names)},), got {hand.shape}"
            )
        return np.concatenate((arm, hand))
