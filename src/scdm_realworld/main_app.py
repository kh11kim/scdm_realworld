from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import interp1d
import viser
import yaml

from rs415.shm_io import RS415SharedMemoryReader
from scdm_realworld.environment import BoxEnvironment
from scdm_realworld.robot_real import RobotReal
from scdm_realworld.runtime_config import ArmApi, HandApi
from scdm_realworld.utils.geometry import matrix_to_wxyz, rpy_to_matrix
from scdm_realworld.visualize.camera_view import CameraView
from scdm_realworld.visualize.panels import ArmControlPanel, HandControlPanel, HealthPanel
from scdm_realworld.visualize.robot_scene import RobotScene


GOTO_MAX_DISTANCE = 1.0
GOTO_DURATION_S = 3.0
DEFAULT_PCD_COUNT = 3000


@dataclass
class AppConfig:
    urdf: Path
    env: Path
    system_calibration: Path
    arm_api: ArmApi
    hand_api: HandApi
    arm_home_q: np.ndarray | None = None
    hand_home_q: np.ndarray | None = None
    host: str = "0.0.0.0"
    port: int = 8080
    scale: float = 1.0
    dt: float = 0.1


def _box_wxyz(rpy) -> tuple[float, float, float, float]:
    return matrix_to_wxyz(rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2])))


def _normalize_to_limits(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    normalized = np.asarray(q, dtype=np.float64).copy()
    two_pi = 2.0 * np.pi
    for index, value in enumerate(normalized):
        if q_min[index] <= value <= q_max[index]:
            continue
        candidates = [value + k * two_pi for k in range(-2, 3)]
        feasible = [candidate for candidate in candidates if q_min[index] <= candidate <= q_max[index]]
        if feasible:
            normalized[index] = min(feasible, key=lambda candidate: abs(candidate - value))
        else:
            normalized[index] = np.clip(value, q_min[index], q_max[index])
    return normalized


def _find_joint_limit_violation(
    q: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> tuple[int, float, float, float] | None:
    for index, value in enumerate(np.asarray(q, dtype=np.float64)):
        if value < q_min[index] or value > q_max[index]:
            return index, float(value), float(q_min[index]), float(q_max[index])
    return None


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML structure: {path}")
    return payload


def _transform_from_xyz_rpy(
    xyz: np.ndarray | list[float] | tuple[float, float, float],
    rpy: np.ndarray | list[float] | tuple[float, float, float],
) -> np.ndarray:
    xyz_arr = np.asarray(xyz, dtype=np.float64)
    rpy_arr = np.asarray(rpy, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_matrix(float(rpy_arr[0]), float(rpy_arr[1]), float(rpy_arr[2]))
    transform[:3, 3] = xyz_arr
    return transform


class MainApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        if not config.urdf.exists():
            raise FileNotFoundError(f"URDF not found: {config.urdf}")

        self.server = viser.ViserServer(host=config.host, port=config.port)
        self.box_env = BoxEnvironment.load(config.env)
        self.real_robot = RobotReal.from_urdf(
            config.urdf,
            get_joints_fn=config.arm_api.get_joints,
            execute_trajectory_fn=config.arm_api.execute_trajectory,
            joint_position_limits=config.arm_api.joint_position_limits,
        )
        self.planning_robot = RobotReal.from_urdf(
            config.urdf,
            get_joints_fn=config.arm_api.get_joints,
            execute_trajectory_fn=config.arm_api.execute_trajectory,
            joint_position_limits=config.arm_api.joint_position_limits,
        )
        self.plan_fn = self.planning_robot.get_plan_fn(self.box_env)
        self.q_min, self.q_max = self.real_robot.get_joint_limits()
        arm_home_q = self.real_robot.configuration.copy() if config.arm_home_q is None else np.asarray(config.arm_home_q, dtype=np.float64)
        self.home_q = _normalize_to_limits(arm_home_q, self.q_min, self.q_max)

        self.robot_scene = RobotScene(
            self.server,
            config.urdf,
            scale=config.scale,
            full_joint_names=self.planning_robot.joint_names,
            arm_joint_names=self.real_robot.arm_joint_names,
            default_full_q=self.planning_robot.visual_configuration,
        )

        self.current_arm_q = self.real_robot.configuration.copy()
        if config.hand_home_q is not None and np.asarray(config.hand_home_q).shape == self.robot_scene.default_hand_q.shape:
            self.current_hand_q = np.asarray(config.hand_home_q, dtype=np.float64).copy()
        else:
            self.current_hand_q = self.robot_scene.default_hand_q
        self.trajectory: np.ndarray | None = None
        self.trajectory_interp = None

        self._init_scene()
        self._init_gui()
        self._init_cameras()
        self._bind_callbacks()
        self.robot_scene.set_real_q(self.current_arm_q, self.current_hand_q)
        self.robot_scene.set_desired_q(self.home_q, self.current_hand_q)

    def _init_scene(self) -> None:
        self.server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy")
        for index, box in enumerate(self.box_env.boxes):
            self.server.scene.add_box(
                f"/env/{box.name}",
                color=((220 - 20 * (index % 4)), 100 + 20 * (index % 5), 180),
                dimensions=(1.0, 1.0, 1.0),
                opacity=0.35,
                position=tuple(box.center.tolist()),
                wxyz=_box_wxyz(box.rpy),
                scale=tuple(box.size.tolist()),
            )

    def _init_gui(self) -> None:
        with self.server.gui.add_folder("Status", expand_by_default=False):
            self.status_text = self.server.gui.add_text("Status", initial_value="Running", disabled=True)
            self.joints_text = self.server.gui.add_text("Real Joints", initial_value="", disabled=True)
            self.desired_text = self.server.gui.add_text(
                "Desired Joints",
                initial_value=str([round(value, 4) for value in self.home_q]),
                disabled=True,
            )
        self.arm_panel = ArmControlPanel(
            self.server,
            joint_names=self.real_robot.arm_joint_names,
            q_min=self.q_min,
            q_max=self.q_max,
            home_q=self.home_q,
        )
        self.hand_panel = HandControlPanel(
            self.server,
            joint_names=self.robot_scene.hand_joint_names,
            initial_q=self.current_hand_q,
        )
        self.health_panel = HealthPanel(self.server)

    def _init_cameras(self) -> None:
        system_calibration = _load_yaml(self.config.system_calibration)
        self.base_T_cam_ext = None
        self.cam_wrist_correction = np.eye(4, dtype=np.float64)
        base_T_cam_ext_raw = system_calibration.get("base_T_cam_ext")
        if isinstance(base_T_cam_ext_raw, dict):
            xyz = np.asarray(base_T_cam_ext_raw.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
            rpy = np.asarray(base_T_cam_ext_raw.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float64)
            self.base_T_cam_ext = np.eye(4, dtype=np.float64)
            self.base_T_cam_ext[:3, :3] = rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            self.base_T_cam_ext[:3, 3] = xyz
        cam_wrist_correction_raw = system_calibration.get("cam_wrist_correction")
        if isinstance(cam_wrist_correction_raw, dict):
            self.cam_wrist_correction = _transform_from_xyz_rpy(
                cam_wrist_correction_raw.get("xyz", [0.0, 0.0, 0.0]),
                cam_wrist_correction_raw.get("rpy", [0.0, 0.0, 0.0]),
            )

        ext_reader = None
        wrist_reader = None
        ext_serial = system_calibration.get("cam_ext_serial")
        wrist_serial = system_calibration.get("cam_wrist_serial")
        if isinstance(ext_serial, str) and ext_serial and self.base_T_cam_ext is not None:
            try:
                ext_reader = RS415SharedMemoryReader(ext_serial)
            except Exception:
                ext_reader = None
        if isinstance(wrist_serial, str) and wrist_serial:
            try:
                wrist_reader = RS415SharedMemoryReader(wrist_serial)
            except Exception:
                wrist_reader = None

        self.cam_ext_view = CameraView(
            self.server,
            label="cam_ext",
            reader=ext_reader,
            frame_path="/cam_ext",
            pcd_path="/pcd/cam_ext",
            color=(40, 160, 255),
            default_pcd_count=DEFAULT_PCD_COUNT,
        )
        self.cam_wrist_view = CameraView(
            self.server,
            label="cam_wrist",
            reader=wrist_reader,
            frame_path="/cam_wrist",
            pcd_path="/pcd/cam_wrist",
            color=(255, 140, 40),
            default_pcd_count=DEFAULT_PCD_COUNT,
        )

    def _bind_callbacks(self) -> None:
        @self.arm_panel.home_button.on_click
        def _(_) -> None:
            self.clear_plan()
            self.arm_panel.set_desired_q(self.home_q)
            self._sync_desired_view()

        @self.arm_panel.current_button.on_click
        def _(_) -> None:
            self.clear_plan()
            self.arm_panel.set_desired_q(self.current_arm_q)
            self._sync_desired_view()

        for slider in self.arm_panel.sliders:
            @slider.on_update
            def _(_) -> None:
                self.clear_plan()
                self._sync_desired_view()

        for slider in self.hand_panel.sliders:
            @slider.on_update
            def _(_) -> None:
                self._sync_desired_view()

        @self.arm_panel.plan_button.on_click
        def _(_) -> None:
            try:
                q0 = self.current_arm_q.copy()
                qg = self.arm_panel.desired_q()
                validation_error = self._validate_plan_endpoints(q0, qg)
                if validation_error is not None:
                    self.arm_panel.plan_info_text.value = validation_error
                    self.status_text.value = "Plan failed"
                    return
                self.status_text.value = "Planning..."
                trajectory = self.plan_fn(q0, qg, rrt_max_time=2.0, smooth_max_time=1.0)
                if trajectory is None:
                    self.arm_panel.plan_info_text.value = "Plan failed"
                    self.status_text.value = "Plan failed"
                    return
                self.trajectory = np.asarray(trajectory, dtype=np.float64)
                self.trajectory_interp = interp1d(
                    np.linspace(0.0, 1.0, len(self.trajectory)),
                    self.trajectory,
                    axis=0,
                    kind="linear",
                )
                self.arm_panel.plan_info_text.value = f"{len(self.trajectory)} waypoints"
                self.arm_panel.traj_slider.value = 0.0
                self.robot_scene.set_desired_q(self.trajectory[0], self.hand_panel.desired_q())
                self._update_desired_text()
                self.status_text.value = "Plan ready"
            except Exception as exc:
                self.arm_panel.plan_info_text.value = f"Plan error: {exc}"
                self.status_text.value = f"Error: {exc}"

        @self.arm_panel.traj_slider.on_update
        def _(_) -> None:
            if self.trajectory_interp is None:
                return
            q = np.asarray(self.trajectory_interp(self.arm_panel.traj_slider.value), dtype=np.float64)
            self.robot_scene.set_desired_q(q, self.hand_panel.desired_q())
            self._update_desired_text(q)

        @self.arm_panel.execute_button.on_click
        def _(_) -> None:
            if self.trajectory is None:
                self.status_text.value = "No trajectory"
                return
            try:
                self._execute_trajectory(
                    self.trajectory,
                    total_time_s=float(self.arm_panel.execution_time_gui.value),
                )
            except Exception as exc:
                self.status_text.value = f"Execute error: {exc}"
            finally:
                self.clear_plan()

        @self.arm_panel.goto_button.on_click
        def _(_) -> None:
            try:
                q0 = self.current_arm_q.copy()
                qg = self.arm_panel.desired_q()
                validation_error = self._validate_plan_endpoints(q0, qg)
                if validation_error is not None:
                    self.arm_panel.plan_info_text.value = validation_error
                    self.status_text.value = "Goto failed"
                    return
                goto_distance = float(np.linalg.norm(qg - q0))
                if goto_distance > GOTO_MAX_DISTANCE:
                    self.arm_panel.plan_info_text.value = (
                        f"Goto too far: L2={goto_distance:.4f} > {GOTO_MAX_DISTANCE:.4f}"
                    )
                    self.status_text.value = "Goto failed"
                    return
                trajectory = self.plan_fn(q0, qg, rrt_max_time=2.0, smooth_max_time=1.0)
                if trajectory is None:
                    self.arm_panel.plan_info_text.value = "Goto plan failed"
                    self.status_text.value = "Goto failed"
                    return
                self._execute_trajectory(np.asarray(trajectory, dtype=np.float64), total_time_s=GOTO_DURATION_S)
            except Exception as exc:
                self.status_text.value = f"Goto error: {exc}"
            finally:
                self.clear_plan()

        @self.hand_panel.goto_button.on_click
        def _(_) -> None:
            try:
                self.status_text.value = "Hand goto..."
                self.config.hand_api.goto_joints(
                    self.hand_panel.desired_q().tolist(),
                    max(float(self.hand_panel.q_vel_gui.value), 1e-3),
                )
                self.current_hand_q = self.hand_panel.desired_q().copy()
                self.robot_scene.set_desired_q(self.arm_panel.desired_q(), self.current_hand_q)
                self.status_text.value = "Hand goto sent"
            except Exception as exc:
                self.status_text.value = f"Hand goto error: {exc}"

    def run(self) -> int:
        print(f"URDF: {self.config.urdf}")
        print(f"Environment: {self.config.env}")
        print(f"System calibration: {self.config.system_calibration}")
        print("Open the viser URL shown above to inspect the live state.")
        try:
            while True:
                self._poll_once()
                time.sleep(self.config.dt)
        except KeyboardInterrupt:
            return 0
        finally:
            self.cam_ext_view.close()
            self.cam_wrist_view.close()
            try:
                self.server.stop()
            except RuntimeError:
                pass

    def _poll_once(self) -> None:
        arm_ok = False
        arm_error = ""
        try:
            self.current_arm_q = self.real_robot.sync_from_real()
            arm_ok = True
        except Exception as exc:
            arm_error = str(exc)
            self.health_panel.set_kinova(False, arm_error)
            self.status_text.value = f"Arm error: {exc}"
        else:
            self.health_panel.set_kinova(True, "")

        hand_ok = False
        hand_error = ""
        try:
            hand_q = np.asarray(self.config.hand_api.get_joints(), dtype=np.float64)
            if hand_q.shape == self.current_hand_q.shape:
                self.current_hand_q = hand_q
                hand_ok = True
            else:
                hand_error = (
                    f"unexpected hand shape: {hand_q.shape}, expected {self.current_hand_q.shape}"
                )
        except Exception:
            hand_error = "hand state unavailable"
        if hand_ok:
            self.health_panel.set_allegro(True, "")
        else:
            self.health_panel.set_allegro(False, hand_error)

        if arm_ok:
            self.robot_scene.set_real_q(self.current_arm_q, self.current_hand_q)
            self.joints_text.value = str([round(value, 4) for value in self.current_arm_q])

        cam_ext_ok, cam_ext_error = self.cam_ext_view.update(self.base_T_cam_ext)
        self.health_panel.set_cam_ext(cam_ext_ok, cam_ext_error)

        cam_wrist_pose = None
        if arm_ok:
            try:
                cam_wrist_pose = self.real_robot.get_link_pose("camera") @ self.cam_wrist_correction
            except Exception as exc:
                cam_wrist_pose = None
                self.health_panel.set_cam_wrist(False, str(exc))
        cam_wrist_ok, cam_wrist_error = self.cam_wrist_view.update(cam_wrist_pose)
        self.health_panel.set_cam_wrist(cam_wrist_ok, cam_wrist_error)
        if not self.status_text.value.startswith(("Plan", "Goto", "Execute", "Hand", "Arm error")):
            self.status_text.value = "Running"

    def _sync_desired_view(self) -> None:
        self.robot_scene.set_desired_q(self.arm_panel.desired_q(), self.hand_panel.desired_q())
        self._update_desired_text()

    def _update_desired_text(self, q_arm: np.ndarray | None = None) -> None:
        q = self.arm_panel.desired_q() if q_arm is None else np.asarray(q_arm, dtype=np.float64)
        self.desired_text.value = str([round(value, 4) for value in q])

    def clear_plan(self) -> None:
        self.trajectory = None
        self.trajectory_interp = None
        self.arm_panel.clear_plan()

    def _validate_plan_endpoints(self, q0: np.ndarray, qg: np.ndarray) -> str | None:
        q0_violation = _find_joint_limit_violation(q0, self.q_min, self.q_max)
        if q0_violation is not None:
            joint_index, value, lower, upper = q0_violation
            return (
                f"Start out of limits: joint {joint_index + 1} "
                f"value={value:.4f}, limits=[{lower:.4f}, {upper:.4f}]"
            )
        qg_violation = _find_joint_limit_violation(qg, self.q_min, self.q_max)
        if qg_violation is not None:
            joint_index, value, lower, upper = qg_violation
            return (
                f"Goal out of limits: joint {joint_index + 1} "
                f"value={value:.4f}, limits=[{lower:.4f}, {upper:.4f}]"
            )
        return None

    def _execute_trajectory(self, trajectory: np.ndarray, *, total_time_s: float) -> None:
        self.status_text.value = "Executing..."
        self.real_robot.execute_trajectory(
            trajectory.tolist(),
            total_time_s=total_time_s,
            wait=True,
        )
        self.current_arm_q = self.real_robot.sync_from_real()
        try:
            hand_q = np.asarray(self.config.hand_api.get_joints(), dtype=np.float64)
            if hand_q.shape == self.current_hand_q.shape:
                self.current_hand_q = hand_q
        except Exception:
            pass
        self.robot_scene.set_real_q(self.current_arm_q, self.current_hand_q)
        self.arm_panel.set_desired_q(self.current_arm_q)
        self.hand_panel.set_desired_q(self.current_hand_q)
        self.robot_scene.set_desired_q(self.current_arm_q, self.current_hand_q)
        self._update_desired_text(self.current_arm_q)
        self.status_text.value = "Executed"
