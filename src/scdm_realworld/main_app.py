from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import interp1d
import viser
import yaml

from rs415.shm_io import RS415SharedMemoryReader
from scdmv2.comm.client import sample_grasps
from scdmv2.utils import build_query_grid
from scdmv2.utils.query_grid import _depth_to_points_with_mask
from scdm_realworld.environment import BoxEnvironment
from scdm_realworld.robot_real import RobotReal
from scdm_realworld.runtime_config import ArmApi, HandApi
from scdm_realworld.sam3_client import get_seg_mask
from scdm_realworld.utils.geometry import matrix_to_wxyz, rpy_to_matrix
from scdm_realworld.visualize.camera_view import CameraView
from scdm_realworld.visualize.panels import ArmControlPanel, GraspControlPanel, HandControlPanel, SamControlPanel, StatusPanel
from scdm_realworld.visualize.robot_scene import RobotScene


GOTO_MAX_DISTANCE = 1.0
GOTO_DURATION_S = 3.0
DEFAULT_PCD_COUNT = 3000
QUERY_GRID_PATH = Path("tmp/query_grid.npz")
SCDM_SERVER_URL = "http://127.0.0.1:8001"


@dataclass
class AppConfig:
    urdf: Path
    env: Path
    system_calibration: Path
    arm_api: ArmApi
    hand_api: HandApi
    arm_home_q: np.ndarray | None = None
    hand_home_q: np.ndarray | None = None
    arm_presets: dict[str, np.ndarray] | None = None
    hand_presets: dict[str, np.ndarray] | None = None
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


def _draw_uv_marker(rgb: np.ndarray, u: int, v: int) -> np.ndarray:
    marked = rgb.copy()
    height, width = marked.shape[:2]
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))
    radius = 6
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2
    marked[mask] = np.array([255, 255, 0], dtype=np.uint8)
    return marked


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    if mask_uint8.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {mask_uint8.shape}")
    mask_255 = mask_uint8 * 255
    return np.repeat(mask_255[:, :, None], 3, axis=2)


def _pixel_to_camera_xyz(
    depth_mm: np.ndarray,
    intrinsic: dict[str, object],
    u: int,
    v: int,
) -> np.ndarray:
    height, width = depth_mm.shape
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))
    depth_m = float(depth_mm[v, u]) * 0.001
    if depth_m <= 0.0:
        raise ValueError(f"depth is invalid at uv=({u}, {v})")
    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])
    x = (float(u) - cx) * depth_m / fx
    y = (float(v) - cy) * depth_m / fy
    z = depth_m
    return np.asarray((x, y, z), dtype=np.float64)


def _camera_point_to_world(camera_pose: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    rotation_wc = camera_pose[:3, :3]
    translation_wc = camera_pose[:3, 3]
    return rotation_wc @ np.asarray(point_xyz, dtype=np.float64) + translation_wc


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    rot = np.asarray(transform[:3, :3], dtype=np.float64)
    trans = np.asarray(transform[:3, 3], dtype=np.float64)
    return (pts @ rot.T + trans).astype(np.float32, copy=False)


def _quality_to_rgb(quality: np.ndarray) -> np.ndarray:
    q = np.asarray(quality, dtype=np.float32).reshape(-1)
    if q.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    q_min = float(np.min(q))
    q_max = float(np.max(q))
    q_norm = np.zeros_like(q) if q_max - q_min < 1e-8 else (q - q_min) / (q_max - q_min)
    return np.stack([q_norm, 0.2 + 0.6 * (1.0 - q_norm), 1.0 - q_norm], axis=-1).astype(np.float32, copy=False)


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
        self._latest_cam_ext_rgb: np.ndarray | None = None
        self._latest_cam_ext_depth: np.ndarray | None = None
        self._latest_cam_ext_intrinsics: dict[str, object] | None = None
        self._latest_seg_mask: np.ndarray | None = None
        self._grasp_grid_handle = None
        self._query_grasp_handle = None

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
        self.status_panel = StatusPanel(self.server)
        self.status_text = self.status_panel.status_text
        self.joints_text = self.status_panel.real_joints_text
        self.arm_panel = ArmControlPanel(
            self.server,
            joint_names=self.real_robot.arm_joint_names,
            q_min=self.q_min,
            q_max=self.q_max,
            home_q=self.home_q,
            presets=self.config.arm_presets or {"home": self.home_q},
        )
        self.hand_panel = HandControlPanel(
            self.server,
            joint_names=self.robot_scene.hand_joint_names,
            initial_q=self.current_hand_q,
            presets=self.config.hand_presets or {"home": self.current_hand_q},
        )
        self.sam_panel = SamControlPanel(self.server)
        self.grasp_panel = GraspControlPanel(self.server)

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
        @self.arm_panel.set_desired_button.on_click
        def _(_) -> None:
            preset_name = str(self.arm_panel.preset_dropdown.value)
            if preset_name == "current":
                preset_q = self.current_arm_q
            else:
                preset_q = (self.config.arm_presets or {}).get(preset_name)
            if preset_q is None:
                self.status_text.value = f"Unknown arm preset: {preset_name}"
                return
            self.clear_plan()
            self.arm_panel.set_desired_q(preset_q)
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

        @self.sam_panel.u_slider.on_update
        def _(_) -> None:
            self._update_sam_preview()

        @self.sam_panel.v_slider.on_update
        def _(_) -> None:
            self._update_sam_preview()

        @self.sam_panel.send_button.on_click
        def _(_) -> None:
            if self._latest_cam_ext_rgb is None:
                self.sam_panel.log_text.value = "cam_ext image unavailable"
                return
            u = int(self.sam_panel.u_slider.value)
            v = int(self.sam_panel.v_slider.value)
            try:
                response = get_seg_mask(self._latest_cam_ext_rgb, [u, v])
                mask_raw = response.get("mask")
                if mask_raw is None:
                    raise RuntimeError(f"missing mask in response: {response}")
                mask = np.asarray(mask_raw, dtype=np.uint8)
                self._latest_seg_mask = mask
                self.sam_panel.result_image.image = _mask_to_rgb(mask)
                score = response.get("score")
                if score is None:
                    self.sam_panel.log_text.value = f"sent uv=({u}, {v})"
                else:
                    self.sam_panel.log_text.value = f"sent uv=({u}, {v}), score={float(score):.4f}"
            except Exception as exc:
                self._latest_seg_mask = None
                self.sam_panel.log_text.value = str(exc)

        @self.grasp_panel.visualize_grid_button.on_click
        def _(_) -> None:
            if self._latest_cam_ext_depth is None or self._latest_cam_ext_intrinsics is None:
                self.grasp_panel.log_text.value = "cam_ext depth/intrinsics unavailable"
                return
            if self.base_T_cam_ext is None:
                self.grasp_panel.log_text.value = "cam_ext pose unavailable"
                return
            u = int(self.sam_panel.u_slider.value)
            v = int(self.sam_panel.v_slider.value)
            try:
                point_xyz = _pixel_to_camera_xyz(
                    self._latest_cam_ext_depth,
                    self._latest_cam_ext_intrinsics,
                    u,
                    v,
                )
                self.grasp_panel.point_xyz.value = tuple(float(value) for value in point_xyz)
                center_offset = np.asarray(self.grasp_panel.center_offset.value, dtype=np.float64)
                point_world = _camera_point_to_world(self.base_T_cam_ext, point_xyz)
                center_world = point_world + center_offset
                edge_length = max(float(self.grasp_panel.edge_length.value), 1e-3)
                if self._grasp_grid_handle is not None:
                    self._grasp_grid_handle.remove()
                self._grasp_grid_handle = self.server.scene.add_box(
                    "/grasp/grid",
                    color=(255, 0, 0),
                    dimensions=(edge_length, edge_length, edge_length),
                    wireframe=True,
                    position=tuple(float(value) for value in center_world),
                )
                self.grasp_panel.log_text.value = f"uv=({u}, {v})"
            except Exception as exc:
                self.grasp_panel.log_text.value = str(exc)

        @self.grasp_panel.query_grasp_button.on_click
        def _(_) -> None:
            if self._latest_cam_ext_depth is None or self._latest_cam_ext_intrinsics is None:
                self.grasp_panel.log_text.value = "cam_ext depth/intrinsics unavailable"
                return
            if self.base_T_cam_ext is None:
                self.grasp_panel.log_text.value = "cam_ext pose unavailable"
                return
            if self._latest_seg_mask is None:
                self.grasp_panel.log_text.value = "segmentation mask unavailable"
                return
            u = int(self.sam_panel.u_slider.value)
            v = int(self.sam_panel.v_slider.value)
            try:
                point_xyz = _pixel_to_camera_xyz(
                    self._latest_cam_ext_depth,
                    self._latest_cam_ext_intrinsics,
                    u,
                    v,
                )
                self.grasp_panel.point_xyz.value = tuple(float(value) for value in point_xyz)
                center_offset = np.asarray(self.grasp_panel.center_offset.value, dtype=np.float64)
                point_world = _camera_point_to_world(self.base_T_cam_ext, point_xyz)
                center_world = point_world + center_offset
                edge_length = max(float(self.grasp_panel.edge_length.value), 1e-3)

                world_T_grid = np.eye(4, dtype=np.float32)
                world_T_grid[:3, 3] = center_world.astype(np.float32)
                cam_T_world = np.linalg.inv(self.base_T_cam_ext).astype(np.float32)
                cam_T_grid = cam_T_world @ world_T_grid
                intrinsics = np.asarray(
                    [
                        float(self._latest_cam_ext_intrinsics["fx"]),
                        float(self._latest_cam_ext_intrinsics["fy"]),
                        float(self._latest_cam_ext_intrinsics["cx"]),
                        float(self._latest_cam_ext_intrinsics["cy"]),
                    ],
                    dtype=np.float32,
                )
                query_grid = build_query_grid(
                    self._latest_cam_ext_depth,
                    self._latest_seg_mask.astype(bool),
                    intrinsics,
                    cam_T_grid,
                    grid_edge=edge_length,
                )
                full_points_cam, seg_points_cam = _depth_to_points_with_mask(
                    self._latest_cam_ext_depth,
                    self._latest_seg_mask.astype(bool),
                    intrinsics,
                    depth_scale=1000.0,
                )
                grid_T_cam = np.linalg.inv(cam_T_grid).astype(np.float32)
                full_points_grid = _transform_points(grid_T_cam, full_points_cam)
                seg_points_grid = _transform_points(grid_T_cam, seg_points_cam)

                result = sample_grasps(
                    query_grid,
                    server_url=SCDM_SERVER_URL,
                    row={
                        "grid_pose_wrt_cam": cam_T_grid,
                        "grid_pose_wrt_world": world_T_grid,
                        "intrinsic": intrinsics,
                        "grid_edge": np.asarray(edge_length, dtype=np.float32),
                    },
                    num_samples=32,
                    num_inference_steps=32,
                    clip=True,
                    return_trajectory=False,
                    timeout_s=120.0,
                )
                pose9d = np.asarray(result.get("pose9d", []), dtype=np.float32)
                quality = np.asarray(result.get("quality", []), dtype=np.float32)
                if pose9d.ndim != 2 or pose9d.shape[1] < 3:
                    raise RuntimeError(f"unexpected pose9d shape: {pose9d.shape}")
                palm_points_grid = np.asarray(pose9d[:, :3], dtype=np.float32)
                palm_points_world = _transform_points(world_T_grid, pose9d[:, :3])
                q_contact = np.asarray(result.get("q_contact", []), dtype=np.float32)
                q_open = np.asarray(result.get("q_open", []), dtype=np.float32)
                q_squeeze = np.asarray(result.get("q_squeeze", []), dtype=np.float32)
                pose9d_traj = np.asarray(result.get("pose9d_traj", []), dtype=np.float32)
                q_contact_traj = np.asarray(result.get("q_contact_traj", []), dtype=np.float32)
                QUERY_GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    QUERY_GRID_PATH,
                    grid_occupancy=query_grid,
                    grid_pose_wrt_cam=cam_T_grid,
                    grid_pose_wrt_world=world_T_grid,
                    intrinsic=intrinsics,
                    grid_edge=np.asarray(edge_length, dtype=np.float32),
                    full_points_grid=full_points_grid,
                    seg_points_grid=seg_points_grid,
                    sampled_pose9d=pose9d,
                    sampled_palm_points_grid=palm_points_grid,
                    sampled_palm_points_world=palm_points_world,
                    sampled_quality=quality,
                    sampled_q_contact=q_contact,
                    sampled_q_open=q_open,
                    sampled_q_squeeze=q_squeeze,
                    sampled_pose9d_traj=pose9d_traj,
                    sampled_q_contact_traj=q_contact_traj,
                )
                if self._query_grasp_handle is not None:
                    self._query_grasp_handle.remove()
                self._query_grasp_handle = self.server.scene.add_point_cloud(
                    "/grasp/query_points",
                    points=palm_points_world,
                    colors=_quality_to_rgb(quality),
                    point_size=0.012,
                    point_shape="sparkle",
                )
                self.grasp_panel.log_text.value = (
                    f"saved {QUERY_GRID_PATH}, queried {len(palm_points_world)} grasps"
                )
            except Exception as exc:
                self.grasp_panel.log_text.value = str(exc)

        @self.hand_panel.set_desired_button.on_click
        def _(_) -> None:
            preset_name = str(self.hand_panel.preset_dropdown.value)
            if preset_name == "current":
                preset_q = self.current_hand_q
            else:
                preset_q = (self.config.hand_presets or {}).get(preset_name)
            if preset_q is None:
                self.status_text.value = f"Unknown hand preset: {preset_name}"
                return
            self.hand_panel.set_desired_q(preset_q)
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
                desired_q = self.hand_panel.desired_q()
                print(
                    "[main_app] hand goto "
                    f"q_vel={max(float(self.hand_panel.q_vel_gui.value), 1e-3):.3f} "
                    f"desired_q={[round(float(value), 4) for value in desired_q]}",
                    flush=True,
                )
                self.config.hand_api.goto_joints(
                    desired_q.tolist(),
                    target_q_vel=max(float(self.hand_panel.q_vel_gui.value), 1e-3),
                )
                self.current_hand_q = desired_q.copy()
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
            self.status_panel.set_kinova(False, arm_error)
            self.status_text.value = f"Arm error: {exc}"
        else:
            self.status_panel.set_kinova(True, "")

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
        except Exception as exc:
            hand_error = str(exc)
        if hand_ok:
            self.status_panel.set_allegro(True, "")
        else:
            self.status_panel.set_allegro(False, hand_error)
            if not self.status_text.value.startswith(("Plan", "Goto", "Execute", "Arm error")):
                self.status_text.value = f"Hand error: {hand_error}"

        if arm_ok:
            self.robot_scene.set_real_q(self.current_arm_q, self.current_hand_q)
            self.joints_text.value = str([round(value, 4) for value in self.current_arm_q])

        cam_ext_ok, cam_ext_error = self.cam_ext_view.update(self.base_T_cam_ext)
        self.status_panel.set_cam_ext(cam_ext_ok, cam_ext_error)
        self._latest_cam_ext_rgb = self.cam_ext_view.latest_rgb
        self._latest_cam_ext_depth = self.cam_ext_view.latest_depth
        self._latest_cam_ext_intrinsics = self.cam_ext_view.latest_intrinsics
        self._update_sam_preview()

        cam_wrist_pose = None
        if arm_ok:
            try:
                cam_wrist_pose = self.real_robot.get_link_pose("camera") @ self.cam_wrist_correction
            except Exception as exc:
                cam_wrist_pose = None
                self.status_panel.set_cam_wrist(False, str(exc))
        cam_wrist_ok, cam_wrist_error = self.cam_wrist_view.update(cam_wrist_pose)
        self.status_panel.set_cam_wrist(cam_wrist_ok, cam_wrist_error)
        if not self.status_text.value.startswith(("Plan", "Goto", "Execute", "Hand", "Arm error")):
            self.status_text.value = "Running"

    def _sync_desired_view(self) -> None:
        self.robot_scene.set_desired_q(self.arm_panel.desired_q(), self.hand_panel.desired_q())
        self._update_desired_text()

    def _update_sam_preview(self) -> None:
        if self._latest_cam_ext_rgb is None:
            self.sam_panel.image.image = np.zeros((480, 640, 3), dtype=np.uint8)
            return
        self.sam_panel.image.image = _draw_uv_marker(
            self._latest_cam_ext_rgb,
            int(self.sam_panel.u_slider.value),
            int(self.sam_panel.v_slider.value),
        )

    def _update_desired_text(self, q_arm: np.ndarray | None = None) -> None:
        _ = self.arm_panel.desired_q() if q_arm is None else np.asarray(q_arm, dtype=np.float64)

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
