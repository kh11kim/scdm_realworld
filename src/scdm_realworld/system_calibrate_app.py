from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import tyro
import viser
import yaml
from viser.extras import ViserUrdf

from rs415.shm_io import RS415SharedMemoryReader
from rs415.calibration import (
    CheckerboardSpec,
    IntrinsicInfo,
    Pose,
    align_checkerboard_pose_to_aruco,
    create_aruco_detector,
    detect_aruco_markers,
    detect_checkerboard,
    rotation_matrix_to_rpy,
    select_aruco_pose,
    visualize_checkerboard_detection,
)
from scdm_realworld.robot_real import RobotReal
from scdm_realworld.runtime_config import (
    DEFAULT_APP_CONFIG_PATH,
    load_runtime_config,
    resolve_arm_api,
    resolve_hand_api,
    resolve_robot_urdf,
)
from scdm_realworld.utils.geometry import matrix_to_wxyz, project_depth_to_world, rpy_to_matrix


CHECKERBOARD = CheckerboardSpec(corners_x=5, corners_y=4, square_size_m=0.03)
ARUCO_DICTIONARY_NAME = "DICT_4X4_50"
ARUCO_MARKER_LENGTH_M = 0.05
PCD_COUNT = 3000


@dataclass
class Args:
    app_config: Path = DEFAULT_APP_CONFIG_PATH
    config: Path = Path("assets/system_calibration.yaml")
    host: str = "0.0.0.0"
    port: int = 8080
    dt: float = 0.1


def _load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid system calibration YAML: {path}")
    return payload


def _intrinsic_from_meta(meta: dict[str, object]) -> IntrinsicInfo:
    intr = meta.get("intrinsics")
    if not isinstance(intr, dict):
        raise ValueError("shared memory metadata is missing intrinsics")
    coeffs_raw = intr.get("coeffs", [])
    coeffs = tuple(float(value) for value in coeffs_raw)
    return IntrinsicInfo(
        width=int(intr["width"]),
        height=int(intr["height"]),
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        model="unknown",
        coeffs=coeffs,
    )


def _draw_pose_text(image_bgr: np.ndarray, world_pose: Pose | None) -> np.ndarray:
    preview = image_bgr.copy()
    if world_pose is None:
        cv2.putText(
            preview,
            "final pose: unavailable",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    rotation, _ = cv2.Rodrigues(world_pose.rvec)
    roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
    translation = world_pose.tvec.reshape(3)
    cv2.putText(
        preview,
        "final pose",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"xyz = [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"rpy = [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]",
        (12, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def _pose_to_matrix(pose: Pose) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose.tvec.reshape(3)
    return transform


def _update_frame(handle: viser.FrameHandle, transform: np.ndarray) -> None:
    handle.position = tuple(transform[:3, 3].tolist())
    handle.wxyz = matrix_to_wxyz(transform[:3, :3])


def _camera_frustum_params(intrinsic: IntrinsicInfo) -> tuple[float, float]:
    aspect = float(intrinsic.width) / float(intrinsic.height)
    fov = 2.0 * np.arctan2(float(intrinsic.height) / 2.0, float(intrinsic.fy))
    return float(fov), float(aspect)


def _intrinsic_payload(intrinsic: IntrinsicInfo) -> dict[str, float]:
    return {
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "cx": float(intrinsic.cx),
        "cy": float(intrinsic.cy),
    }


def _pcd_step_from_count(depth_mm: np.ndarray, count: int) -> int:
    valid_count = int(np.count_nonzero(depth_mm > 0))
    if valid_count <= 0:
        return 1
    return max(1, valid_count // count)


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


def run(args: Args) -> int:
    runtime_config = load_runtime_config(args.app_config)
    urdf_path = resolve_robot_urdf(runtime_config).resolve()
    arm_api = resolve_arm_api(runtime_config)
    hand_api = resolve_hand_api(runtime_config)
    config = _load_config(args.config)
    server = viser.ViserServer(host=args.host, port=args.port)
    aruco_detector = create_aruco_detector(ARUCO_DICTIONARY_NAME)
    robot = RobotReal.from_urdf(
        urdf_path,
        get_joints_fn=arm_api.get_joints,
        execute_trajectory_fn=arm_api.execute_trajectory,
        joint_position_limits=arm_api.joint_position_limits,
    )
    arm_dof = len(robot.arm_joint_names)
    default_full_q = robot.visual_configuration.copy()
    current_hand_q = default_full_q[arm_dof:].copy()
    robot_vis = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        scale=1.0,
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=False,
    )

    selected_role = {"value": "cam_ext"}
    reader_states: dict[str, RS415SharedMemoryReader | None] = {
        "cam_ext": None,
        "cam_wrist": None,
    }
    attached_serials = {"cam_ext": "", "cam_wrist": ""}
    estimated_board_poses: dict[str, Pose | None] = {"cam_ext": None, "cam_wrist": None}
    frustum_handles: dict[str, object | None] = {"cam_ext": None, "cam_wrist": None}
    point_cloud_handles: dict[str, object | None] = {"cam_ext": None, "cam_wrist": None}
    latest_bundles: dict[str, object | None] = {"cam_ext": None, "cam_wrist": None}
    latest_intrinsics: dict[str, IntrinsicInfo | None] = {"cam_ext": None, "cam_wrist": None}
    latest_previews: dict[str, np.ndarray | None] = {"cam_ext": None, "cam_wrist": None}
    base_T_cam_ext_state: dict[str, np.ndarray | None] = {"value": None}
    wrist_correction_cfg = config.get("cam_wrist_correction", {})
    if not isinstance(wrist_correction_cfg, dict):
        wrist_correction_cfg = {}
    wrist_correction_xyz = np.asarray(
        wrist_correction_cfg.get("xyz", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    wrist_correction_rpy = np.asarray(
        wrist_correction_cfg.get("rpy", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )

    server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy")
    ee_frame = server.scene.add_frame("/ee", axes_length=0.08, axes_radius=0.004)
    wrist_cam_frame = server.scene.add_frame("/cam_wrist", axes_length=0.08, axes_radius=0.004)
    ext_cam_frame = server.scene.add_frame("/cam_ext", axes_length=0.08, axes_radius=0.004)
    board_frame = server.scene.add_frame("/board", axes_length=0.1, axes_radius=0.005)

    with server.gui.add_folder("System Calibrate"):
        role_gui = server.gui.add_dropdown(
            "Camera",
            ("cam_ext", "cam_wrist"),
            initial_value="cam_ext",
        )
        cam_ext_serial_gui = server.gui.add_text(
            "cam_ext serial",
            initial_value=str(config.get("cam_ext_serial", "")),
        )
        cam_wrist_serial_gui = server.gui.add_text(
            "cam_wrist serial",
            initial_value=str(config.get("cam_wrist_serial", "")),
        )
        cam_ext_connected_gui = server.gui.add_checkbox(
            "cam_ext connected",
            initial_value=False,
            disabled=True,
        )
        cam_wrist_connected_gui = server.gui.add_checkbox(
            "cam_wrist connected",
            initial_value=False,
            disabled=True,
        )
        cam_ext_pcd_gui = server.gui.add_checkbox("cam_ext pcd", initial_value=False)
        cam_ext_pcd_count_gui = server.gui.add_number("cam_ext pcd_count", initial_value=PCD_COUNT, step=500)
        cam_wrist_pcd_gui = server.gui.add_checkbox("cam_wrist pcd", initial_value=False)
        cam_wrist_pcd_count_gui = server.gui.add_number(
            "cam_wrist pcd_count",
            initial_value=PCD_COUNT,
            step=500,
        )
        kinova_connected_gui = server.gui.add_checkbox(
            "kinova connected",
            initial_value=False,
            disabled=True,
        )
        allegro_connected_gui = server.gui.add_checkbox(
            "allegro connected",
            initial_value=False,
            disabled=True,
        )
        cam_wrist_tx_gui = server.gui.add_number("cam_wrist tx", initial_value=float(wrist_correction_xyz[0]), step=0.001)
        cam_wrist_ty_gui = server.gui.add_number("cam_wrist ty", initial_value=float(wrist_correction_xyz[1]), step=0.001)
        cam_wrist_tz_gui = server.gui.add_number("cam_wrist tz", initial_value=float(wrist_correction_xyz[2]), step=0.001)
        cam_wrist_rr_gui = server.gui.add_number("cam_wrist roll", initial_value=float(wrist_correction_rpy[0]), step=0.01)
        cam_wrist_rp_gui = server.gui.add_number("cam_wrist pitch", initial_value=float(wrist_correction_rpy[1]), step=0.01)
        cam_wrist_ry_gui = server.gui.add_number("cam_wrist yaw", initial_value=float(wrist_correction_rpy[2]), step=0.01)
        export_button = server.gui.add_button("Export")
        status_gui = server.gui.add_text("Status", initial_value="Idle", disabled=True)
        preview_gui = server.gui.add_image(
            np.zeros((480, 640, 3), dtype=np.uint8),
            label="Camera Preview",
        )

    def _serial_for_role(role: str) -> str:
        if role == "cam_ext":
            return str(cam_ext_serial_gui.value).strip()
        return str(cam_wrist_serial_gui.value).strip()

    def _connected_gui_for_role(role: str):
        if role == "cam_ext":
            return cam_ext_connected_gui
        return cam_wrist_connected_gui

    def _reset_reader(role: str) -> None:
        reader = reader_states[role]
        if reader is not None:
            reader.close()
            reader_states[role] = None
        attached_serials[role] = ""
        latest_bundles[role] = None
        latest_intrinsics[role] = None

    def _ensure_reader(role: str) -> RS415SharedMemoryReader | None:
        serial = _serial_for_role(role)
        if not serial:
            _reset_reader(role)
            return None
        if attached_serials[role] != serial:
            _reset_reader(role)
        if reader_states[role] is None:
            reader_states[role] = RS415SharedMemoryReader(serial=serial)
            attached_serials[role] = serial
        return reader_states[role]

    @role_gui.on_update
    def _(_) -> None:
        selected_role["value"] = str(role_gui.value)
        preview = latest_previews[selected_role["value"]]
        if preview is not None:
            preview_gui.image = preview
            status_gui.value = f"{selected_role['value']}: connected"
            return
        preview_gui.image = np.zeros((480, 640, 3), dtype=np.uint8)
        if not _serial_for_role(selected_role["value"]):
            status_gui.value = f"{selected_role['value']}: serial not set"
        else:
            status_gui.value = f"{selected_role['value']}: waiting for frame"

    @cam_ext_serial_gui.on_update
    def _(_) -> None:
        _reset_reader("cam_ext")
        if selected_role["value"] == "cam_ext" and not _serial_for_role("cam_ext"):
            status_gui.value = "cam_ext: serial not set"

    @cam_wrist_serial_gui.on_update
    def _(_) -> None:
        _reset_reader("cam_wrist")
        if selected_role["value"] == "cam_wrist" and not _serial_for_role("cam_wrist"):
            status_gui.value = "cam_wrist: serial not set"

    @export_button.on_click
    def _(_) -> None:
        output_path = args.config
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cam_ext_serial": str(cam_ext_serial_gui.value).strip(),
            "cam_wrist_serial": str(cam_wrist_serial_gui.value).strip(),
        }
        if base_T_cam_ext_state["value"] is not None:
            rotation = base_T_cam_ext_state["value"][:3, :3]
            translation = base_T_cam_ext_state["value"][:3, 3]
            roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
            payload["base_T_cam_ext"] = {
                "xyz": [float(value) for value in translation.tolist()],
                "rpy": [float(roll), float(pitch), float(yaw)],
            }
        payload["cam_wrist_correction"] = {
            "xyz": [
                float(cam_wrist_tx_gui.value),
                float(cam_wrist_ty_gui.value),
                float(cam_wrist_tz_gui.value),
            ],
            "rpy": [
                float(cam_wrist_rr_gui.value),
                float(cam_wrist_rp_gui.value),
                float(cam_wrist_ry_gui.value),
            ],
        }
        output_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        status_gui.value = f"Exported {output_path}"

    print(f"System calibration config: {args.config.resolve()}")
    print("Open the viser URL shown above.")

    try:
        while True:
            q_current: np.ndarray | None = None
            for role in ("cam_ext", "cam_wrist"):
                connected_gui = _connected_gui_for_role(role)
                try:
                    reader = _ensure_reader(role)
                    if reader is None:
                        connected_gui.value = False
                        estimated_board_poses[role] = None
                        continue
                    connected_gui.value = True
                    bundle = reader.read(copy=True)
                except Exception as exc:
                    connected_gui.value = False
                    estimated_board_poses[role] = None
                    latest_previews[role] = None
                    _reset_reader(role)
                    if frustum_handles[role] is not None:
                        frustum_handles[role].remove()
                        frustum_handles[role] = None
                    if point_cloud_handles[role] is not None:
                        point_cloud_handles[role].remove()
                        point_cloud_handles[role] = None
                    if role == selected_role["value"]:
                        preview_gui.image = np.zeros((480, 640, 3), dtype=np.uint8)
                        status_gui.value = f"{role}: {exc}"
                    continue

                try:
                    intrinsic = _intrinsic_from_meta(bundle.meta)
                    latest_bundles[role] = bundle
                    latest_intrinsics[role] = intrinsic
                    latest_previews[role] = bundle.rgb
                    if role == selected_role["value"]:
                        preview_gui.image = bundle.rgb
                        status_gui.value = f"{selected_role['value']}: connected"
                    frame_bgr = cv2.cvtColor(bundle.rgb, cv2.COLOR_RGB2BGR)
                    checkerboard_detection = detect_checkerboard(
                        frame_bgr=frame_bgr,
                        checkerboard=CHECKERBOARD,
                        intrinsic=intrinsic,
                    )
                    aruco_detection = detect_aruco_markers(
                        frame_bgr=frame_bgr,
                        detector=aruco_detector,
                        intrinsic=intrinsic,
                        marker_length_m=ARUCO_MARKER_LENGTH_M,
                    )
                    world_pose = None
                    if (
                        checkerboard_detection is not None
                        and checkerboard_detection.rvec is not None
                        and checkerboard_detection.tvec is not None
                    ):
                        checkerboard_pose = Pose(
                            rvec=checkerboard_detection.rvec,
                            tvec=checkerboard_detection.tvec,
                        )
                        marker_pose = select_aruco_pose(aruco_detection)
                        world_pose = align_checkerboard_pose_to_aruco(
                            checkerboard_pose=checkerboard_pose,
                            aruco_pose=marker_pose,
                        )
                    estimated_board_poses[role] = world_pose

                    preview_bgr = visualize_checkerboard_detection(
                        frame_bgr=frame_bgr,
                        checkerboard=CHECKERBOARD,
                        detection=checkerboard_detection,
                        intrinsic=intrinsic,
                        aruco_detection=aruco_detection,
                        world_pose=world_pose,
                    )
                    preview_bgr = _draw_pose_text(preview_bgr, world_pose)
                    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
                    latest_previews[role] = preview_rgb

                    if role == selected_role["value"]:
                        preview_gui.image = preview_rgb
                        status_gui.value = f"{selected_role['value']}: connected"
                except Exception as exc:
                    estimated_board_poses[role] = None
                    latest_previews[role] = bundle.rgb
                    if role == selected_role["value"]:
                        preview_gui.image = bundle.rgb
                        status_gui.value = f"{role} processing: {exc}"

            try:
                q_current = np.asarray(arm_api.get_joints(), dtype=np.float64)
                kinova_connected_gui.value = True
                robot.set_joint_positions(q=q_current)
                base_T_ee = robot.get_link_pose("ee")
                base_T_cam_wrist_raw = robot.get_link_pose("camera")
                cam_wrist_correction = _transform_from_xyz_rpy(
                    [
                        cam_wrist_tx_gui.value,
                        cam_wrist_ty_gui.value,
                        cam_wrist_tz_gui.value,
                    ],
                    [
                        cam_wrist_rr_gui.value,
                        cam_wrist_rp_gui.value,
                        cam_wrist_ry_gui.value,
                    ],
                )
                base_T_cam_wrist = base_T_cam_wrist_raw @ cam_wrist_correction
                _update_frame(ee_frame, base_T_ee)
                _update_frame(wrist_cam_frame, base_T_cam_wrist)
            except Exception:
                kinova_connected_gui.value = False
                q_current = None

            try:
                hand_q = np.asarray(hand_api.get_joints(), dtype=np.float64)
                if hand_q.shape == current_hand_q.shape:
                    current_hand_q = hand_q
                    allegro_connected_gui.value = True
                else:
                    allegro_connected_gui.value = False
            except Exception:
                allegro_connected_gui.value = False

            full_q = robot.visual_configuration.copy()
            if current_hand_q.shape == full_q[arm_dof:].shape:
                full_q[arm_dof:] = current_hand_q
            robot_vis.update_cfg(full_q)

            if reader_states[selected_role["value"]] is None:
                preview_gui.image = np.zeros((480, 640, 3), dtype=np.uint8)
                status_gui.value = f"{selected_role['value']}: serial not set"

            if q_current is not None and estimated_board_poses["cam_wrist"] is not None:
                cam_wrist_T_board = _pose_to_matrix(estimated_board_poses["cam_wrist"])
                base_T_board = base_T_cam_wrist @ cam_wrist_T_board
                _update_frame(board_frame, base_T_board)
                if estimated_board_poses["cam_ext"] is not None:
                    cam_ext_T_board = _pose_to_matrix(estimated_board_poses["cam_ext"])
                    base_T_cam_ext = base_T_board @ np.linalg.inv(cam_ext_T_board)
                    base_T_cam_ext_state["value"] = base_T_cam_ext
                    _update_frame(ext_cam_frame, base_T_cam_ext)
                else:
                    base_T_cam_ext_state["value"] = None
            else:
                base_T_cam_ext_state["value"] = None

            pcd_enabled = {
                "cam_ext": bool(cam_ext_pcd_gui.value),
                "cam_wrist": bool(cam_wrist_pcd_gui.value),
            }
            pcd_count = {
                "cam_ext": max(1, int(cam_ext_pcd_count_gui.value)),
                "cam_wrist": max(1, int(cam_wrist_pcd_count_gui.value)),
            }
            camera_world_transforms: dict[str, np.ndarray | None] = {
                "cam_ext": base_T_cam_ext_state["value"],
                "cam_wrist": base_T_cam_wrist if q_current is not None else None,
            }
            for role in ("cam_ext", "cam_wrist"):
                intrinsic = latest_intrinsics[role]
                preview_rgb = latest_previews[role]
                camera_transform = camera_world_transforms[role]
                if intrinsic is None or preview_rgb is None or camera_transform is None:
                    if frustum_handles[role] is not None:
                        frustum_handles[role].remove()
                        frustum_handles[role] = None
                    continue
                fov, aspect = _camera_frustum_params(intrinsic)
                if frustum_handles[role] is not None:
                    frustum_handles[role].remove()
                frustum_handles[role] = server.scene.add_camera_frustum(
                    f"/{role}/frustum",
                    fov=fov,
                    aspect=aspect,
                    scale=0.1,
                    color=(40, 160, 255) if role == "cam_ext" else (255, 140, 40),
                    image=preview_rgb,
                    position=(0.0, 0.0, 0.0),
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                )

            for role in ("cam_ext", "cam_wrist"):
                if not pcd_enabled[role]:
                    if point_cloud_handles[role] is not None:
                        point_cloud_handles[role].remove()
                        point_cloud_handles[role] = None
                    continue

                bundle = latest_bundles[role]
                intrinsic = latest_intrinsics[role]
                camera_transform = camera_world_transforms[role]
                if bundle is None or intrinsic is None or camera_transform is None:
                    continue

                rotation_wc = camera_transform[:3, :3]
                translation_wc = camera_transform[:3, 3]
                points, colors = project_depth_to_world(
                    bundle.depth,
                    bundle.rgb,
                    _intrinsic_payload(intrinsic),
                    rotation_wc,
                    translation_wc,
                    _pcd_step_from_count(bundle.depth, pcd_count[role]),
                )
                if point_cloud_handles[role] is not None:
                    point_cloud_handles[role].remove()
                point_cloud_handles[role] = server.scene.add_point_cloud(
                    f"/pcd/{role}",
                    points=points,
                    colors=colors,
                    point_size=0.003,
                    point_shape="circle",
                )
            time.sleep(args.dt)
    except KeyboardInterrupt:
        return 0
    finally:
        for role in ("cam_ext", "cam_wrist"):
            _reset_reader(role)
        try:
            server.stop()
        except RuntimeError:
            pass


def main() -> int:
    return run(tyro.cli(Args))


if __name__ == "__main__":
    raise SystemExit(main())
