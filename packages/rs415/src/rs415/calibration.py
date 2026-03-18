from __future__ import annotations

"""ArUco GridBoard-based camera calibration helpers."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rs415.rs415 import Camera, IntrinsicInfo


@dataclass(frozen=True)
class BoardSpec:
    markers_x: int = 5
    markers_y: int = 7
    marker_length_m: float = 0.04
    marker_separation_m: float = 0.01
    dictionary_name: str = "DICT_4X4_50"

    def create_board(self) -> cv2.aruco.GridBoard:
        return cv2.aruco.GridBoard(
            (self.markers_x, self.markers_y),
            self.marker_length_m,
            self.marker_separation_m,
            get_aruco_dictionary(self.dictionary_name),
        )


@dataclass(frozen=True)
class Sample:
    object_points: np.ndarray
    image_points: np.ndarray
    marker_count: int


@dataclass(frozen=True)
class CalibrationResult:
    image_width: int
    image_height: int
    sample_count: int
    reprojection_error: float
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    board: dict[str, Any]


@dataclass(frozen=True)
class CheckerboardSpec:
    corners_x: int = 5
    corners_y: int = 4
    square_size_m: float = 0.03

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.corners_x, self.corners_y)

    def object_points(self) -> np.ndarray:
        col_center = (self.corners_x - 1) / 2.0
        row_center = (self.corners_y - 1) / 2.0
        points: list[list[float]] = []
        for row in range(self.corners_y):
            for col in range(self.corners_x):
                x = (col - col_center) * self.square_size_m
                y = 0.0
                z = (row_center - row) * self.square_size_m
                points.append([x, y, z])
        return np.asarray(points, dtype=np.float32)


@dataclass(frozen=True)
class CheckerboardDetection:
    corners: np.ndarray
    rvec: np.ndarray | None
    tvec: np.ndarray | None


@dataclass(frozen=True)
class ArucoDetection:
    corners: tuple[np.ndarray, ...]
    ids: np.ndarray
    rvecs: tuple[np.ndarray, ...]
    tvecs: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray


def get_aruco_dictionary(dictionary_name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2.aruco, dictionary_name):
        available = sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))
        raise ValueError(
            f"Unsupported ArUco dictionary '{dictionary_name}'. "
            f"Available values include: {', '.join(available[:12])}"
        )
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def create_detector(board_spec: BoardSpec) -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(
        get_aruco_dictionary(board_spec.dictionary_name), parameters
    )


def create_aruco_detector(dictionary_name: str) -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(
        get_aruco_dictionary(dictionary_name),
        parameters,
    )


def build_initial_guess(intrinsic: IntrinsicInfo | None) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if intrinsic is None:
        return None, None, 0
    camera_matrix = np.array(
        [
            [intrinsic.fx, 0.0, intrinsic.cx],
            [0.0, intrinsic.fy, intrinsic.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array(intrinsic.coeffs[:5], dtype=np.float64)
    return camera_matrix, dist_coeffs, cv2.CALIB_USE_INTRINSIC_GUESS


def intrinsic_to_matrices(intrinsic: IntrinsicInfo) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [
            [intrinsic.fx, 0.0, intrinsic.cx],
            [0.0, intrinsic.fy, intrinsic.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array(intrinsic.coeffs[:5], dtype=np.float64)
    return camera_matrix, dist_coeffs


def detect_checkerboard(
    frame_bgr: np.ndarray,
    checkerboard: CheckerboardSpec,
    intrinsic: IntrinsicInfo | None = None,
) -> CheckerboardDetection | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, checkerboard.pattern_size, flags)
    if not found or corners is None:
        return None

    term_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=term_criteria,
    )

    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    if intrinsic is not None:
        camera_matrix, dist_coeffs = intrinsic_to_matrices(intrinsic)
        solved, rvec, tvec = cv2.solvePnP(
            checkerboard.object_points(),
            refined.reshape(-1, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            rvec = None
            tvec = None

    return CheckerboardDetection(corners=refined, rvec=rvec, tvec=tvec)


def detect_aruco_markers(
    frame_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    intrinsic: IntrinsicInfo | None = None,
    marker_length_m: float | None = None,
) -> ArucoDetection | None:
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None or len(ids) == 0:
        return None
    rvecs: tuple[np.ndarray, ...] = ()
    tvecs: tuple[np.ndarray, ...] = ()
    if intrinsic is not None and marker_length_m is not None and marker_length_m > 0.0:
        camera_matrix, dist_coeffs = intrinsic_to_matrices(intrinsic)
        half = marker_length_m / 2.0
        marker_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        estimated_rvecs: list[np.ndarray] = []
        estimated_tvecs: list[np.ndarray] = []
        for marker_corners in corners:
            image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            solved, rvec, tvec = cv2.solvePnP(
                marker_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not solved:
                estimated_rvecs.append(np.zeros((3, 1), dtype=np.float64))
                estimated_tvecs.append(np.zeros((3, 1), dtype=np.float64))
                continue
            estimated_rvecs.append(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
            estimated_tvecs.append(np.asarray(tvec, dtype=np.float64).reshape(3, 1))
        rvecs = tuple(estimated_rvecs)
        tvecs = tuple(estimated_tvecs)
    return ArucoDetection(
        corners=tuple(corners),
        ids=np.asarray(ids, dtype=np.int32),
        rvecs=rvecs,
        tvecs=tvecs,
    )


def select_aruco_pose(
    aruco_detection: ArucoDetection | None,
    preferred_marker_id: int | None = None,
) -> Pose | None:
    if aruco_detection is None or len(aruco_detection.rvecs) != len(aruco_detection.ids):
        return None

    selected_index = 0
    if preferred_marker_id is not None:
        matches = np.where(aruco_detection.ids.reshape(-1) == preferred_marker_id)[0]
        if len(matches) == 0:
            return None
        selected_index = int(matches[0])

    rvec = aruco_detection.rvecs[selected_index]
    tvec = aruco_detection.tvecs[selected_index]
    if not np.any(tvec):
        return None
    return Pose(rvec=rvec, tvec=tvec)


def align_checkerboard_pose_to_aruco(
    checkerboard_pose: Pose | None,
    aruco_pose: Pose | None,
) -> Pose | None:
    if checkerboard_pose is None:
        return None

    checkerboard_rotation, _ = cv2.Rodrigues(checkerboard_pose.rvec)
    x_axis = checkerboard_rotation[:, 0]
    y_axis = checkerboard_rotation[:, 2]
    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm == 0.0:
        return checkerboard_pose
    z_axis = z_axis / z_norm

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    aligned_rotation = np.column_stack((x_axis, y_axis, z_axis))
    aligned_rvec, _ = cv2.Rodrigues(aligned_rotation.astype(np.float64))
    return Pose(
        rvec=np.asarray(aligned_rvec, dtype=np.float64),
        tvec=np.asarray(checkerboard_pose.tvec, dtype=np.float64),
    )


def visualize_checkerboard_detection(
    frame_bgr: np.ndarray,
    checkerboard: CheckerboardSpec,
    detection: CheckerboardDetection | None,
    intrinsic: IntrinsicInfo | None = None,
    aruco_detection: ArucoDetection | None = None,
    world_pose: Pose | None = None,
) -> np.ndarray:
    preview = frame_bgr.copy()
    if aruco_detection is not None:
        cv2.aruco.drawDetectedMarkers(preview, list(aruco_detection.corners), aruco_detection.ids)
        if intrinsic is not None and len(aruco_detection.rvecs) == len(aruco_detection.corners):
            camera_matrix, dist_coeffs = intrinsic_to_matrices(intrinsic)
            for rvec, tvec in zip(aruco_detection.rvecs, aruco_detection.tvecs, strict=True):
                if not np.any(tvec):
                    continue
                cv2.drawFrameAxes(
                    preview,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    0.03,
                    2,
                )

    found = detection is not None
    corners = (
        np.empty((0, 1, 2), dtype=np.float32)
        if detection is None
        else detection.corners.astype(np.float32)
    )
    cv2.drawChessboardCorners(preview, checkerboard.pattern_size, corners, found)

    if world_pose is not None and intrinsic is not None:
        camera_matrix, dist_coeffs = intrinsic_to_matrices(intrinsic)
        world_axis_length = checkerboard.square_size_m * 4.0
        cv2.drawFrameAxes(
            preview,
            camera_matrix,
            dist_coeffs,
            world_pose.rvec,
            world_pose.tvec,
            world_axis_length,
            4,
        )

    return preview


def save_image(image: np.ndarray, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image to {path}")
    return path


def invert_pose(pose: Pose) -> Pose:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ pose.tvec.reshape(3, 1)
    inv_rvec, _ = cv2.Rodrigues(inv_rotation)
    return Pose(
        rvec=np.asarray(inv_rvec, dtype=np.float64),
        tvec=np.asarray(inv_translation, dtype=np.float64),
    )


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = 0.0
    return roll, pitch, yaw


def format_yaml_value(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(format_yaml_value(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(format_yaml_value(item, indent + 2))
            else:
                lines.append(f"{prefix}- {item}")
        return lines
    return [f"{prefix}{value}"]


def intrinsic_to_dict(intrinsic: IntrinsicInfo) -> dict[str, Any]:
    return {
        "width": intrinsic.width,
        "height": intrinsic.height,
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "cx": float(intrinsic.cx),
        "cy": float(intrinsic.cy),
    }


def save_camera_yaml(
    camera: Camera,
    world_pose: Pose,
    checkerboard: CheckerboardSpec,
    output_dir: str | Path = "calib",
) -> Path:
    serial = camera.connected_serial
    if serial is None:
        raise RuntimeError("Camera serial is not available.")

    color_intrinsic = camera.get_color_intrinsics()

    camera_pose_world = invert_pose(world_pose)
    rotation, _ = cv2.Rodrigues(camera_pose_world.rvec)
    roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
    translation = camera_pose_world.tvec.reshape(3).tolist()

    payload = {
        "serial_num": serial,
        "cam_pose": {
            "frame": "world",
            "trans": [float(value) for value in translation],
            "rpy": [float(roll), float(pitch), float(yaw)],
        },
        "checkerboard": {
            "corners_x": checkerboard.corners_x,
            "corners_y": checkerboard.corners_y,
            "square_size_m": checkerboard.square_size_m,
        },
        "intrinsic": intrinsic_to_dict(color_intrinsic),
    }

    output_path = Path(output_dir) / f"{serial}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(format_yaml_value(payload)) + "\n", encoding="utf-8")
    return output_path


def run_checkerboard_pose_preview(
    camera: Camera,
    checkerboard: CheckerboardSpec,
    output_dir: str | Path = "calib",
    window_name: str = "RS415 Checkerboard",
    aruco_dictionary_name: str = "DICT_4X4_50",
    aruco_marker_length_m: float = 0.05,
    aruco_world_id: int | None = None,
) -> Path:
    color_intrinsic = camera.get_color_intrinsics()
    aruco_detector = create_aruco_detector(aruco_dictionary_name)

    print("Checkerboard preview controls:")
    print(
        f"- Show the {checkerboard.corners_x}x{checkerboard.corners_y} "
        "inner-corner board to the RGB camera."
    )
    print("- The frame origin is at the checkerboard center.")
    print("- The final world frame uses checkerboard pose with ArUco axis convention.")
    print(
        f"- ArUco markers are detected with {aruco_dictionary_name} "
        f"and drawn with {aruco_marker_length_m:.3f} m frame axes."
    )
    if aruco_world_id is not None:
        print(f"- World orientation marker id: {aruco_world_id}")
    print("- Press s to save calib/{serial}.yaml for the current pose.")
    print("- Press q or ESC to quit.")

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    latest_saved: Path | None = None
    try:
        while True:
            frame_rgb = camera.get_frames(types=("rgb",), timeout_ms=5000)["rgb"]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            detection = detect_checkerboard(
                frame_bgr=frame_bgr,
                checkerboard=checkerboard,
                intrinsic=color_intrinsic,
            )
            aruco_detection = detect_aruco_markers(
                frame_bgr=frame_bgr,
                detector=aruco_detector,
                intrinsic=color_intrinsic,
                marker_length_m=aruco_marker_length_m,
            )
            world_pose: Pose | None = None
            if detection is not None and detection.rvec is not None and detection.tvec is not None:
                checkerboard_pose = Pose(rvec=detection.rvec, tvec=detection.tvec)
                marker_pose = select_aruco_pose(
                    aruco_detection=aruco_detection,
                    preferred_marker_id=aruco_world_id,
                )
                world_pose = align_checkerboard_pose_to_aruco(
                    checkerboard_pose=checkerboard_pose,
                    aruco_pose=marker_pose,
                )
            preview = visualize_checkerboard_detection(
                frame_bgr=frame_bgr,
                checkerboard=checkerboard,
                detection=detection,
                intrinsic=color_intrinsic,
                aruco_detection=aruco_detection,
                world_pose=world_pose,
            )

            status = "detected" if detection is not None else "not-detected"
            color = (0, 255, 0) if detection is not None else (0, 140, 255)
            aruco_count = 0 if aruco_detection is None else len(aruco_detection.ids)
            world_status = "ready" if world_pose is not None else "waiting"
            cv2.putText(
                preview,
                f"checkerboard={status} pattern={checkerboard.corners_x}x{checkerboard.corners_y}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"aruco={aruco_count} world={world_status} | s save | q quit",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                if world_pose is None:
                    print("World pose is not available in the current frame.")
                    continue
                latest_saved = save_camera_yaml(
                    camera=camera,
                    world_pose=world_pose,
                    checkerboard=checkerboard,
                    output_dir=output_dir,
                )
                print(f"Saved calibration YAML to {latest_saved}")
    finally:
        cv2.destroyWindow(window_name)

    if latest_saved is None:
        raise RuntimeError("No visualization was saved.")
    return latest_saved


def detect_board_sample(
    frame_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    board: cv2.aruco.Board,
) -> tuple[Sample | None, np.ndarray, int]:
    corners, ids, rejected = detector.detectMarkers(frame_bgr)
    preview = frame_bgr.copy()
    rejected_count = 0 if rejected is None else len(rejected)

    if ids is None or len(ids) == 0:
        return None, preview, rejected_count

    cv2.aruco.drawDetectedMarkers(preview, corners, ids)
    object_points, image_points = board.matchImagePoints(corners, ids)
    if len(object_points) == 0 or len(image_points) == 0:
        return None, preview, rejected_count

    sample = Sample(
        object_points=np.asarray(object_points, dtype=np.float32).reshape(-1, 1, 3),
        image_points=np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2),
        marker_count=len(ids),
    )
    return sample, preview, rejected_count


def calibrate_from_samples(
    samples: list[Sample],
    image_size: tuple[int, int],
    intrinsic_guess: IntrinsicInfo | None = None,
    board_spec: BoardSpec | None = None,
) -> CalibrationResult:
    if len(samples) < 3:
        raise ValueError("At least 3 samples are required for calibration.")

    object_points = [sample.object_points for sample in samples]
    image_points = [sample.image_points for sample in samples]
    camera_matrix, dist_coeffs, flags = build_initial_guess(intrinsic_guess)

    reprojection_error, calibrated_matrix, calibrated_dist_coeffs, _, _ = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=flags,
        )
    )

    return CalibrationResult(
        image_width=image_size[0],
        image_height=image_size[1],
        sample_count=len(samples),
        reprojection_error=float(reprojection_error),
        camera_matrix=np.asarray(calibrated_matrix, dtype=np.float64).tolist(),
        dist_coeffs=np.asarray(calibrated_dist_coeffs, dtype=np.float64).ravel().tolist(),
        board=asdict(board_spec) if board_spec is not None else {},
    )


def save_calibration(result: CalibrationResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def render_board(
    board_spec: BoardSpec,
    image_width_px: int,
    image_height_px: int,
    margin_px: int = 24,
    border_bits: int = 1,
) -> np.ndarray:
    board = board_spec.create_board()
    marker_ids = np.asarray(board.getIds(), dtype=np.int32).tolist()
    usable_width = image_width_px - (2 * margin_px)
    usable_height = image_height_px - (2 * margin_px)
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("Image size must be larger than twice the margin.")

    board_width_m = (
        board_spec.markers_x * board_spec.marker_length_m
        + (board_spec.markers_x - 1) * board_spec.marker_separation_m
    )
    board_height_m = (
        board_spec.markers_y * board_spec.marker_length_m
        + (board_spec.markers_y - 1) * board_spec.marker_separation_m
    )
    px_per_meter = min(usable_width / board_width_m, usable_height / board_height_m)
    marker_px = max(1, int(round(board_spec.marker_length_m * px_per_meter)))
    separation_px = max(0, int(round(board_spec.marker_separation_m * px_per_meter)))

    drawn_board_width = (
        board_spec.markers_x * marker_px
        + (board_spec.markers_x - 1) * separation_px
    )
    drawn_board_height = (
        board_spec.markers_y * marker_px
        + (board_spec.markers_y - 1) * separation_px
    )
    offset_x = (image_width_px - drawn_board_width) // 2
    offset_y = (image_height_px - drawn_board_height) // 2

    image = np.full((image_height_px, image_width_px), 255, dtype=np.uint8)
    dictionary = get_aruco_dictionary(board_spec.dictionary_name)
    for marker_index, marker_id in enumerate(marker_ids):
        col = marker_index % board_spec.markers_x
        row = marker_index // board_spec.markers_x
        x0 = offset_x + col * (marker_px + separation_px)
        y0 = offset_y + row * (marker_px + separation_px)
        marker = cv2.aruco.generateImageMarker(
            dictionary, marker_id, marker_px, borderBits=border_bits
        )
        image[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker
    return image


def save_board_image(
    board_spec: BoardSpec,
    output_path: str | Path,
    image_width_px: int,
    image_height_px: int,
    margin_px: int = 24,
    border_bits: int = 1,
) -> Path:
    image = render_board(
        board_spec=board_spec,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        margin_px=margin_px,
        border_bits=border_bits,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write board image to {path}")
    return path


def run_live_calibration(
    camera: Camera,
    board_spec: BoardSpec,
    output_path: str | Path,
    target_samples: int,
    min_markers: int,
    window_name: str = "RS415 Calibration",
) -> CalibrationResult:
    detector = create_detector(board_spec)
    board = board_spec.create_board()
    intrinsics = camera.get_color_intrinsics()
    samples: list[Sample] = []
    image_size: tuple[int, int] | None = None

    print("Calibration controls:")
    print("- Present the GridBoard to the RGB camera from varied angles/distances.")
    print("- Press SPACE to capture the current detection as a sample.")
    print("- Press ENTER to solve calibration once enough samples are collected.")
    print("- Press q or ESC to cancel.")

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame_rgb = camera.get_frames(types=("rgb",), timeout_ms=5000)["rgb"]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            image_size = (frame_bgr.shape[1], frame_bgr.shape[0])

            detected_sample, preview, rejected_count = detect_board_sample(
                frame_bgr=frame_bgr,
                detector=detector,
                board=board,
            )

            detected_markers = 0 if detected_sample is None else detected_sample.marker_count
            status = (
                f"samples={len(samples)}/{target_samples} "
                f"markers={detected_markers} rejected={rejected_count}"
            )
            cv2.putText(
                preview,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if detected_markers >= min_markers else (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                "SPACE capture | ENTER solve | q quit",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key == 32:
                if detected_sample is None or detected_markers < min_markers:
                    print(
                        f"Skipped sample: need at least {min_markers} detected markers "
                        f"(got {detected_markers})."
                    )
                    continue
                samples.append(detected_sample)
                print(
                    f"Captured sample {len(samples)}/{target_samples} "
                    f"with {detected_markers} markers."
                )
            if key in (10, 13):
                if len(samples) < 3:
                    print("Need at least 3 valid samples before solving calibration.")
                    continue
                break
            if len(samples) >= target_samples:
                break
    finally:
        cv2.destroyWindow(window_name)

    if image_size is None:
        raise RuntimeError("No image frames were captured for calibration.")

    result = calibrate_from_samples(
        samples=samples,
        image_size=image_size,
        intrinsic_guess=intrinsics,
        board_spec=board_spec,
    )
    saved_path = save_calibration(result, output_path)
    print(f"Saved calibration to {saved_path}")
    print(f"Reprojection error: {result.reprojection_error:.6f}")
    return result
