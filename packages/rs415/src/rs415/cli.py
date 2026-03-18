from __future__ import annotations
"""CLI entry module that maps subcommands to camera and service actions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import tyro

from rs415 import Camera
from rs415.calibration import (
    BoardSpec,
    CheckerboardSpec,
    run_checkerboard_pose_preview,
    run_live_calibration,
    save_board_image,
)
from rs415.shm_io import run_shm_service


@dataclass
class ListCommand:
    def run(self) -> int:
        """Print all currently connected RealSense devices."""
        camera = Camera()
        devices = camera.list_devices()
        if not devices:
            print("No RealSense devices detected.")
            return 2

        for device in devices:
            print(f"{device.name} (serial={device.serial})")
        return 0


@dataclass
class InfoCommand:
    serial: str | None = None
    fps: int = 30

    def run(self) -> int:
        """Connect to a device and print stream intrinsics."""
        camera = Camera(fps=self.fps)
        try:
            connected = camera.connect(serial=self.serial)
            print(f"Connected: {connected.name} (serial={connected.serial})")
            intrinsics = camera.get_intrinsics()
            print("Intrinsics:")
            for stream_name, intr in intrinsics.items():
                print(
                    f"- {stream_name}: {intr.width}x{intr.height}, "
                    f"fx={intr.fx:.3f}, fy={intr.fy:.3f}, "
                    f"cx={intr.cx:.3f}, cy={intr.cy:.3f}"
                )
            return 0
        except RuntimeError as exc:
            print(f"Failed to get intrinsics: {exc}")
            return 2
        finally:
            camera.disconnect()


@dataclass
class ConnectCommand:
    serial: str | None = None
    mode: Literal["window", "server"] = "server"
    fps: int = 30

    def run(self) -> int:
        """Connect to a device and run either window mode or shared-memory mode."""
        camera = Camera(window=self.mode == "window", fps=self.fps)
        try:
            connected = camera.connect(serial=self.serial)
            print(f"Connected: {connected.name} (serial={connected.serial})")
            if self.mode == "window":
                return camera.spin()
            if self.mode == "server":
                return run_shm_service(camera)
            raise ValueError(f"Unsupported mode: {self.mode}")
        except RuntimeError as exc:
            print(f"Failed to connect: {exc}")
            return 2
        except KeyboardInterrupt:
            print("Stopping connect loop.")
            return 0
        finally:
            camera.disconnect()


@dataclass
class GenerateBoardCommand:
    output: Path = Path("aruco_gridboard.png")
    markers_x: int = 5
    markers_y: int = 7
    marker_length_m: float = 0.04
    marker_separation_m: float = 0.01
    dictionary: str = "DICT_4X4_50"
    image_width_px: int = 1600
    image_height_px: int = 2200
    margin_px: int = 32
    border_bits: int = 1

    def run(self) -> int:
        """Generate a printable ArUco GridBoard image."""
        board_spec = BoardSpec(
            markers_x=self.markers_x,
            markers_y=self.markers_y,
            marker_length_m=self.marker_length_m,
            marker_separation_m=self.marker_separation_m,
            dictionary_name=self.dictionary,
        )
        try:
            path = save_board_image(
                board_spec=board_spec,
                output_path=self.output,
                image_width_px=self.image_width_px,
                image_height_px=self.image_height_px,
                margin_px=self.margin_px,
                border_bits=self.border_bits,
            )
            print(f"Saved board image to {path}")
            return 0
        except (RuntimeError, ValueError) as exc:
            print(f"Failed to generate board image: {exc}")
            return 2


@dataclass
class CalibrateCommand:
    serial: str | None = None
    output: Path = Path("calibration.json")
    markers_x: int = 5
    markers_y: int = 7
    marker_length_m: float = 0.04
    marker_separation_m: float = 0.01
    dictionary: str = "DICT_4X4_50"
    target_samples: int = 20
    min_markers: int = 8

    def run(self) -> int:
        """Calibrate the RGB camera with a live ArUco GridBoard capture loop."""
        board_spec = BoardSpec(
            markers_x=self.markers_x,
            markers_y=self.markers_y,
            marker_length_m=self.marker_length_m,
            marker_separation_m=self.marker_separation_m,
            dictionary_name=self.dictionary,
        )
        camera = Camera(window=False)
        try:
            connected = camera.connect(serial=self.serial)
            print(f"Connected: {connected.name} (serial={connected.serial})")
            result = run_live_calibration(
                camera=camera,
                board_spec=board_spec,
                output_path=self.output,
                target_samples=self.target_samples,
                min_markers=self.min_markers,
            )
            print("Camera matrix:")
            for row in result.camera_matrix:
                print(f"- {row}")
            print(f"Distortion coefficients: {result.dist_coeffs}")
            return 0
        except KeyboardInterrupt:
            print("Calibration cancelled.")
            return 0
        except (RuntimeError, ValueError) as exc:
            print(f"Failed to calibrate camera: {exc}")
            return 2
        finally:
            camera.disconnect()


@dataclass
class DetectCheckerboardCommand:
    serial: str | None = None
    corners_x: int = 5
    corners_y: int = 4
    square_size_m: float = 0.03
    aruco_dictionary: str = "DICT_4X4_50"
    aruco_marker_length_m: float = 0.05
    aruco_world_id: int | None = None
    output_dir: Path = Path("calib")

    def run(self) -> int:
        """Preview checkerboard corners and solvePnP pose, then save a visualization."""
        checkerboard = CheckerboardSpec(
            corners_x=self.corners_x,
            corners_y=self.corners_y,
            square_size_m=self.square_size_m,
        )
        camera = Camera(window=False)
        try:
            connected = camera.connect(serial=self.serial)
            print(f"Connected: {connected.name} (serial={connected.serial})")
            saved = run_checkerboard_pose_preview(
                camera=camera,
                checkerboard=checkerboard,
                output_dir=self.output_dir,
                aruco_dictionary_name=self.aruco_dictionary,
                aruco_marker_length_m=self.aruco_marker_length_m,
                aruco_world_id=self.aruco_world_id,
            )
            print(f"Saved YAML: {saved}")
            return 0
        except KeyboardInterrupt:
            print("Checkerboard preview cancelled.")
            return 0
        except (RuntimeError, ValueError) as exc:
            print(f"Failed to detect checkerboard: {exc}")
            return 2
        finally:
            camera.disconnect()


Command = tyro.extras.subcommand_type_from_defaults(
    {
        "list": ListCommand(),
        "info": InfoCommand(),
        "connect": ConnectCommand(),
        "generate-board": GenerateBoardCommand(),
        "calibrate": DetectCheckerboardCommand(),
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute the selected subcommand."""
    command = tyro.cli(Command, args=list(argv) if argv is not None else None)
    return command.run()
