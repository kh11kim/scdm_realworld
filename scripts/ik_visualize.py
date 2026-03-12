from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import tyro
import viser
import viser.transforms as vtf
from viser.extras import ViserUrdf

from scdm_realworld.robot_model import RobotModel


@dataclass
class Args:
    urdf: Path = Path("assets/gen3_allegro/gen3_allegro.urdf")
    host: str = "0.0.0.0"
    port: int = 8080
    scale: float = 1.0


def _make_initial_arm_q(robot: RobotModel) -> np.ndarray:
    values: list[float] = []
    for joint_name in robot.arm_joint_names:
        joint = robot._urdf.joint_map[joint_name]
        lower = -np.pi if joint.limit is None or joint.limit.lower is None else float(joint.limit.lower)
        upper = np.pi if joint.limit is None or joint.limit.upper is None else float(joint.limit.upper)
        values.append(0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0)
    return np.asarray(values, dtype=np.float64)


def _set_arm_q(robot: RobotModel, q_arm: np.ndarray) -> np.ndarray:
    robot.set_joint_positions(q=q_arm)
    return robot.visual_configuration


def _update_frame(frame: viser.FrameHandle, transform: np.ndarray, scale: float) -> None:
    frame.wxyz = vtf.SO3.from_matrix(transform[:3, :3]).wxyz
    frame.position = transform[:3, 3] * scale


def main() -> int:
    args = tyro.cli(Args)
    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    server = viser.ViserServer(host=args.host, port=args.port)
    current_robot = RobotModel.from_urdf(urdf_path)
    sample_robot = RobotModel.from_urdf(urdf_path)

    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        scale=args.scale,
        load_meshes=True,
        load_collision_meshes=False,
    )

    q0_default = _make_initial_arm_q(current_robot)
    qsol_default = np.array((0.25, -0.35, 0.55, -1.0, 0.4, 0.75, -0.3), dtype=np.float64)

    _set_arm_q(current_robot, q0_default)
    _set_arm_q(sample_robot, qsol_default)
    viser_urdf.update_cfg(current_robot.visual_configuration)

    palm_frame = server.scene.add_frame(
        "/palm_pose",
        axes_length=0.09,
        axes_radius=0.0045,
    )
    tsol_frame = server.scene.add_frame(
        "/Tsol",
        axes_length=0.11,
        axes_radius=0.0055,
    )
    server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy")

    with server.gui.add_folder("q0"):
        q0_sliders: list[viser.GuiInputHandle[float]] = []
        for index, joint_name in enumerate(current_robot.arm_joint_names):
            joint = current_robot._urdf.joint_map[joint_name]
            lower = -np.pi if joint.limit is None or joint.limit.lower is None else float(joint.limit.lower)
            upper = np.pi if joint.limit is None or joint.limit.upper is None else float(joint.limit.upper)
            q0_sliders.append(
                server.gui.add_slider(
                    label=joint_name,
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=float(q0_default[index]),
                )
            )

    with server.gui.add_folder("qsol sample"):
        qsol_sliders: list[viser.GuiInputHandle[float]] = []
        for index, joint_name in enumerate(sample_robot.arm_joint_names):
            joint = sample_robot._urdf.joint_map[joint_name]
            lower = -np.pi if joint.limit is None or joint.limit.lower is None else float(joint.limit.lower)
            upper = np.pi if joint.limit is None or joint.limit.upper is None else float(joint.limit.upper)
            qsol_sliders.append(
                server.gui.add_slider(
                    label=joint_name,
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=float(qsol_default[index]),
                )
            )

    with server.gui.add_folder("IK"):
        solve_button = server.gui.add_button("Solve")
        solved_q_text = server.gui.add_text("Solved q", initial_value="", disabled=True)

    def _current_q0() -> np.ndarray:
        return np.asarray([slider.value for slider in q0_sliders], dtype=np.float64)

    def _current_qsol() -> np.ndarray:
        return np.asarray([slider.value for slider in qsol_sliders], dtype=np.float64)

    def _refresh_current_robot() -> None:
        q0 = _current_q0()
        _set_arm_q(current_robot, q0)
        viser_urdf.update_cfg(current_robot.visual_configuration)
        _update_frame(palm_frame, current_robot.get_link_pose("palm"), args.scale)

    def _refresh_target_frame() -> np.ndarray:
        qsol = _current_qsol()
        _set_arm_q(sample_robot, qsol)
        tsol = sample_robot.get_link_pose("palm")
        _update_frame(tsol_frame, tsol, args.scale)
        return tsol

    _refresh_current_robot()
    _refresh_target_frame()

    for slider in q0_sliders:
        @slider.on_update
        def _(_) -> None:
            _refresh_current_robot()

    for slider in qsol_sliders:
        @slider.on_update
        def _(_) -> None:
            _refresh_target_frame()

    @solve_button.on_click
    def _(_) -> None:
        q0 = _current_q0()
        tsol = _refresh_target_frame()
        q_arm = current_robot.solve(tsol, q0=q0, target="palm")
        for slider, value in zip(q0_sliders, q_arm, strict=True):
            slider.value = float(value)
        solved_q_text.value = np.array2string(q_arm, precision=4, suppress_small=True)
        _refresh_current_robot()

    print(f"URDF: {urdf_path}")
    print("Open the viser URL shown above to inspect IK behavior.")

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            server.stop()
        except RuntimeError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
