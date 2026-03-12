from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import interp1d
import tyro
import viser
import viser.transforms as vtf
from viser.extras import ViserUrdf

from scdm_realworld.environment import BoxEnvironment
from scdm_realworld.robot_model import RobotModel
from scdm_realworld.utils.geometry import matrix_to_wxyz, rpy_to_matrix


@dataclass
class Args:
    urdf: Path = Path("assets/gen3_allegro/gen3_allegro.urdf")
    env: Path = Path("assets/box_env.yaml")
    host: str = "0.0.0.0"
    port: int = 8080
    scale: float = 1.0
    rrt_max_time: float = 2.0
    smooth_max_time: float = 1.0
    min_goal_distance: float = 0.8
    max_goal_distance: float = 1.6
    sample_attempts: int = 1000


def _update_frame(frame: viser.FrameHandle, transform: np.ndarray, scale: float) -> None:
    frame.wxyz = vtf.SO3.from_matrix(transform[:3, :3]).wxyz
    frame.position = transform[:3, 3] * scale


def _box_wxyz(rpy: np.ndarray) -> tuple[float, float, float, float]:
    return matrix_to_wxyz(rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2])))


def _set_arm_q(robot: RobotModel, q_arm: np.ndarray) -> np.ndarray:
    robot.set_joint_positions(q=q_arm)
    return robot.visual_configuration


def _sample_collision_free_q(
    robot: RobotModel,
    box_env: BoxEnvironment,
    *,
    attempts: int,
) -> np.ndarray:
    q_min, q_max = robot.get_joint_limits()
    for _ in range(attempts):
        q = np.random.uniform(q_min, q_max)
        if not robot.is_collision(q, box_env):
            return q
    raise RuntimeError(f"Failed to sample collision-free q within {attempts} attempts.")


def _sample_nearby_collision_free_q(
    robot: RobotModel,
    box_env: BoxEnvironment,
    q_center: np.ndarray,
    *,
    min_distance: float,
    max_distance: float,
    attempts: int,
) -> np.ndarray:
    q_min, q_max = robot.get_joint_limits()
    radius_scale = np.full_like(q_center, max_distance / np.sqrt(len(q_center)))
    for _ in range(attempts):
        delta = np.random.uniform(-radius_scale, radius_scale)
        q = np.clip(q_center + delta, q_min, q_max)
        distance = np.linalg.norm(q - q_center)
        if distance < min_distance or distance > max_distance:
            continue
        if not robot.is_collision(q, box_env):
            return q
    raise RuntimeError(f"Failed to sample nearby collision-free q within {attempts} attempts.")


def main() -> int:
    args = tyro.cli(Args)
    urdf_path = args.urdf.resolve()
    env_path = args.env.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    box_env = BoxEnvironment.load(env_path)
    robot = RobotModel.from_urdf(urdf_path)
    plan_fn = robot.get_plan_fn(box_env)
    robot.set_joint_positions(q=np.zeros(len(robot.arm_joint_names), dtype=np.float64))

    server = viser.ViserServer(host=args.host, port=args.port)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        scale=args.scale,
        load_meshes=True,
        load_collision_meshes=False,
    )
    viser_urdf.update_cfg(robot.visual_configuration)

    for index, box in enumerate(box_env.boxes):
        server.scene.add_box(
            f"/env/{box.name}",
            color=((220 - 20 * (index % 4)), 100 + 20 * (index % 5), 180),
            dimensions=(1.0, 1.0, 1.0),
            opacity=0.35,
            position=tuple(box.center.tolist()),
            wxyz=_box_wxyz(box.rpy),
            scale=tuple(box.size.tolist()),
        )

    q0_frame = server.scene.add_frame("/q0_palm", axes_length=0.09, axes_radius=0.0045)
    qg_frame = server.scene.add_frame("/qg_palm", axes_length=0.09, axes_radius=0.0045)
    server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy")

    with server.gui.add_folder("Planning"):
        gen_problem_button = server.gui.add_button("Gen Problem")
        plan_button = server.gui.add_button("Plan")
        max_plan_time_gui = server.gui.add_number("Max Plan Time", initial_value=args.rrt_max_time, step=0.1)
        max_smooth_time_gui = server.gui.add_number("Max Smooth Time", initial_value=args.smooth_max_time, step=0.1)
        time_slider = server.gui.add_slider("Time", min=0.0, max=1.0, step=0.001, initial_value=0.0)
        status_text = server.gui.add_text("Status", initial_value="Idle", disabled=True)
        q0_text = server.gui.add_text("q0", initial_value="", disabled=True)
        qg_text = server.gui.add_text("qg", initial_value="", disabled=True)
        traj_text = server.gui.add_text("Trajectory", initial_value="None", disabled=True)

    q0_state = {"value": None}
    qg_state = {"value": None}
    traj_state = {"value": None}
    traj_interp = {"value": None}

    def _refresh_robot(q_arm: np.ndarray) -> None:
        _set_arm_q(robot, q_arm)
        viser_urdf.update_cfg(robot.visual_configuration)

    def _refresh_problem_frames() -> None:
        if q0_state["value"] is not None:
            _set_arm_q(robot, q0_state["value"])
            _update_frame(q0_frame, robot.get_link_pose("palm"), args.scale)
        if qg_state["value"] is not None:
            _set_arm_q(robot, qg_state["value"])
            _update_frame(qg_frame, robot.get_link_pose("palm"), args.scale)
        if q0_state["value"] is not None:
            _refresh_robot(q0_state["value"])

    def _set_trajectory(trajectory: list[np.ndarray] | None) -> None:
        traj_state["value"] = trajectory
        if trajectory is None or len(trajectory) == 0:
            traj_interp["value"] = None
            traj_text.value = "None"
            return
        times = np.linspace(0.0, 1.0, len(trajectory))
        qs = np.asarray(trajectory, dtype=np.float64)
        traj_interp["value"] = interp1d(times, qs, axis=0, kind="linear")
        traj_text.value = str(len(trajectory))

    @gen_problem_button.on_click
    def _(_) -> None:
        q0 = _sample_collision_free_q(robot, box_env, attempts=args.sample_attempts)
        qg = _sample_nearby_collision_free_q(
            robot,
            box_env,
            q0,
            min_distance=args.min_goal_distance,
            max_distance=args.max_goal_distance,
            attempts=args.sample_attempts,
        )
        q0_state["value"] = q0
        qg_state["value"] = qg
        q0_text.value = np.array2string(q0, precision=3, suppress_small=True)
        qg_text.value = np.array2string(qg, precision=3, suppress_small=True)
        _set_trajectory(None)
        time_slider.value = 0.0
        status_text.value = "Problem Generated"
        _refresh_problem_frames()

    @plan_button.on_click
    def _(_) -> None:
        q0 = q0_state["value"]
        qg = qg_state["value"]
        if q0 is None or qg is None:
            status_text.value = "Missing Problem"
            return
        status_text.value = "Planning..."
        trajectory = plan_fn(
            q0,
            qg,
            rrt_max_time=float(max_plan_time_gui.value),
            smooth_max_time=float(max_smooth_time_gui.value),
        )
        _set_trajectory(trajectory)
        time_slider.value = 0.0
        if trajectory is None:
            status_text.value = "Plan Failed"
            return
        status_text.value = "Plan Succeeded"
        _refresh_robot(np.asarray(trajectory[0], dtype=np.float64))

    @time_slider.on_update
    def _(_) -> None:
        interp = traj_interp["value"]
        if interp is None:
            return
        q = np.asarray(interp(time_slider.value), dtype=np.float64)
        _refresh_robot(q)

    print(f"URDF: {urdf_path}")
    print(f"Environment: {env_path}")
    print("Open the viser URL shown above to inspect planning.")

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
