from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import tyro
import viser
from viser.extras import ViserUrdf

from kinova_gen3.client import get_joints
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
    dt: float = 0.1


def _box_wxyz(rpy) -> tuple[float, float, float, float]:
    return matrix_to_wxyz(rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2])))


def main() -> int:
    args = tyro.cli(Args)
    urdf_path = args.urdf.resolve()
    env_path = args.env.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    server = viser.ViserServer(host=args.host, port=args.port)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        scale=args.scale,
        load_meshes=True,
        load_collision_meshes=False,
    )
    robot = RobotModel.from_urdf(urdf_path)
    robot.set_joint_positions(q=get_joints())
    viser_urdf.update_cfg(robot.visual_configuration)
    server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy")
    box_env = BoxEnvironment.load(env_path)
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

    status_text = server.gui.add_text("Status", initial_value="Running", disabled=True)
    joints_text = server.gui.add_text("Joints", initial_value="", disabled=True)

    print(f"URDF: {urdf_path}")
    print(f"Environment: {env_path}")
    print("Open the viser URL shown above to inspect the live Kinova state.")

    try:
        while True:
            try:
                joints_rad = get_joints()
                robot.set_joint_positions(q=joints_rad)
                viser_urdf.update_cfg(robot.visual_configuration)
                joints_text.value = str([round(value, 4) for value in joints_rad])
                status_text.value = "Running"
            except Exception as exc:
                status_text.value = f"Error: {exc}"
            time.sleep(args.dt)
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            server.stop()
        except RuntimeError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
