from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import tyro
import viser
import viser.transforms as vtf
from viser.extras import ViserUrdf


@dataclass
class Args:
    urdf: Path = Path("packages/ros2_kortex/kortex_description/robots/gen3.urdf")
    host: str = "0.0.0.0"
    port: int = 8080
    scale: float = 1.0
    collision: bool = False
    no_visual: bool = False
    ee_link: str | None = None


def _create_joint_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []

    def _update_robot() -> None:
        viser_urdf.update_cfg(np.array([slider.value for slider in slider_handles]))

    for joint_name, (lower, upper) in viser_urdf.get_actuated_joint_limits().items():
        lower = -np.pi if lower is None else lower
        upper = np.pi if upper is None else upper
        initial_value = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_value,
        )

        @slider.on_update
        def _(_) -> None:
            _update_robot()

        slider_handles.append(slider)
        initial_config.append(initial_value)

    return slider_handles, initial_config


def _resolve_ee_link(viser_urdf: ViserUrdf, requested: str | None) -> str:
    if requested is not None:
        return requested

    candidates = (
        "gen3_end_effector_link",
        "tool_frame",
        "end_effector_link",
        "ee_link",
    )
    link_names = {link.name for link in viser_urdf._urdf.link_map.values()}
    for candidate in candidates:
        if candidate in link_names:
            return candidate

    for link_name in link_names:
        if link_name.endswith("end_effector_link") or link_name.endswith("tool_frame"):
            return link_name
    raise ValueError("Failed to determine end effector link. Pass --ee-link.")


def _resolve_optional_link(viser_urdf: ViserUrdf, candidates: tuple[str, ...]) -> str | None:
    link_names = {link.name for link in viser_urdf._urdf.link_map.values()}
    for candidate in candidates:
        if candidate in link_names:
            return candidate
    return None


def _update_ee_frame(
    viser_urdf: ViserUrdf,
    frame: viser.FrameHandle,
    link_name: str,
    scale: float,
) -> None:
    transform = viser_urdf._urdf.get_transform(link_name, "world")
    frame.wxyz = vtf.SO3.from_matrix(transform[:3, :3]).wxyz
    frame.position = transform[:3, 3] * scale


def main() -> int:
    args = tyro.cli(Args)
    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    server = viser.ViserServer(host=args.host, port=args.port)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        scale=args.scale,
        load_meshes=not args.no_visual,
        load_collision_meshes=args.collision,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.35),
    )
    ee_link = _resolve_ee_link(viser_urdf, args.ee_link)
    camera_link = _resolve_optional_link(viser_urdf, ("camera", "camera_frame", "camera_link"))

    with server.gui.add_folder("Joint position control"):
        slider_handles, initial_config = _create_joint_sliders(server, viser_urdf)

    with server.gui.add_folder("Visibility"):
        show_visual_cb = server.gui.add_checkbox("Show visual", viser_urdf.show_visual)
        show_collision_cb = server.gui.add_checkbox(
            "Show collision",
            viser_urdf.show_collision,
        )

    @show_visual_cb.on_update
    def _(_) -> None:
        viser_urdf.show_visual = show_visual_cb.value

    @show_collision_cb.on_update
    def _(_) -> None:
        viser_urdf.show_collision = show_collision_cb.value

    show_visual_cb.visible = not args.no_visual
    show_collision_cb.visible = args.collision

    viser_urdf.update_cfg(np.array(initial_config))
    ee_frame = server.scene.add_frame(
        "/ee_frame",
        axes_length=0.12,
        axes_radius=0.006,
    )
    _update_ee_frame(viser_urdf, ee_frame, ee_link, args.scale)
    camera_frame = None
    if camera_link is not None:
        camera_frame = server.scene.add_frame(
            "/camera",
            axes_length=0.09,
            axes_radius=0.0045,
        )
        _update_ee_frame(viser_urdf, camera_frame, camera_link, args.scale)

    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    grid_z = 0.0 if trimesh_scene is None else float(trimesh_scene.bounds[0, 2])
    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, grid_z))

    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_) -> None:
        for slider, initial_value in zip(slider_handles, initial_config, strict=True):
            slider.value = initial_value
        _update_ee_frame(viser_urdf, ee_frame, ee_link, args.scale)
        if camera_frame is not None and camera_link is not None:
            _update_ee_frame(viser_urdf, camera_frame, camera_link, args.scale)

    for slider in slider_handles:
        @slider.on_update
        def _(_) -> None:
            _update_ee_frame(viser_urdf, ee_frame, ee_link, args.scale)
            if camera_frame is not None and camera_link is not None:
                _update_ee_frame(viser_urdf, camera_frame, camera_link, args.scale)

    print(f"URDF: {urdf_path}")
    print(f"End effector link: {ee_link}")
    if camera_link is not None:
        print(f"Camera link: {camera_link}")
    print("Open the viser URL shown above to inspect the robot.")

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
