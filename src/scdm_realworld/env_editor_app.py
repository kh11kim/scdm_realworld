from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import tyro
import viser
from viser.extras import ViserUrdf

from scdm_realworld.collision import compute_world_spheres, sphere_intersects_box
from scdm_realworld.environment import Box, BoxEnvironment
from scdm_realworld.robot_model import RobotModel
from scdm_realworld.runtime_config import (
    DEFAULT_APP_CONFIG_PATH,
    load_runtime_config,
    resolve_robot_urdf,
)
from scdm_realworld.utils.geometry import matrix_to_wxyz, rpy_to_matrix


@dataclass
class Args:
    app_config: Path = DEFAULT_APP_CONFIG_PATH
    urdf: Path | None = None
    env: Path = Path("assets/box_env.yaml")
    host: str = "0.0.0.0"
    port: int = 8080
    urdf_scale: float = 1.0


@dataclass
class EditableBox:
    name: str
    center: np.ndarray
    size: np.ndarray
    rpy: np.ndarray
    color: tuple[int, int, int]
    handle: viser.BoxHandle


@dataclass
class SphereVisual:
    handle: viser.IcosphereHandle
    radius: float


def _make_color(index: int) -> tuple[int, int, int]:
    palette = (
        (220, 80, 80),
        (80, 160, 240),
        (100, 200, 120),
        (240, 180, 60),
        (180, 120, 220),
        (90, 210, 210),
    )
    return palette[index % len(palette)]


def _box_wxyz(rpy: np.ndarray) -> tuple[float, float, float, float]:
    rotation = rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return matrix_to_wxyz(rotation)


def _select_box(
    name: str | None,
    boxes: dict[str, EditableBox],
    active_name: dict[str, str | None],
    controls: viser.TransformControlsHandle,
    size_gui: viser.GuiVector3Handle,
    center_gui: viser.GuiVector3Handle,
    name_gui: viser.GuiTextHandle,
) -> None:
    if name is None:
        active_name["value"] = None
        controls.visible = False
        name_gui.value = ""
        size_gui.value = (0.0, 0.0, 0.0)
        center_gui.value = (0.0, 0.0, 0.0)
        return

    box = boxes[name]
    controls.position = tuple(box.center.tolist())
    controls.wxyz = _box_wxyz(box.rpy)
    controls.visible = True
    active_name["value"] = name
    size_gui.value = tuple(box.size.tolist())
    center_gui.value = tuple(box.center.tolist())
    name_gui.value = box.name


def _make_initial_cfg(viser_urdf: ViserUrdf) -> np.ndarray:
    values: list[float] = []
    for lower, upper in viser_urdf.get_actuated_joint_limits().values():
        lower = -np.pi if lower is None else lower
        upper = np.pi if upper is None else upper
        values.append(0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0)
    return np.asarray(values, dtype=np.float64)


def _create_joint_sliders(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
) -> tuple[list[viser.GuiInputHandle[float]], np.ndarray]:
    sliders: list[viser.GuiInputHandle[float]] = []
    initial_cfg = _make_initial_cfg(viser_urdf)

    for index, (joint_name, (lower, upper)) in enumerate(
        viser_urdf.get_actuated_joint_limits().items()
    ):
        lower = -np.pi if lower is None else lower
        upper = np.pi if upper is None else upper
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=float(initial_cfg[index]),
        )
        sliders.append(slider)

    return sliders, initial_cfg


def run(args: Args) -> int:
    runtime_config = load_runtime_config(args.app_config)
    urdf_path = (resolve_robot_urdf(runtime_config) if args.urdf is None else args.urdf).resolve()
    server = viser.ViserServer(host=args.host, port=args.port)
    robot_model = RobotModel.from_urdf(urdf_path)

    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        scale=args.urdf_scale,
        load_meshes=True,
        load_collision_meshes=False,
    )
    link_spheres = robot_model._link_spheres
    sphere_visuals: list[SphereVisual] = []
    default_sphere_color = (70, 170, 255)
    collision_sphere_color = (255, 70, 70)

    initial_cfg = _make_initial_cfg(viser_urdf)
    viser_urdf.update_cfg(initial_cfg)
    robot_model.set_joint_positions(q=initial_cfg[: len(robot_model.arm_joint_names)])
    server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy")

    boxes: dict[str, EditableBox] = {}
    active_name: dict[str, str | None] = {"value": None}
    next_box_index = {"value": 0}
    suppress_gui_callbacks = {"value": False}
    suppress_controls_callbacks = {"value": False}

    controls = server.scene.add_transform_controls(
        "/box_transform",
        scale=0.25,
        disable_rotations=True,
        disable_sliders=True,
        visible=False,
    )

    with server.gui.add_folder("Joints"):
        joint_sliders, _ = _create_joint_sliders(server, viser_urdf)

    with server.gui.add_folder("Boxes"):
        add_button = server.gui.add_button("Add Box")
        export_button = server.gui.add_button("Export")
        delete_button = server.gui.add_button("Delete Selected")
        selected_name_gui = server.gui.add_text("Selected", initial_value="", disabled=True)
        center_gui = server.gui.add_vector3("Center", initial_value=(0.0, 0.0, 0.0), step=0.001)
        size_gui = server.gui.add_vector3("Size", initial_value=(0.1, 0.1, 0.1), step=0.05)

    def _box_environment() -> BoxEnvironment:
        return BoxEnvironment(
            [
                Box(name=box.name, center=box.center, size=box.size, rpy=box.rpy)
                for box in boxes.values()
            ]
        )

    def _current_arm_q() -> np.ndarray:
        return np.asarray(
            [slider.value for slider in joint_sliders[: len(robot_model.arm_joint_names)]],
            dtype=np.float64,
        )

    def _current_q() -> np.ndarray:
        return np.asarray([slider.value for slider in joint_sliders], dtype=np.float64)

    def _refresh_collision_visuals() -> None:
        robot_model.set_joint_positions(q=_current_arm_q())
        link_poses = robot_model.get_all_link_poses()
        world_spheres = compute_world_spheres(link_spheres, link_poses)
        env_boxes = list(_box_environment().boxes)

        while len(sphere_visuals) < len(world_spheres):
            sphere = world_spheres[len(sphere_visuals)]
            handle = server.scene.add_icosphere(
                f"/robot_spheres/{len(sphere_visuals)}",
                radius=float(sphere.radius),
                color=default_sphere_color,
                opacity=0.35,
                subdivisions=2,
                position=tuple(sphere.center.tolist()),
            )
            sphere_visuals.append(SphereVisual(handle=handle, radius=float(sphere.radius)))

        for visual, sphere in zip(sphere_visuals, world_spheres, strict=True):
            visual.handle.position = tuple(sphere.center.tolist())
            colliding = any(sphere_intersects_box(sphere, box) for box in env_boxes)
            visual.handle.color = collision_sphere_color if colliding else default_sphere_color
            visual.handle.visible = True

        for visual in sphere_visuals[len(world_spheres):]:
            visual.handle.visible = False

    def _refresh_selection(name: str | None) -> None:
        suppress_gui_callbacks["value"] = True
        suppress_controls_callbacks["value"] = True
        try:
            _select_box(
                name=name,
                boxes=boxes,
                active_name=active_name,
                controls=controls,
                size_gui=size_gui,
                center_gui=center_gui,
                name_gui=selected_name_gui,
            )
        finally:
            suppress_gui_callbacks["value"] = False
            suppress_controls_callbacks["value"] = False

    def _next_default_name() -> str:
        while True:
            index = next_box_index["value"]
            next_box_index["value"] += 1
            candidate = f"box_{index}"
            if candidate not in boxes:
                return candidate

    def _make_handle(
        box_name: str,
        center: np.ndarray,
        size: np.ndarray,
        rpy: np.ndarray,
        color: tuple[int, int, int],
    ) -> viser.BoxHandle:
        return server.scene.add_box(
            f"/env/{box_name}",
            color=color,
            dimensions=(1.0, 1.0, 1.0),
            opacity=0.35,
            position=tuple(center.tolist()),
            wxyz=_box_wxyz(rpy),
            scale=tuple(size.tolist()),
        )

    def _bind_click(box_name: str, handle: viser.BoxHandle) -> None:
        @handle.on_click
        def _(_) -> None:
            _refresh_selection(box_name)

    def _add_box(
        *,
        name: str | None = None,
        center: np.ndarray | None = None,
        size: np.ndarray | None = None,
        rpy: np.ndarray | None = None,
    ) -> EditableBox:
        index = next_box_index["value"]
        box_name = _next_default_name() if name is None else name
        box_center = (
            np.array((0.4, 0.0, 0.1 + 0.05 * index), dtype=np.float64)
            if center is None
            else np.asarray(center, dtype=np.float64)
        )
        box_size = (
            np.array((0.1, 0.1, 0.1), dtype=np.float64)
            if size is None
            else np.asarray(size, dtype=np.float64)
        )
        box_rpy = np.zeros(3, dtype=np.float64) if rpy is None else np.asarray(rpy, dtype=np.float64)
        color = _make_color(index)

        handle = _make_handle(box_name, box_center, box_size, box_rpy, color)
        editable = EditableBox(
            name=box_name,
            center=box_center,
            size=box_size,
            rpy=box_rpy,
            color=color,
            handle=handle,
        )
        boxes[box_name] = editable
        _bind_click(box_name, handle)
        _refresh_collision_visuals()
        return editable

    existing = BoxEnvironment.load(args.env)
    for box in existing.boxes:
        _add_box(name=box.name, center=box.center, size=box.size, rpy=box.rpy)

    if boxes:
        _refresh_selection(next(iter(boxes)))
    _refresh_collision_visuals()

    @add_button.on_click
    def _(_) -> None:
        editable = _add_box()
        _refresh_selection(editable.name)

    @delete_button.on_click
    def _(_) -> None:
        name = active_name["value"]
        if name is None:
            return
        boxes[name].handle.remove()
        del boxes[name]
        _refresh_selection(next(iter(boxes)) if boxes else None)
        _refresh_collision_visuals()

    @export_button.on_click
    def _(_) -> None:
        saved = _box_environment().save(args.env)
        print(f"Saved box environment to {saved}")

    @controls.on_update
    def _(_) -> None:
        if suppress_controls_callbacks["value"]:
            return
        name = active_name["value"]
        if name is None:
            return
        center = np.asarray(controls.position, dtype=np.float64)
        box = boxes[name]
        box.center = center
        box.handle.position = tuple(center.tolist())
        suppress_gui_callbacks["value"] = True
        try:
            center_gui.value = tuple(center.tolist())
        finally:
            suppress_gui_callbacks["value"] = False
        _refresh_collision_visuals()

    @center_gui.on_update
    def _(_) -> None:
        if suppress_gui_callbacks["value"]:
            return
        name = active_name["value"]
        if name is None:
            return
        center = np.asarray(center_gui.value, dtype=np.float64)
        box = boxes[name]
        box.center = center
        box.handle.position = tuple(center.tolist())
        suppress_controls_callbacks["value"] = True
        try:
            controls.position = tuple(center.tolist())
        finally:
            suppress_controls_callbacks["value"] = False
        _refresh_collision_visuals()

    @size_gui.on_update
    def _(_) -> None:
        if suppress_gui_callbacks["value"]:
            return
        name = active_name["value"]
        if name is None:
            return
        size = np.asarray(size_gui.value, dtype=np.float64)
        box = boxes[name]
        box.size = size
        box.handle.scale = tuple(size.tolist())
        _refresh_collision_visuals()

    def _update_robot_cfg() -> None:
        q = _current_q()
        viser_urdf.update_cfg(q)
        _refresh_collision_visuals()

    for slider in joint_sliders:
        @slider.on_update
        def _(_) -> None:
            _update_robot_cfg()

    print(f"URDF: {urdf_path}")
    print(f"Environment file: {args.env.resolve()}")
    print("Open the viser URL shown above to edit environment boxes.")

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


def main() -> int:
    return run(tyro.cli(Args))


if __name__ == "__main__":
    raise SystemExit(main())
