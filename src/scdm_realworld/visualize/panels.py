from __future__ import annotations

import numpy as np
import viser


class ArmControlPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        *,
        joint_names: tuple[str, ...],
        q_min: np.ndarray,
        q_max: np.ndarray,
        home_q: np.ndarray,
        presets: dict[str, np.ndarray],
    ) -> None:
        with server.gui.add_folder("Move arm"):
            preset_names = ("current",) + tuple(name for name in sorted(presets) if name != "current")
            self.preset_dropdown = server.gui.add_dropdown(
                "Preset",
                preset_names,
                initial_value="current",
            )
            self.set_desired_button = server.gui.add_button("Set Desired")
            self.sliders: list[viser.GuiInputHandle[float]] = []
            for index, joint_name in enumerate(joint_names):
                self.sliders.append(
                    server.gui.add_slider(
                        label=joint_name,
                        min=float(q_min[index]),
                        max=float(q_max[index]),
                        step=1e-3,
                        initial_value=float(home_q[index]),
                    )
                )
            self.plan_button = server.gui.add_button("Plan")
            self.execute_button = server.gui.add_button("Execute", color="red")
            self.goto_button = server.gui.add_button("Goto", color="red")
            self.execution_time_gui = server.gui.add_number(
                "Execution Time",
                initial_value=5.0,
                step=0.1,
            )
            self.traj_slider = server.gui.add_slider(
                "Trajectory Ratio",
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=0.0,
            )
            self.plan_info_text = server.gui.add_text("Plan Info", initial_value="None", disabled=True)

    def desired_q(self) -> np.ndarray:
        return np.asarray([slider.value for slider in self.sliders], dtype=np.float64)

    def set_desired_q(self, q: np.ndarray) -> None:
        for slider, value in zip(self.sliders, np.asarray(q, dtype=np.float64), strict=True):
            slider.value = float(value)

    def clear_plan(self) -> None:
        self.plan_info_text.value = "None"
        self.traj_slider.value = 0.0


class HandControlPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        *,
        joint_names: tuple[str, ...],
        initial_q: np.ndarray,
        presets: dict[str, np.ndarray],
    ) -> None:
        with server.gui.add_folder("Move hand"):
            preset_names = ("current",) + tuple(name for name in sorted(presets) if name != "current")
            self.preset_dropdown = server.gui.add_dropdown(
                "Preset",
                preset_names,
                initial_value="current",
            )
            self.set_desired_button = server.gui.add_button("Set Desired")
            self.sliders: list[viser.GuiInputHandle[float]] = []
            for index, joint_name in enumerate(joint_names):
                self.sliders.append(
                    server.gui.add_slider(
                        label=joint_name,
                        min=-np.pi,
                        max=np.pi,
                        step=1e-3,
                        initial_value=float(initial_q[index]),
                    )
                )
            self.q_vel_gui = server.gui.add_number("q_vel", initial_value=0.5, step=0.05)
            self.goto_button = server.gui.add_button("Goto", color="red")

    def desired_q(self) -> np.ndarray:
        return np.asarray([slider.value for slider in self.sliders], dtype=np.float64)

    def set_desired_q(self, q: np.ndarray) -> None:
        for slider, value in zip(self.sliders, np.asarray(q, dtype=np.float64), strict=True):
            slider.value = float(value)


class SamControlPanel:
    def __init__(self, server: viser.ViserServer) -> None:
        with server.gui.add_folder("Segmentation"):
            self.u_slider = server.gui.add_slider(
                "u",
                min=0,
                max=639,
                step=1,
                initial_value=320,
            )
            self.v_slider = server.gui.add_slider(
                "v",
                min=0,
                max=479,
                step=1,
                initial_value=240,
            )
            self.image = server.gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                label="cam_ext image",
            )
            self.send_button = server.gui.add_button("Send", color="red")
            self.result_image = server.gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                label="mask",
            )
            self.log_text = server.gui.add_text("log", initial_value="", disabled=True)


class GraspControlPanel:
    def __init__(self, server: viser.ViserServer) -> None:
        with server.gui.add_folder("Grasp"):
            self.edge_length = server.gui.add_number(
                "edge length",
                initial_value=0.3,
                step=0.01,
            )
            self.center_offset = server.gui.add_vector3(
                "center offset xyz",
                initial_value=(0.0, 0.0, 0.05),
                step=0.001,
            )
            self.visualize_grid_button = server.gui.add_button("Visualize grid", color="red")
            self.query_grasp_button = server.gui.add_button("Query grasp", color="red")
            self.point_xyz = server.gui.add_vector3(
                "point xyz",
                initial_value=(0.0, 0.0, 0.0),
                step=0.001,
                disabled=True,
            )
            self.log_text = server.gui.add_text("log", initial_value="", disabled=True)


class StatusPanel:
    def __init__(self, server: viser.ViserServer) -> None:
        with server.gui.add_folder("Status", expand_by_default=False):
            self.status_text = server.gui.add_text("Status", initial_value="Running", disabled=True)
            self.real_joints_text = server.gui.add_text("Real Joints", initial_value="", disabled=True)
            self.kinova_ok = server.gui.add_checkbox("kinova", initial_value=False, disabled=True)
            self.kinova_error = server.gui.add_text("kinova_error", initial_value="", disabled=True)
            self.allegro_ok = server.gui.add_checkbox("allegro", initial_value=False, disabled=True)
            self.allegro_error = server.gui.add_text("allegro_error", initial_value="", disabled=True)
            self.cam_ext_ok = server.gui.add_checkbox("cam_ext", initial_value=False, disabled=True)
            self.cam_ext_error = server.gui.add_text("cam_ext_error", initial_value="", disabled=True)
            self.cam_wrist_ok = server.gui.add_checkbox("cam_wrist", initial_value=False, disabled=True)
            self.cam_wrist_error = server.gui.add_text("cam_wrist_error", initial_value="", disabled=True)

    def set_kinova(self, ok: bool, error: str = "") -> None:
        self.kinova_ok.value = bool(ok)
        self.kinova_error.value = error

    def set_allegro(self, ok: bool, error: str = "") -> None:
        self.allegro_ok.value = bool(ok)
        self.allegro_error.value = error

    def set_cam_ext(self, ok: bool, error: str = "") -> None:
        self.cam_ext_ok.value = bool(ok)
        self.cam_ext_error.value = error

    def set_cam_wrist(self, ok: bool, error: str = "") -> None:
        self.cam_wrist_ok.value = bool(ok)
        self.cam_wrist_error.value = error
