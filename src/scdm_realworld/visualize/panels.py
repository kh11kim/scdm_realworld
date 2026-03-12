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
    ) -> None:
        with server.gui.add_folder("Move arm"):
            self.home_button = server.gui.add_button("Home")
            self.current_button = server.gui.add_button("Current")
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
    ) -> None:
        with server.gui.add_folder("Move hand"):
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


class HealthPanel:
    def __init__(self, server: viser.ViserServer) -> None:
        with server.gui.add_folder("Health"):
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
