from __future__ import annotations

import numpy as np
import viser

from rs415.shm_io import RS415SharedMemoryReader
from scdm_realworld.utils.geometry import matrix_to_wxyz, project_depth_to_world


def _frustum_params_from_intrinsics(intrinsics: dict[str, object]) -> tuple[float, float]:
    width = float(intrinsics["width"])
    height = float(intrinsics["height"])
    fy = float(intrinsics["fy"])
    aspect = width / height
    fov = 2.0 * np.arctan2(height / 2.0, fy)
    return float(fov), float(aspect)


def _pcd_step_from_count(depth_mm: np.ndarray, count: int) -> int:
    valid_count = int(np.count_nonzero(depth_mm > 0))
    if valid_count <= 0:
        return 1
    return max(1, valid_count // count)


class CameraView:
    def __init__(
        self,
        server: viser.ViserServer,
        *,
        label: str,
        reader: RS415SharedMemoryReader | None,
        frame_path: str,
        pcd_path: str,
        color: tuple[int, int, int],
        default_pcd_count: int,
    ) -> None:
        self._server = server
        self._reader = reader
        self._frame_path = frame_path
        self._frustum_path = f"{frame_path}/frustum"
        self._pcd_path = pcd_path
        self._color = color
        self._frame = None
        self._frustum = None
        self._pcd = None
        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_intrinsics: dict[str, object] | None = None

        with server.gui.add_folder(label):
            self.visualize_checkbox = server.gui.add_checkbox("visualize", initial_value=True)
            self.pcd_checkbox = server.gui.add_checkbox("pcd", initial_value=False)
            self.pcd_count_gui = server.gui.add_number(
                "pcd_count",
                initial_value=default_pcd_count,
                step=500,
            )

    def update(self, camera_pose: np.ndarray | None) -> tuple[bool, str]:
        if self._reader is None or camera_pose is None:
            self.clear()
            if self._reader is None:
                return False, "reader unavailable"
            return False, "camera pose unavailable"

        try:
            bundle = self._reader.read(copy=True)
        except Exception as exc:
            self._latest_rgb = None
            self._latest_depth = None
            self._latest_intrinsics = None
            self.clear()
            return False, str(exc)
        intrinsics = bundle.meta.get("intrinsics")
        if not isinstance(intrinsics, dict):
            self._latest_rgb = None
            self._latest_depth = None
            self._latest_intrinsics = None
            self.clear()
            return False, "missing intrinsics"
        self._latest_rgb = bundle.rgb
        self._latest_depth = bundle.depth
        self._latest_intrinsics = dict(intrinsics)

        if self.visualize_checkbox.value:
            self._render_frustum(camera_pose, intrinsics, bundle.rgb)
        else:
            self._remove_frustum()

        if self.pcd_checkbox.value:
            self._render_pcd(camera_pose, intrinsics, bundle.depth, bundle.rgb)
        else:
            self._remove_pcd()
        return True, ""

    def clear(self) -> None:
        self._remove_frustum()
        self._remove_pcd()

    def close(self) -> None:
        self.clear()
        if self._reader is not None:
            self._reader.close()

    @property
    def latest_rgb(self) -> np.ndarray | None:
        if self._latest_rgb is None:
            return None
        return self._latest_rgb.copy()

    @property
    def latest_depth(self) -> np.ndarray | None:
        if self._latest_depth is None:
            return None
        return self._latest_depth.copy()

    @property
    def latest_intrinsics(self) -> dict[str, object] | None:
        if self._latest_intrinsics is None:
            return None
        return dict(self._latest_intrinsics)

    def _render_frustum(
        self,
        camera_pose: np.ndarray,
        intrinsics: dict[str, object],
        rgb: np.ndarray,
    ) -> None:
        self._remove_frustum()
        fov, aspect = _frustum_params_from_intrinsics(intrinsics)
        self._frame = self._server.scene.add_frame(
            self._frame_path,
            axes_length=0.12,
            axes_radius=0.006,
            position=tuple(camera_pose[:3, 3].tolist()),
            wxyz=matrix_to_wxyz(camera_pose[:3, :3]),
        )
        self._frustum = self._server.scene.add_camera_frustum(
            self._frustum_path,
            fov=fov,
            aspect=aspect,
            scale=0.1,
            color=self._color,
            image=rgb,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    def _render_pcd(
        self,
        camera_pose: np.ndarray,
        intrinsics: dict[str, object],
        depth: np.ndarray,
        rgb: np.ndarray,
    ) -> None:
        self._remove_pcd()
        points, colors = project_depth_to_world(
            depth,
            rgb,
            intrinsics,
            camera_pose[:3, :3],
            camera_pose[:3, 3],
            _pcd_step_from_count(depth, int(self.pcd_count_gui.value)),
        )
        self._pcd = self._server.scene.add_point_cloud(
            self._pcd_path,
            points=points,
            colors=colors,
            point_size=0.003,
            point_shape="circle",
        )

    def _remove_frustum(self) -> None:
        if self._frustum is not None:
            self._frustum.remove()
            self._frustum = None
        if self._frame is not None:
            self._frame.remove()
            self._frame = None

    def _remove_pcd(self) -> None:
        if self._pcd is not None:
            self._pcd.remove()
            self._pcd = None
