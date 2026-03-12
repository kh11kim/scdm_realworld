from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from scdm_realworld.environment import Box


DEFAULT_SPHERE_SET_PATH = Path("assets/gen3_allegro/gen3_allegro_spherized.yml")


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float
    link_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float64))
        object.__setattr__(self, "radius", float(self.radius))
        if self.center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {self.center.shape}")


def load_link_spheres(
    path: str | Path = DEFAULT_SPHERE_SET_PATH,
) -> dict[str, list[Sphere]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    collision_spheres = payload.get("collision_spheres", {})
    result: dict[str, list[Sphere]] = {}
    for link_name, spheres in collision_spheres.items():
        result[str(link_name)] = [
            Sphere(
                center=np.asarray(item["center"], dtype=np.float64),
                radius=float(item["radius"]),
                link_name=str(link_name),
            )
            for item in spheres
        ]
    return result


def compute_world_spheres(
    link_spheres: dict[str, list[Sphere]],
    link_poses: dict[str, np.ndarray],
) -> list[Sphere]:
    world_spheres: list[Sphere] = []
    for link_name, spheres in link_spheres.items():
        if link_name not in link_poses:
            continue
        transform = np.asarray(link_poses[link_name], dtype=np.float64)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        for sphere in spheres:
            center_world = rotation @ sphere.center + translation
            world_spheres.append(
                Sphere(center=center_world, radius=sphere.radius, link_name=link_name)
            )
    world_spheres.sort(key=lambda sphere: float(np.linalg.norm(sphere.center)), reverse=True)
    return world_spheres


def sphere_box_distance_squared(sphere: Sphere, box: Box) -> float:
    local_center = box.rotation_bw @ (sphere.center - box.center)
    clamped = np.clip(local_center, -box.half_extents, box.half_extents)
    delta = local_center - clamped
    return float(delta @ delta)


def sphere_intersects_box(sphere: Sphere, box: Box) -> bool:
    return sphere_box_distance_squared(sphere, box) <= sphere.radius * sphere.radius


def check_spheres_vs_boxes(
    spheres: list[Sphere],
    boxes: list[Box],
) -> list[tuple[Sphere, Box]]:
    collisions: list[tuple[Sphere, Box]] = []
    for sphere in spheres:
        for box in boxes:
            if sphere_intersects_box(sphere, box):
                collisions.append((sphere, box))
    return collisions


def has_collision(
    spheres: list[Sphere],
    boxes: list[Box],
) -> bool:
    for sphere in spheres:
        for box in boxes:
            if sphere_intersects_box(sphere, box):
                return True
    return False
