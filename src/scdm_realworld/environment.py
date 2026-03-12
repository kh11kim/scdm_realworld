from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from scdm_realworld.utils.geometry import rpy_to_matrix


DEFAULT_BOX_ENV_PATH = Path("assets/box_env.yaml")


@dataclass(frozen=True)
class Box:
    name: str
    center: np.ndarray
    size: np.ndarray
    rpy: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float64))
        object.__setattr__(self, "size", np.asarray(self.size, dtype=np.float64))
        object.__setattr__(self, "rpy", np.asarray(self.rpy, dtype=np.float64))
        if self.center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {self.center.shape}")
        if self.size.shape != (3,):
            raise ValueError(f"size must have shape (3,), got {self.size.shape}")
        if self.rpy.shape != (3,):
            raise ValueError(f"rpy must have shape (3,), got {self.rpy.shape}")

    @property
    def half_extents(self) -> np.ndarray:
        return self.size * 0.5

    @cached_property
    def rotation_wb(self) -> np.ndarray:
        return rpy_to_matrix(float(self.rpy[0]), float(self.rpy[1]), float(self.rpy[2]))

    @cached_property
    def rotation_bw(self) -> np.ndarray:
        return self.rotation_wb.T

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "rpy": self.rpy.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Box:
        return cls(
            name=str(payload["name"]),
            center=np.asarray(payload["center"], dtype=np.float64),
            size=np.asarray(payload["size"], dtype=np.float64),
            rpy=np.asarray(payload.get("rpy", (0.0, 0.0, 0.0)), dtype=np.float64),
        )


class BoxEnvironment:
    def __init__(self, boxes: list[Box] | None = None) -> None:
        self._boxes = [] if boxes is None else list(boxes)

    @property
    def boxes(self) -> tuple[Box, ...]:
        return tuple(self._boxes)

    def add_box(self, box: Box) -> None:
        self._boxes.append(box)

    def clear(self) -> None:
        self._boxes.clear()

    def to_dict(self) -> dict[str, object]:
        return {"boxes": [box.to_dict() for box in self._boxes]}

    def save(self, path: str | Path = DEFAULT_BOX_ENV_PATH) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BOX_ENV_PATH) -> BoxEnvironment:
        input_path = Path(path)
        if not input_path.exists():
            return cls()
        payload = yaml.safe_load(input_path.read_text(encoding="utf-8"))
        boxes_payload = [] if payload is None else payload.get("boxes", [])
        return cls(boxes=[Box.from_dict(item) for item in boxes_payload])
