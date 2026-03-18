from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml
from PIL import Image


CONFIG_PATH = Path("assets/config.yaml")


def _load_sam3_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config file: {CONFIG_PATH}")
    sam3_cfg = payload.get("sam3", {})
    if not isinstance(sam3_cfg, dict):
        raise ValueError(f"Invalid sam3 config in: {CONFIG_PATH}")
    return sam3_cfg


def _base_url() -> str:
    cfg = _load_sam3_config()
    host = str(cfg.get("host", "127.0.0.1"))
    port = int(cfg.get("port", 8000))
    return f"http://{host}:{port}"


def _timeout_s() -> float:
    cfg = _load_sam3_config()
    return float(cfg.get("timeout_s", 10.0))


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    print(f"[sam3_client] send {path}", flush=True)
    response = requests.post(
        f"{_base_url()}{path}",
        json=payload,
        timeout=_timeout_s(),
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response payload: {type(data).__name__}")
    print(f"[sam3_client] recv {path}", flush=True)
    return data


def _encode_rgb_image(rgb: np.ndarray) -> str:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must have dtype uint8, got {rgb.dtype}")
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def echo(text: str) -> str:
    """Simple connectivity check.

    Before calling this, forward the GPU server port to local 8000, for example:
    `ssh -L 8000:localhost:8000 <user>@<gpu-server>`
    """
    data = _post_json("/echo", {"text": text})
    reply = data.get("reply")
    if not isinstance(reply, str):
        raise RuntimeError(f"Missing 'reply' in response: {data}")
    return reply


def get_seg_mask(rgb: np.ndarray, input_point: list[float]) -> dict[str, Any]:
    """Request a segmentation mask for a point prompt.

    Expected server endpoint: POST /seg_mask
    Payload: {"rgb_b64": "...", "input_point": [x, y]}
    """
    if len(input_point) != 2:
        raise ValueError(f"input_point must have length 2, got {len(input_point)}")
    return _post_json(
        "/seg_mask",
        {
            "rgb_b64": _encode_rgb_image(rgb),
            "input_point": [float(v) for v in input_point],
        },
    )


__all__ = ["echo", "get_seg_mask"]
