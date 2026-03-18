from __future__ import annotations

import base64
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


SAM3_ROOT = Path("/home/irsl/kkh_ws/sam3")
SAM3_SRC_ROOT = SAM3_ROOT / "src" / "sam3"
DEFAULT_IMAGE_PATH = SAM3_ROOT / "assets" / "images" / "truck.jpg"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


class EchoRequest(BaseModel):
    text: str


class SegMaskRequest(BaseModel):
    rgb_b64: str
    input_point: list[float]


@dataclass
class ConnectCommand:
    """SAM3 FastAPI 서버 실행."""


Command = Annotated[
    ConnectCommand,
    tyro.conf.subcommand(name="connect"),
]


def _ensure_sam3_importable() -> None:
    sam3_python_root = SAM3_ROOT / "src"
    if str(sam3_python_root) not in sys.path:
        sys.path.insert(0, str(sam3_python_root))


def _decode_rgb_image(rgb_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(rgb_b64.encode("ascii"))
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def _load_runtime() -> tuple[FastAPI, object, object]:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe_path = SAM3_SRC_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model)

    app = FastAPI()

    @app.post("/echo")
    async def echo(request: EchoRequest) -> dict[str, str]:
        print("[sam_server] recv /echo", flush=True)
        return {"reply": f"SERVER ECHO: {request.text.upper()}"}

    @app.post("/seg_mask")
    async def seg_mask(request: SegMaskRequest) -> dict[str, object]:
        print("[sam_server] recv /seg_mask", flush=True)
        if len(request.input_point) != 2:
            return {"error": "input_point must have length 2"}

        image = _decode_rgb_image(request.rgb_b64)
        inference_state = processor.set_image(image)
        input_point = np.array([[float(request.input_point[0]), float(request.input_point[1])]])
        input_label = np.array([1])

        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]

        best_mask = masks[0].astype(np.uint8)
        best_score = float(scores[0])
        print("[sam_server] send /seg_mask", flush=True)
        return {
            "mask": best_mask.tolist(),
            "score": best_score,
        }

    return app, model, processor


def _run_connect(_: ConnectCommand) -> int:
    _ensure_sam3_importable()

    if not DEFAULT_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"SAM3 example image not found: {DEFAULT_IMAGE_PATH}. "
            "서버 머신에서 sam3 경로 하드코딩을 확인해 주세요."
        )

    print("[sam_server] starting SAM3 server", flush=True)
    print(f"[sam_server] sam3_root={SAM3_ROOT}", flush=True)
    print(f"[sam_server] image={DEFAULT_IMAGE_PATH}", flush=True)
    print(
        "[sam_server] run this on the GPU server and expose it with "
        f"`uv run sam_server connect` on port {DEFAULT_PORT}",
        flush=True,
    )

    app, _, _ = _load_runtime()
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
    return 0


def main() -> int:
    command = tyro.cli(Command)
    return _run_connect(command)


if __name__ == "__main__":
    raise SystemExit(main())
