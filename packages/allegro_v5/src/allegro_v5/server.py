from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path
import json
import subprocess
import threading
import time

import zmq

from allegro_v5.client import DEFAULT_SOCKET_PATH
from allegro_v5.protocol import (
    AckResponse,
    GetStateRequest,
    SetDesiredPositionRequest,
    StateResponse,
)


@dataclass
class ServerConfig:
    can: str = "can0"
    hand: str = "right"
    tip_type: str = "B"
    write: bool = False
    verbose: bool = False
    rep_port: int = 5555
    pub_port: int = 5556
    socket_path: str = DEFAULT_SOCKET_PATH


class _ZmqTelemetrySubscriber:
    def __init__(self, *, host: str, port: int):
        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.connect(f"tcp://{host}:{port}")
        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def recv_latest(self) -> dict:
        with self._lock:
            if self._latest is None:
                raise TimeoutError("No Allegro telemetry received yet")
            return dict(self._latest)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=0.5)
        self._socket.close()

    def _poll_loop(self) -> None:
        while self._running:
            try:
                msg = self._socket.recv_json()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            with self._lock:
                self._latest = msg


def _binary_path() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "bin" / "allegro_run"


def _run_args(config: ServerConfig) -> list[str]:
    args = [
        str(_binary_path()),
        "--can",
        config.can,
        "--hand",
        config.hand,
        "--type",
        config.tip_type,
        "--rep-port",
        str(config.rep_port),
        "--pub-port",
        str(config.pub_port),
    ]
    if config.write:
        args.append("--write")
    if config.verbose:
        args.append("--verbose")
    return args


def _handle_request(request, telem: _ZmqTelemetrySubscriber, req_socket) -> StateResponse | AckResponse:
    if isinstance(request, GetStateRequest):
        try:
            msg = telem.recv_latest()
        except TimeoutError as exc:
            return StateResponse(error=str(exc))
        return StateResponse(
            frame=int(msg.get("frame", 0)),
            motion=str(msg.get("motion", "")),
            position=list(msg.get("position", [])),
            torque=list(msg.get("torque", [])),
            tactile=list(msg.get("tactile", [])),
            temperature=list(msg.get("temperature", [])),
            imu_rpy=list(msg.get("imu_rpy", [])),
        )

    if isinstance(request, SetDesiredPositionRequest):
        payload = {
            "cmd": "set_joint_command",
            "desired": list(request.desired_position),
        }
        req_socket.send_string(json.dumps(payload))
        reply = req_socket.recv_string()
        try:
            parsed = json.loads(reply)
        except json.JSONDecodeError:
            return AckResponse(ok=False, error=f"invalid reply: {reply}")
        if bool(parsed.get("ok", False)):
            return AckResponse(ok=True)
        return AckResponse(ok=False, error=str(parsed.get("error", "unknown error")))

    return AckResponse(ok=False, error=f"unsupported request: {type(request).__name__}")


def serve_forever(config: ServerConfig) -> int:
    binary_path = _binary_path()
    if not binary_path.exists():
        print(
            "[allegro_v5.server] build/bin/allegro_run not found. Build it first with:\n"
            "  cmake -S . -B build\n"
            "  cmake --build build",
            flush=True,
        )
        return 1

    socket_path = Path(config.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    run_args = _run_args(config)
    print(
        "[allegro_v5.server] starting allegro_run "
        f"can={config.can} hand={config.hand} tip_type={config.tip_type} "
        f"write={config.write} verbose={config.verbose}",
        flush=True,
    )
    print(f"[allegro_v5.server] binary={binary_path}", flush=True)
    if not config.write:
        print(
            "[allegro_v5.server] write is disabled; telemetry/state only. "
            "Use --write to enable torque commands.",
            flush=True,
        )
    process = subprocess.Popen(run_args)

    try:
        time.sleep(0.5)
        if process.poll() is not None:
            print(
                f"[allegro_v5.server] allegro_run exited early with code {process.returncode}",
                flush=True,
            )
            return int(process.returncode)

        telem = _ZmqTelemetrySubscriber(host="localhost", port=config.pub_port)
        ctx = zmq.Context.instance()
        req_socket = ctx.socket(zmq.REQ)
        req_socket.connect(f"tcp://localhost:{config.rep_port}")
        req_socket.setsockopt(zmq.LINGER, 0)
        listener = Listener(str(socket_path), family="AF_UNIX")
        print(f"[allegro_v5.server] listening on {socket_path}", flush=True)
        print("[allegro_v5.server] server loop running. Press Ctrl+C to stop.", flush=True)
        try:
            while True:
                conn = listener.accept()
                try:
                    request = conn.recv()
                    response = _handle_request(request, telem, req_socket)
                    conn.send(response)
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
                finally:
                    conn.close()
        except KeyboardInterrupt:
            return 0
        finally:
            listener.close()
            req_socket.close()
            telem.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if socket_path.exists():
            socket_path.unlink()
