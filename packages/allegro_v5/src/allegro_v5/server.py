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
        self._last_warning_time = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def recv_latest(self) -> dict:
        with self._lock:
            if self._latest is None:
                raise TimeoutError("No Allegro telemetry received yet")
            return dict(self._latest)

    def has_latest(self) -> bool:
        with self._lock:
            return self._latest is not None

    def should_warn(self, *, interval_s: float = 2.0) -> bool:
        now = time.monotonic()
        with self._lock:
            if self._latest is not None:
                return False
            if now - self._last_warning_time < interval_s:
                return False
            self._last_warning_time = now
            return True

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
            if telem.should_warn():
                print(
                    "[allegro_v5.server] no telemetry received yet; "
                    "check hand power/CAN connection and that the hand is initialized.",
                    flush=True,
                )
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
        print(f"[allegro_v5.server] request {type(request).__name__}", flush=True)
        payload = {
            "cmd": "set_joint_command",
            "desired": list(request.desired_position),
        }
        print("[allegro_v5.server] -> allegro_run set_joint_command", flush=True)
        req_socket.send_string(json.dumps(payload))
        reply = req_socket.recv_string()
        try:
            parsed = json.loads(reply)
        except json.JSONDecodeError:
            return AckResponse(ok=False, error=f"invalid reply: {reply}")
        if bool(parsed.get("ok", False)):
            response = AckResponse(ok=True)
            print(f"[allegro_v5.server] response {type(response).__name__}(ok=True)", flush=True)
            return response
        response = AckResponse(ok=False, error=str(parsed.get("error", "unknown error")))
        print(
            f"[allegro_v5.server] response {type(response).__name__}(ok=False, error={response.error})",
            flush=True,
        )
        return response

    response = AckResponse(ok=False, error=f"unsupported request: {type(request).__name__}")
    print(
        f"[allegro_v5.server] response {type(response).__name__}(ok=False, error={response.error})",
        flush=True,
    )
    return response


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
        startup_deadline = time.monotonic() + 2.0
        while time.monotonic() < startup_deadline:
            if telem.has_latest():
                break
            time.sleep(0.05)
        if not telem.has_latest():
            print(
                "[allegro_v5.server] warning: no telemetry received after startup; "
                "state reads will fail until the hand starts publishing.",
                flush=True,
            )
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
