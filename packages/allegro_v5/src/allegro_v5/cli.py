from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys

import tyro

from allegro_v5.client import DEFAULT_SOCKET_PATH
from allegro_v5.server import ServerConfig, serve_forever


@dataclass
class ConnectCommand:
    can: str = "can0"
    hand: str = "right"
    tip_type: str = "B"
    bring_up_can: bool = False
    verbose: bool = False
    rep_port: int = 5555
    pub_port: int = 5556
    socket_path: str = DEFAULT_SOCKET_PATH


def _bring_up_can(can: str) -> None:
    commands = [
        ["ip", "link", "set", can, "down"],
        ["ip", "link", "set", can, "type", "can", "bitrate", "1000000"],
        ["ip", "link", "set", can, "up"],
    ]
    for command in commands:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or f"exit code {result.returncode}"
            raise RuntimeError(f"{' '.join(command)} failed: {detail}")

def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] != "connect":
        print("Usage: allegro_v5 connect [options]", file=sys.stderr, flush=True)
        return 1

    command = tyro.cli(ConnectCommand, args=argv[1:])
    try:
        if command.bring_up_can:
            print(
                f"[allegro_v5.cli] bringing up {command.can} with bitrate 1000000",
                flush=True,
            )
            _bring_up_can(command.can)
        return serve_forever(
            ServerConfig(
                can=command.can,
                hand=command.hand,
                tip_type=command.tip_type,
                write=True,
                verbose=command.verbose,
                rep_port=command.rep_port,
                pub_port=command.pub_port,
                socket_path=command.socket_path,
            )
        )
    except KeyboardInterrupt:
        print("[allegro_v5.cli] interrupted", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
