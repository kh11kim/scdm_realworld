from __future__ import annotations

from dataclasses import dataclass
import sys

import tyro

from allegro_v5.client import DEFAULT_SOCKET_PATH
from allegro_v5.server import ServerConfig, serve_forever


@dataclass
class ConnectCommand:
    can: str = "can0"
    hand: str = "right"
    tip_type: str = "B"
    verbose: bool = False
    rep_port: int = 5555
    pub_port: int = 5556
    socket_path: str = DEFAULT_SOCKET_PATH

def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] != "connect":
        print("Usage: allegro_v5 connect [options]", file=sys.stderr, flush=True)
        return 1

    command = tyro.cli(ConnectCommand, args=argv[1:])
    try:
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
