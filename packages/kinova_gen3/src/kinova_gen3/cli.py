from __future__ import annotations

import collections
import collections.abc
from dataclasses import dataclass
import sys

import tyro

if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence

from kortex_api.Exceptions.KServerException import KServerException

from kinova_gen3.config import (
    KINOVA_IP,
    KINOVA_PASSWORD,
    KINOVA_SOCKET_PATH,
    KINOVA_USERNAME,
)
from kinova_gen3.server import ServerConfig, serve_forever


@dataclass
class ConnectCommand:
    ip: str = KINOVA_IP
    username: str = KINOVA_USERNAME
    password: str = KINOVA_PASSWORD
    socket_path: str = KINOVA_SOCKET_PATH
    dt: float = 0.05
    echo_joints: bool = False


def _run_server(cmd: ConnectCommand) -> int:
    return serve_forever(
        ServerConfig(
            ip=cmd.ip,
            username=cmd.username,
            password=cmd.password,
            socket_path=cmd.socket_path,
            dt=cmd.dt,
            echo_joints=cmd.echo_joints,
        )
    )


def main() -> int:
    try:
        argv = sys.argv[1:]
        if not argv or argv[0] != "connect":
            print("Usage: kinova_gen3 connect [options]")
            return 1
        command = tyro.cli(ConnectCommand, args=argv[1:])
        return _run_server(command)
    except KServerException as exc:
        print(
            "Kortex API error:",
            f"error_code={exc.get_error_code()}",
            f"sub_error_code={exc.get_error_sub_code()}",
        )
        print(str(exc))
        return 1

__all__ = ["ConnectCommand", "main"]
