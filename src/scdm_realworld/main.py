from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
import tyro

from scdm_realworld.env_editor_app import Args as EnvArgs
from scdm_realworld.env_editor_app import run as run_env_editor_app
from scdm_realworld.main_app import AppConfig, MainApp
from scdm_realworld.runtime_config import (
    DEFAULT_APP_CONFIG_PATH,
    load_runtime_config,
    resolve_arm_home_q,
    resolve_arm_api,
    resolve_hand_home_q,
    resolve_hand_api,
    resolve_robot_urdf,
)
from scdm_realworld.system_calibrate_app import Args as CalibrateArgs
from scdm_realworld.system_calibrate_app import run as run_calibrate_app


@dataclass
class RunArgs:
    app_config: Path = DEFAULT_APP_CONFIG_PATH
    urdf: Path | None = None
    env: Path = Path("assets/box_env.yaml")
    system_calibration: Path = Path("assets/system_calibration.yaml")
    host: str = "0.0.0.0"
    port: int = 8080
    scale: float = 1.0
    dt: float = 0.1


Command = Annotated[
    RunArgs,
    tyro.conf.subcommand(name="run"),
] | Annotated[
    CalibrateArgs,
    tyro.conf.subcommand(name="calibrate"),
] | Annotated[
    EnvArgs,
    tyro.conf.subcommand(name="env"),
]


def main() -> int:
    args = tyro.cli(Command)
    if isinstance(args, CalibrateArgs):
        return run_calibrate_app(args)
    if isinstance(args, EnvArgs):
        return run_env_editor_app(args)
    runtime_config = load_runtime_config(args.app_config)
    urdf = resolve_robot_urdf(runtime_config) if args.urdf is None else args.urdf
    app = MainApp(
        AppConfig(
            urdf=urdf.resolve(),
            env=args.env.resolve(),
            system_calibration=args.system_calibration.resolve(),
            arm_api=resolve_arm_api(runtime_config),
            hand_api=resolve_hand_api(runtime_config),
            arm_home_q=resolve_arm_home_q(runtime_config),
            hand_home_q=resolve_hand_home_q(runtime_config),
            host=args.host,
            port=args.port,
            scale=args.scale,
            dt=args.dt,
        )
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
