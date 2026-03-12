"""
Allegro Hand teleop UI using viser GUI sliders.

Reads telemetry from the local allegro_v5 multiprocessing server and sends
desired joint positions through the same server. Sliders publish desired joint
angles; a "Hold current" button syncs sliders to the latest measured pose.

Usage:
  python -m allegro_v5.scripts.control_allegro --socket-path /tmp/allegro_v5.sock --urdf right_B
"""

import argparse
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from allegro_v5.client import DEFAULT_SOCKET_PATH, set_desired_positions
from allegro_v5.telemetry import TelemetryConfig, ZmqTelemetryClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", type=str, default=DEFAULT_SOCKET_PATH, help="multiprocessing server socket path")
    parser.add_argument("--urdf", type=str, default="right_B", help="URDF name (right_B/left_B/left_A/right_A; default: right_B)")
    args = parser.parse_args()

    # URDF load
    ROOT = Path(__file__).resolve().parents[3]
    ASSETS_DIR = ROOT / "assets" / "allegro_hand_description"
    URDF_DIR = ASSETS_DIR / "urdf"
    PACKAGE_PREFIX = "package://allegro_hand_description/"

    def resolve_urdf(name: str) -> Path:
        if not name.endswith(".urdf"):
            fname = f"allegro_hand_description_{name}.urdf"
        else:
            fname = name
        path = URDF_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"URDF not found: {path}")
        return path

    def filename_handler(fname):
        s = str(fname)
        if s.startswith(PACKAGE_PREFIX):
            rel = s[len(PACKAGE_PREFIX) :]
            return (ASSETS_DIR / rel).resolve()
        p = Path(fname)
        if not p.is_absolute():
            return (URDF_DIR / p).resolve()
        return p

    urdf_path = resolve_urdf(args.urdf)
    urdf = yourdfpy.URDF.load(urdf_path, filename_handler=filename_handler)
    joint_names = urdf.actuated_joint_names

    telem = ZmqTelemetryClient(TelemetryConfig(socket_path=args.socket_path))

    server = viser.ViserServer()
    viser_urdf = ViserUrdf(
        server,
        urdf_path,
        root_node_name="/allegro",
        load_collision_meshes=False,
    )
    info_text = server.gui.add_markdown(f"**URDF:** {urdf_path.name}\n\nWaiting for state...")

    # GUI elements
    sliders = {}
    with server.gui.add_folder("Joint position control"):
        for jname in joint_names:
            joint = urdf.joint_map[jname]
            central = float((joint.limit.lower + joint.limit.upper) / 2.0)
            slider = server.gui.add_slider(
                label=jname,
                min=joint.limit.lower,
                max=joint.limit.upper,
                step=1e-3,
                initial_value=central,
            )
            sliders[jname] = slider

    hold_button = server.gui.add_button("Hold current pose")

    latest_pos = None
    sliders_synced = False
    last_sent = None

    def send_joint_cmd(desired: np.ndarray):
        nonlocal last_sent
        set_desired_positions(desired.tolist(), socket_path=args.socket_path)
        last_sent = desired.copy()
        print("Sent desired[0:4]:", desired[:4], flush=True)

    @hold_button.on_click
    def _(_evt):
        nonlocal sliders_synced
        if latest_pos is None:
            return
        for j, name in enumerate(joint_names):
            sliders[name].value = float(latest_pos[j])
        send_joint_cmd(latest_pos)
        sliders_synced = True

    try:
        while True:
            msg = telem.recv_latest()
            pos = np.array(msg.get("position", []), dtype=float)
            tactile = np.array(msg.get("tactile", [0, 0, 0, 0]), dtype=np.int32)
            motion = msg.get("motion")
            frame = msg.get("frame")

            if pos.shape[0] != len(joint_names):
                info_text.content = f"Bad joint array len={pos.shape}"
                continue

            latest_pos = pos
            # On first valid telemetry, sync sliders to measured pose so we don't
            # overwrite the robot with default slider values.
            if not sliders_synced:
                for j, name in enumerate(joint_names):
                    sliders[name].value = float(pos[j])
                sliders_synced = True

            info_text.content = (
                f"Frame: {frame} | Motion: {motion} | URDF: {urdf_path.name}\n\n"
                f"Tactile: {tactile.tolist()}"
            )
            viser_urdf.update_cfg(pos)

            # If user moved sliders, send command (only after initial sync).
            if sliders_synced:
                desired = np.array([sliders[n].value for n in joint_names], dtype=float)
                if last_sent is None or not np.allclose(desired, last_sent):
                    send_joint_cmd(desired)

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        telem.close()


if __name__ == "__main__":
    main()
