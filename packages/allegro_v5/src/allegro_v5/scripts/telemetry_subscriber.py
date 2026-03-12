"""
Simple telemetry subscriber for allegro_v5 multiprocessing server.

Usage:
  python -m allegro_v5.scripts.telemetry_subscriber --socket-path /tmp/allegro_v5.sock
"""

import argparse
import time

from allegro_v5.client import DEFAULT_SOCKET_PATH
from allegro_v5.telemetry import TelemetryConfig, ZmqTelemetryClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", type=str, default=DEFAULT_SOCKET_PATH, help="multiprocessing server socket path")
    parser.add_argument("--hz", type=float, default=10.0, help="Print rate in Hz, default: 10.0")
    args = parser.parse_args()

    telem = ZmqTelemetryClient(TelemetryConfig(socket_path=args.socket_path))

    i = 0
    interval = 1.0 / args.hz if args.hz > 0 else 0.0
    try:
        while True:
            msg = telem.recv_latest()
            pos = msg.get("position", [])
            pos_fmt = [f"{v:.3f}" for v in pos]
            tactile = msg.get("tactile", [])
            temp = msg.get("temperature", [])
            imu = msg.get("imu_rpy", [])
            imu_fmt = [f"{float(v):.2f}" for v in imu]
            print(f"[{i}] frame={msg.get('frame')} motion={msg.get('motion')} pos={pos_fmt} tactile={tactile} temp={temp} imu={imu_fmt}")
            i += 1
            if interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        telem.close()


if __name__ == "__main__":
    main()
