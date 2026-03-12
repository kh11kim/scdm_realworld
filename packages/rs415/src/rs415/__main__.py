from __future__ import annotations
"""Package entrypoint that delegates execution to the CLI module."""

import sys
from rs415.cli import main as run

def main():
    """Run the rs415 CLI from package entrypoint."""
    return run()

if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
