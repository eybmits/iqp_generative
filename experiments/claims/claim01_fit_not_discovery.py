#!/usr/bin/env python3
"""Claim 1 runner: Fit =/= Discovery."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim01_fit_not_discovery"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 1 (Fit =/= Discovery).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "legacy" / "exp05_discovery_axis.py"),
        "--outdir",
        str(outdir),
        *passthrough,
    ]
    print("[Claim 1] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
