#!/usr/bin/env python3
"""Claim 9 runner: Expected visibility scaling for random ROI holdouts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim09_expected_visibility_scaling"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 9 (expected visibility scaling).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "legacy" / "exp47_expected_visibility_scaling.py"),
        "--outdir",
        str(outdir),
        *passthrough,
    ]
    print("[Claim 9] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
