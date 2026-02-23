#!/usr/bin/env python3
"""Claim 10 runner: Global ROI visibility predictor vs discovery."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim10_global_visibility_predicts_discovery"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 10 (global visibility predictor vs discovery).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "legacy" / "exp48_global_visibility_predicts_discovery.py"),
        "--outdir",
        str(outdir),
        *passthrough,
    ]
    print("[Claim 10] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
