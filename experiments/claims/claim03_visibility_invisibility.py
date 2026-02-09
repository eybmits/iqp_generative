#!/usr/bin/env python3
"""Claim 3 runner: Visibility / Invisibility mechanism."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim03_visibility_invisibility"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 3 (Visibility / Invisibility).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "legacy" / "exp03_visibility_minvis.py"),
        "--outdir",
        str(outdir),
        *passthrough,
    ]
    print("[Claim 3] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
