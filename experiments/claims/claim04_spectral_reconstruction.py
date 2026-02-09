#!/usr/bin/env python3
"""Claim 4 runner: Spectral reconstruction predicts recovery kinetics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim04_spectral_reconstruction"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 4 (Spectral Reconstruction).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "iqp_generative" / "core.py"),
        "--outdir",
        str(outdir),
        "--target-family",
        "paper_even",
        *passthrough,
    ]
    print("[Claim 4] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()

