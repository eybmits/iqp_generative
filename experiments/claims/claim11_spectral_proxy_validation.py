#!/usr/bin/env python3
"""Claim 11 runner: Multi-beta spectral-proxy validation for IQP Q80."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim11_spectral_proxy_validation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 11 (spectral-proxy validation over betas and seeds).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "legacy" / "exp50_validate_spectral_proxy_multibeta.py"),
        "--outdir",
        str(outdir),
        *passthrough,
    ]
    print("[Claim 11] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()

