#!/usr/bin/env python3
"""Claim 6 runner: fair baseline protocol (paired IQP vs classical controls)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim06_fair_baseline_protocol"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 6 (fair baseline protocol).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "legacy" / "exp09_fair_baseline_global_holdout.py"),
        "--outdir",
        str(outdir),
        *passthrough,
    ]
    print("[Claim 6] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
