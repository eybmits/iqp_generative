#!/usr/bin/env python3
"""Claim 5 runner: IQP > strong classical baselines on high-score holdout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim05_iqp_vs_strong_baselines_high_score"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Claim 5 (high-score holdout, strong classical baselines)."
    )
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    unused_global = outdir / "_unused_global"
    unused_global.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "legacy" / "exp10_strong_classical_recovery.py"),
        "--modes",
        "high_value",
        "--claim6-dir",
        str(outdir),
        "--claim7-dir",
        str(unused_global),
        *passthrough,
    ]
    print("[Claim 5] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
