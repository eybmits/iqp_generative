#!/usr/bin/env python3
"""Claim 7 runner: IQP > strong classical baselines on global holdout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim07_iqp_vs_strong_baselines_global"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Claim 7 (global holdout, strong classical baselines)."
    )
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selection_hint = outdir / "_selection_hint"
    selection_hint.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "legacy" / "exp10_strong_classical_recovery.py"),
        "--modes",
        "global",
        "--claim6-dir",
        str(selection_hint),
        "--claim7-dir",
        str(outdir),
        *passthrough,
    ]
    print("[Claim 7] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
