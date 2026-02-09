#!/usr/bin/env python3
"""Claim 8 runner: IQP > strong classical baselines across beta sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim08_iqp_vs_strong_baselines_beta_sweep"


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_modes(s: str) -> List[str]:
    modes: List[str] = []
    for x in s.split(","):
        mode = x.strip().lower()
        if not mode:
            continue
        if mode not in ("global", "high_value"):
            raise ValueError(f"Unsupported holdout mode: {mode}")
        modes.append(mode)
    if not modes:
        raise ValueError("No valid holdout modes provided.")
    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 8 (beta sweep).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--betas", type=str, default="0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4")
    parser.add_argument("--train-ms", type=str, default="200,1000,5000")
    parser.add_argument("--holdout-modes", type=str, default="global,high_value")
    args, passthrough = parser.parse_known_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_ms = _parse_list_ints(args.train_ms)
    holdout_modes = _parse_modes(args.holdout_modes)

    for holdout_mode in holdout_modes:
        for train_m in train_ms:
            run_outdir = outdir / f"{holdout_mode}_m{train_m}"
            run_outdir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(ROOT / "experiments" / "legacy" / "exp11_beta_sweep_global_holdout.py"),
                "--outdir",
                str(run_outdir),
                "--holdout-mode",
                holdout_mode,
                "--train-m",
                str(train_m),
                "--betas",
                args.betas,
                *passthrough,
            ]
            print("[Claim 8] Running:", " ".join(cmd))
            subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
