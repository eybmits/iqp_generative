#!/usr/bin/env python3
"""Claim 2 runner: Budget Law."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "outputs" / "claims" / "claim02_budget_law"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Claim 2 (Budget Law).")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["IQP_EXP02_OUTDIR"] = str(outdir)

    cmd = [sys.executable, str(ROOT / "experiments" / "legacy" / "exp02_budget_law.py")]
    print("[Claim 2] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
