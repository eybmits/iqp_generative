#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WP4 runner: fair baseline matrix over holdout modes, train-m regimes, beta sweep, and seeds.

This wraps exp11 (strong baseline sweep) per seed and aggregates into per-regime summary CSVs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_modes(s: str) -> List[str]:
    out = []
    for x in s.split(","):
        m = x.strip().lower()
        if not m:
            continue
        if m not in ("global", "high_value"):
            raise ValueError(f"Unsupported holdout mode: {m}")
        out.append(m)
    return out


def _run(cmd: List[str]) -> None:
    cmd_s = " ".join(cmd)
    print(f"[run] {cmd_s}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _iqp_settings(mode: str, global_sigma: float, global_k: int, high_sigma: float, high_k: int) -> Dict[str, float | int]:
    if mode == "global":
        return {"sigma": float(global_sigma), "K": int(global_k)}
    return {"sigma": float(high_sigma), "K": int(high_k)}


def _regime_dir(outroot: Path, mode: str, m: int, sigma: float, k: int) -> Path:
    s_tag = int(round(float(sigma))) if abs(float(sigma) - round(float(sigma))) < 1e-9 else str(sigma).replace(".", "p")
    return outroot / f"{mode}_m{int(m)}_sigma{s_tag}_k{int(k)}"


def _aggregate_regime(regime_dir: Path) -> None:
    seed_csvs = sorted(regime_dir.glob("seed_*/summary/beta_sweep_metrics_long.csv"))
    if not seed_csvs:
        print(f"[warn] no seed CSVs in {regime_dir}")
        return

    frames = []
    for p in seed_csvs:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        frames.append(d)
    if not frames:
        print(f"[warn] no readable seed CSVs in {regime_dir}")
        return

    df = pd.concat(frames, ignore_index=True)
    summary_dir = regime_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_path = summary_dir / "beta_sweep_metrics_long.csv"
    df.to_csv(all_path, index=False)

    group_cols = ["beta", "model_key", "model_label"]
    metrics = ["Q80", "qH", "qH_ratio", "R_Q1000", "R_Q10000", "fit_tv_to_pstar"]
    agg_dict = {}
    for c in metrics:
        if c in df.columns:
            agg_dict[c + "_mean"] = (c, "mean")
            agg_dict[c + "_std"] = (c, "std")
            agg_dict[c + "_median"] = (c, "median")
            agg_dict[c + "_iqr"] = (c, lambda s: float(np.nanquantile(s, 0.75) - np.nanquantile(s, 0.25)))
    agg_dict["n"] = ("Q80", "count")
    summary = df.groupby(group_cols, as_index=False).agg(**agg_dict)
    summary.to_csv(summary_dir / "beta_sweep_metrics_summary.csv", index=False)

    # Paired parity-vs-best-classical rows per (seed,beta)
    if "seed" in df.columns:
        par = df[df["model_key"] == "iqp_parity_mse"][["seed", "beta", "Q80"]].rename(columns={"Q80": "Q80_parity"})
        cls = (
            df[df["model_key"] != "iqp_parity_mse"]
            .groupby(["seed", "beta"], as_index=False)["Q80"]
            .min()
            .rename(columns={"Q80": "Q80_best_classical"})
        )
        cmp_df = pd.merge(par, cls, on=["seed", "beta"], how="inner")
        if not cmp_df.empty:
            cmp_df["Q80_ratio_parity_over_best_classical"] = (
                cmp_df["Q80_parity"] / np.maximum(cmp_df["Q80_best_classical"], 1e-12)
            )
            cmp_df.to_csv(summary_dir / "parity_vs_best_classical_pairs.csv", index=False)

    print(f"[saved] {all_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fair baseline matrix via exp11 and aggregate seeds.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "34_claim_beta_sweep_bestparams"),
    )
    ap.add_argument("--holdout-modes", type=str, default="global,high_value")
    ap.add_argument("--train-ms", type=str, default="200,1000,5000")
    ap.add_argument("--betas", type=str, default="0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")

    ap.add_argument("--global-sigma", type=float, default=1.0)
    ap.add_argument("--global-k", type=int, default=512)
    ap.add_argument("--high-sigma", type=float, default=2.0)
    ap.add_argument("--high-k", type=int, default=256)

    ap.add_argument("--iqp-steps", type=int, default=300)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)

    ap.add_argument("--artr-epochs", type=int, default=300)
    ap.add_argument("--artr-d-model", type=int, default=64)
    ap.add_argument("--artr-heads", type=int, default=4)
    ap.add_argument("--artr-layers", type=int, default=2)
    ap.add_argument("--artr-ff", type=int, default=128)
    ap.add_argument("--artr-lr", type=float, default=1e-3)
    ap.add_argument("--artr-batch-size", type=int, default=256)

    ap.add_argument("--maxent-steps", type=int, default=2500)
    ap.add_argument("--maxent-lr", type=float, default=5e-2)

    ap.add_argument("--fit-metric", type=str, default="tv", choices=["tv", "kl"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    modes = _parse_modes(args.holdout_modes)
    train_ms = _parse_ints(args.train_ms)
    betas = _parse_floats(args.betas)
    seeds = _parse_ints(args.seeds)

    if args.smoke:
        modes = modes[:1]
        train_ms = train_ms[:1]
        betas = betas[:2]
        seeds = seeds[:1]

    for mode in modes:
        params = _iqp_settings(mode, args.global_sigma, args.global_k, args.high_sigma, args.high_k)
        sigma = float(params["sigma"])
        k = int(params["K"])
        betas_str = ",".join(f"{b:g}" for b in betas)

        for train_m in train_ms:
            regime = _regime_dir(outroot, mode, train_m, sigma, k)
            regime.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                seed_out = regime / f"seed_{seed}"
                seed_out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(ROOT / "experiments" / "legacy" / "exp11_beta_sweep_global_holdout.py"),
                    "--outdir",
                    str(seed_out),
                    "--holdout-mode",
                    mode,
                    "--train-m",
                    str(int(train_m)),
                    "--betas",
                    betas_str,
                    "--seed",
                    str(int(seed)),
                    "--sigma",
                    str(float(sigma)),
                    "--K",
                    str(int(k)),
                    "--iqp-steps",
                    str(int(args.iqp_steps)),
                    "--iqp-lr",
                    str(float(args.iqp_lr)),
                    "--iqp-eval-every",
                    str(int(args.iqp_eval_every)),
                    "--artr-epochs",
                    str(int(args.artr_epochs)),
                    "--artr-d-model",
                    str(int(args.artr_d_model)),
                    "--artr-heads",
                    str(int(args.artr_heads)),
                    "--artr-layers",
                    str(int(args.artr_layers)),
                    "--artr-ff",
                    str(int(args.artr_ff)),
                    "--artr-lr",
                    str(float(args.artr_lr)),
                    "--artr-batch-size",
                    str(int(args.artr_batch_size)),
                    "--maxent-steps",
                    str(int(args.maxent_steps)),
                    "--maxent-lr",
                    str(float(args.maxent_lr)),
                    "--fit-metric",
                    str(args.fit_metric),
                ]
                _run(cmd)

            _aggregate_regime(regime)

    print(f"[saved] {outroot}")


if __name__ == "__main__":
    main()
