#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 12: IQP sigma-K ablation across beta/train-m/holdout domains.

Goal:
- Determine one robust (sigma, K) pair for IQP-QCBM parity loss across:
  - beta in [0.7, 1.2]
  - train_m in {200, 1000, 5000}
  - holdout_mode in {global, high_value}

Outputs:
- ablation_metrics_long.csv
- ablation_summary_by_combo.csv
- ablation_summary_by_mode.csv
- heatmap_mean_rank_Q80.pdf
- heatmap_winrate_Q80.pdf
- heatmap_mean_qH_ratio.pdf
- best_sigma_k_report.md
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _build_holdout(
    holdout_mode: str,
    p_star: np.ndarray,
    support: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    m_train_for_holdout: int,
    holdout_k: int,
    holdout_pool: int,
    seed: int,
) -> np.ndarray:
    if holdout_mode == "global":
        candidate = support.astype(bool)
    elif holdout_mode == "high_value":
        candidate = good_mask
    else:
        raise ValueError(f"Unsupported holdout_mode: {holdout_mode}")
    return hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate,
        bits_table=bits_table,
        m_train=m_train_for_holdout,
        holdout_k=holdout_k,
        pool_size=holdout_pool,
        seed=seed + 111,
    )


def _plot_heatmap(
    outpath: Path,
    sigmas: List[float],
    Ks: List[int],
    values: Dict[Tuple[float, int], float],
    title: str,
    cbar_label: str,
    fmt: str = "{:.3f}",
) -> None:
    mat = np.full((len(sigmas), len(Ks)), np.nan, dtype=np.float64)
    for i, s in enumerate(sigmas):
        for j, k in enumerate(Ks):
            if (s, k) in values:
                mat[i, j] = float(values[(s, k)])

    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.8), constrained_layout=True)
    im = ax.imshow(mat, cmap="magma_r", aspect="auto")
    ax.set_xticks(np.arange(len(Ks)))
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_yticks(np.arange(len(sigmas)))
    ax.set_yticklabels([f"{s:g}" for s in sigmas])
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title(title)

    for i in range(len(sigmas)):
        for j in range(len(Ks)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(float(v)), ha="center", va="center", fontsize=7)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cbar_label)
    fig.savefig(outpath)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "32_claim_iqp_sigma_k_ablation_b07to12"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--betas", type=str, default="0.7,0.8,0.9,1.0,1.1,1.2")
    ap.add_argument("--train-ms", type=str, default="200,1000,5000")
    ap.add_argument("--holdout-modes", type=str, default="global,high_value")
    ap.add_argument("--sigmas", type=str, default="0.5,1.0,2.0,3.0")
    ap.add_argument("--Ks", type=str, default="128,256,512")
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    ap.add_argument("--iqp-steps", type=int, default=400)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--layers", type=int, default=1)
    args = ap.parse_args()

    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")

    hv.set_style(base=8)
    outdir = Path(_ensure_outdir(args.outdir))

    betas = _parse_list_floats(args.betas)
    train_ms = _parse_list_ints(args.train_ms)
    holdout_modes = [x.strip() for x in args.holdout_modes.split(",") if x.strip()]
    sigmas = _parse_list_floats(args.sigmas)
    Ks = _parse_list_ints(args.Ks)

    bits_table = hv.make_bits_table(args.n)

    long_rows: List[Dict[str, float]] = []
    total = len(holdout_modes) * len(train_ms) * len(betas) * len(sigmas) * len(Ks)
    run_idx = 0

    for holdout_mode in holdout_modes:
        for beta in betas:
            p_star, support, scores = hv.build_target_distribution_paper(args.n, beta)
            good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))
            holdout_mask = _build_holdout(
                holdout_mode=holdout_mode,
                p_star=p_star,
                support=support,
                good_mask=good_mask,
                bits_table=bits_table,
                m_train_for_holdout=int(args.holdout_m_train),
                holdout_k=int(args.holdout_k),
                holdout_pool=int(args.holdout_pool),
                seed=int(args.seed),
            )

            N = p_star.size
            H_size = int(np.sum(holdout_mask))
            q_unif = np.ones(N, dtype=np.float64) / N
            qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0
            p_star_holdout = float(p_star[holdout_mask].sum())

            for train_m in train_ms:
                for sigma in sigmas:
                    for K in Ks:
                        run_idx += 1
                        print(
                            f"[Run {run_idx}/{total}] mode={holdout_mode} beta={beta:g} m={train_m} "
                            f"sigma={sigma:g} K={K}"
                        )

                        cfg = hv.Config(
                            n=int(args.n),
                            beta=float(beta),
                            train_m=int(train_m),
                            holdout_k=int(args.holdout_k),
                            holdout_pool=int(args.holdout_pool),
                            seed=int(args.seed),
                            good_frac=float(args.good_frac),
                            sigmas=[float(sigma)],
                            Ks=[int(K)],
                            Qmax=10000,
                            Q80_thr=float(args.q80_thr),
                            Q80_search_max=int(args.q80_search_max),
                            target_family="paper_even",
                            adversarial=False,
                            use_iqp=True,
                            use_classical=False,
                            iqp_steps=int(args.iqp_steps),
                            iqp_lr=float(args.iqp_lr),
                            iqp_eval_every=int(args.iqp_eval_every),
                            iqp_layers=int(args.layers),
                            iqp_loss="parity_mse",
                            outdir=str(outdir),
                        )

                        art = hv.rerun_single_setting(
                            cfg=cfg,
                            p_star=p_star,
                            holdout_mask=holdout_mask,
                            bits_table=bits_table,
                            sigma=float(sigma),
                            K=int(K),
                            return_hist=False,
                            iqp_loss="parity_mse",
                        )
                        q_iqp = art["q_iqp"]
                        assert isinstance(q_iqp, np.ndarray)
                        met = hv.compute_metrics_for_q(
                            q=q_iqp,
                            holdout_mask=holdout_mask,
                            qH_unif=qH_unif,
                            H_size=H_size,
                            Q80_thr=float(args.q80_thr),
                            Q80_search_max=int(args.q80_search_max),
                        )

                        long_rows.append(
                            {
                                "holdout_mode": holdout_mode,
                                "beta": float(beta),
                                "train_m": int(train_m),
                                "sigma": float(sigma),
                                "K": int(K),
                                "qH": float(met["qH"]),
                                "qH_ratio": float(met["qH_ratio"]),
                                "R_Q1000": float(met["R_Q1000"]),
                                "R_Q10000": float(met["R_Q10000"]),
                                "Q80": float(met["Q80"]),
                                "Q80_pred": float(met["Q80_pred"]),
                                "holdout_size": int(H_size),
                                "p_star_holdout": float(p_star_holdout),
                                "seed": int(args.seed),
                                "iqp_steps": int(args.iqp_steps),
                            }
                        )

    df = pd.DataFrame(long_rows)
    long_csv = outdir / "ablation_metrics_long.csv"
    df.to_csv(long_csv, index=False)

    # Per-task ranks
    df = df.copy()
    df["Q80_for_rank"] = df["Q80"].replace([np.inf, -np.inf], np.nan)
    finite_max = float(np.nanmax(df["Q80_for_rank"].values)) if np.isfinite(np.nanmax(df["Q80_for_rank"].values)) else 1e9
    df["Q80_for_rank"] = df["Q80_for_rank"].fillna(finite_max * 10.0)
    task_cols = ["holdout_mode", "beta", "train_m"]
    df["rank_Q80"] = df.groupby(task_cols)["Q80_for_rank"].rank(method="min", ascending=True)
    df["rank_qH_ratio"] = df.groupby(task_cols)["qH_ratio"].rank(method="min", ascending=False)
    df["win_Q80"] = (df["rank_Q80"] == 1.0).astype(int)

    summary = (
        df.groupby(["sigma", "K"], as_index=False)
        .agg(
            mean_Q80=("Q80_for_rank", "mean"),
            median_Q80=("Q80_for_rank", "median"),
            mean_rank_Q80=("rank_Q80", "mean"),
            mean_rank_qH_ratio=("rank_qH_ratio", "mean"),
            mean_qH_ratio=("qH_ratio", "mean"),
            wins_Q80=("win_Q80", "sum"),
        )
        .sort_values(["mean_rank_Q80", "mean_Q80", "mean_rank_qH_ratio"], ascending=[True, True, True])
    )
    n_tasks = int(df[task_cols].drop_duplicates().shape[0])
    summary["winrate_Q80"] = summary["wins_Q80"] / max(1, n_tasks)
    summary_csv = outdir / "ablation_summary_by_combo.csv"
    summary.to_csv(summary_csv, index=False)

    by_mode = (
        df.groupby(["holdout_mode", "sigma", "K"], as_index=False)
        .agg(
            mean_rank_Q80=("rank_Q80", "mean"),
            mean_Q80=("Q80_for_rank", "mean"),
            mean_qH_ratio=("qH_ratio", "mean"),
            wins_Q80=("win_Q80", "sum"),
        )
        .sort_values(["holdout_mode", "mean_rank_Q80", "mean_Q80"])
    )
    by_mode_csv = outdir / "ablation_summary_by_mode.csv"
    by_mode.to_csv(by_mode_csv, index=False)

    best = summary.iloc[0].to_dict()

    # Heatmaps
    _plot_heatmap(
        outpath=outdir / "heatmap_mean_rank_Q80.pdf",
        sigmas=sigmas,
        Ks=Ks,
        values={(float(r["sigma"]), int(r["K"])): float(r["mean_rank_Q80"]) for _, r in summary.iterrows()},
        title=r"Mean rank by $Q_{80}$ (lower is better)",
        cbar_label=r"Mean rank($Q_{80}$)",
    )
    _plot_heatmap(
        outpath=outdir / "heatmap_winrate_Q80.pdf",
        sigmas=sigmas,
        Ks=Ks,
        values={(float(r["sigma"]), int(r["K"])): float(r["winrate_Q80"]) for _, r in summary.iterrows()},
        title=r"$Q_{80}$ win-rate across tasks (higher is better)",
        cbar_label=r"Win-rate($Q_{80}$)",
    )
    _plot_heatmap(
        outpath=outdir / "heatmap_mean_qH_ratio.pdf",
        sigmas=sigmas,
        Ks=Ks,
        values={(float(r["sigma"]), int(r["K"])): float(r["mean_qH_ratio"]) for _, r in summary.iterrows()},
        title=r"Mean holdout mass ratio $q(H)/q_{\mathrm{unif}}(H)$",
        cbar_label=r"Mean $q_H$ ratio",
    )

    report = outdir / "best_sigma_k_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# IQP sigma-K ablation report\n\n")
        f.write("Selection rule: minimal mean rank by Q80 (tie-break: lower mean Q80, then better qH-ratio rank).\n\n")
        f.write(f"- Tasks: {n_tasks} = |holdout_modes| x |betas| x |train_m|\n")
        f.write(f"- Holdout modes: {holdout_modes}\n")
        f.write(f"- Betas: {betas}\n")
        f.write(f"- train_m: {train_ms}\n")
        f.write(f"- Sigma grid: {sigmas}\n")
        f.write(f"- K grid: {Ks}\n")
        f.write(f"- IQP steps/lr/eval_every: {args.iqp_steps}/{args.iqp_lr}/{args.iqp_eval_every}\n\n")
        f.write("## Best combination\n\n")
        f.write(f"- sigma: {float(best['sigma']):g}\n")
        f.write(f"- K: {int(best['K'])}\n")
        f.write(f"- mean_rank_Q80: {float(best['mean_rank_Q80']):.4f}\n")
        f.write(f"- mean_Q80: {float(best['mean_Q80']):.2f}\n")
        f.write(f"- mean_qH_ratio: {float(best['mean_qH_ratio']):.4f}\n")
        f.write(f"- wins_Q80: {int(best['wins_Q80'])}/{n_tasks} (win-rate {float(best['winrate_Q80']):.3f})\n")

    print(f"Done. Outputs in {outdir}")
    print(f"Best sigma={float(best['sigma']):g}, K={int(best['K'])}")


if __name__ == "__main__":
    main()

