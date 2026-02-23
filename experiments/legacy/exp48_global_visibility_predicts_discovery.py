#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global ROI visibility predictor vs holdout discovery (Q80 primary).

Practical check for proposition-aligned predictor:
  q_lin_pred(H) = |H|/2^n + (|H|/|G|) * Vis_B(G), with G = S_even (global ROI).

For each (beta, seed), several random holdouts H subset G are sampled uniformly.
We compare the holdout-free predictor against:
  - realized q_lin(H) on random holdouts
  - IQP discovery metrics (Q80 primary, R(Q_eval) secondary)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _rankdata(x: np.ndarray) -> np.ndarray:
    idx = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(idx, dtype=np.float64)
    ranks[idx] = np.arange(1, x.size + 1, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    xm = x[mask]
    ym = y[mask]
    if float(np.std(xm)) < 1e-15 or float(np.std(ym)) < 1e-15:
        return float("nan")
    rx = _rankdata(xm)
    ry = _rankdata(ym)
    return float(np.corrcoef(rx, ry)[0, 1])


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sample_uniform_holdout_mask(
    rng: np.random.Generator,
    roi_indices: np.ndarray,
    holdout_k: int,
    N: int,
) -> np.ndarray:
    roi_idx = np.asarray(roi_indices, dtype=int)
    if holdout_k <= 0 or holdout_k > int(roi_idx.size):
        raise ValueError(f"Invalid holdout_k={holdout_k} for roi size={roi_idx.size}.")
    pick = rng.choice(roi_idx, size=int(holdout_k), replace=False)
    mask = np.zeros(int(N), dtype=bool)
    mask[pick] = True
    return mask


def predicted_q_lin_holdout_from_roi(
    P: np.ndarray,
    z: np.ndarray,
    roi_mask: np.ndarray,
    holdout_k: int,
    n: int,
) -> float:
    N = 2 ** int(n)
    G = int(np.sum(roi_mask))
    if G <= 0:
        raise ValueError("ROI is empty.")
    if holdout_k <= 0 or holdout_k > G:
        raise ValueError(f"Invalid holdout_k={holdout_k} for roi size={G}.")
    hat1_g = hv.indicator_walsh_coeffs(P=P, holdout_mask=roi_mask, n=int(n))
    vis_g = float(np.dot(np.asarray(z, dtype=np.float64), np.asarray(hat1_g, dtype=np.float64)))
    return float(float(holdout_k) / float(N) + (float(holdout_k) / float(G)) * vis_g)


def _build_cfg(args: argparse.Namespace, seed: int) -> hv.Config:
    return hv.Config(
        n=int(args.n),
        beta=float(args.beta),
        train_m=int(args.train_m),
        holdout_k=int(args.holdout_k),
        holdout_pool=max(200, int(args.holdout_k) * 10),
        seed=int(seed),
        good_frac=float(args.good_frac),
        sigmas=[float(args.sigma)],
        Ks=[int(args.K)],
        Qmax=max(10000, int(args.q_eval)),
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
        outdir=str(args.outdir),
    )


def _plot_predictor_vs_discovery(
    rep_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    q_eval: int,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), dpi=int(dpi))
    betas = rep_df["beta"].astype(float).to_numpy()
    bmin = float(np.min(betas))
    bmax = float(np.max(betas))
    if bmax <= bmin:
        bmax = bmin + 1e-6
    norm = Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap("Reds")

    # Left: calibration of proposition predictor against realized q_lin(H).
    ax = axes[0]
    ax.scatter(
        rep_df["q_lin_pred_from_roi"].astype(float).to_numpy(),
        rep_df["q_lin_ref_holdout"].astype(float).to_numpy(),
        c=rep_df["beta"].astype(float).to_numpy(),
        cmap=cmap,
        norm=norm,
        s=42,
        alpha=0.80,
        edgecolors="#404040",
        linewidths=0.5,
        label="Replicate",
    )
    ax.scatter(
        agg_df["q_lin_pred_from_roi"].astype(float).to_numpy(),
        agg_df["q_lin_ref_holdout_mean"].astype(float).to_numpy(),
        c=agg_df["beta"].astype(float).to_numpy(),
        cmap=cmap,
        norm=norm,
        s=92,
        alpha=0.95,
        edgecolors="#000000",
        linewidths=1.0,
        marker="D",
        label="Mean per (beta,seed)",
    )
    vals = np.concatenate(
        [
            rep_df["q_lin_pred_from_roi"].astype(float).to_numpy(),
            rep_df["q_lin_ref_holdout"].astype(float).to_numpy(),
        ]
    )
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        hi = lo + 1e-6
    ax.plot([lo, hi], [lo, hi], "--", color="#333333", linewidth=1.1)
    ax.set_xlabel(r"Predicted $\mathbb{E}_H[q_{\mathrm{lin}}(H)]$ from ROI")
    ax.set_ylabel(r"Measured $q_{\mathrm{lin}}(H)$ on random holdouts")
    ax.set_title("Calibration of holdout-free predictor")
    ax.grid(True, alpha=0.20, linestyle="--")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    # Right: predictor vs discovery (Q80 primary, R(Q_eval) secondary).
    ax = axes[1]
    x = agg_df["q_lin_pred_from_roi"].astype(float).to_numpy()
    y_q80 = agg_df["Q80_iqp_median"].astype(float).to_numpy()
    y_r = agg_df["R_qeval_iqp_median"].astype(float).to_numpy()
    beta_c = agg_df["beta"].astype(float).to_numpy()

    scat_q80 = ax.scatter(
        x,
        y_q80,
        c=beta_c,
        cmap=cmap,
        norm=norm,
        s=90,
        alpha=0.92,
        edgecolors="#8a0b0b",
        linewidths=0.8,
        label=r"IQP $Q_{80}$ (median)",
    )
    ax.set_xlabel(r"Predicted $\mathbb{E}_H[q_{\mathrm{lin}}(H)]$ from ROI")
    ax.set_ylabel(r"$Q_{80}$ (lower better)", color="#B22222")
    ax.tick_params(axis="y", labelcolor="#B22222")
    ax.grid(True, alpha=0.20, linestyle="--")

    ax2 = ax.twinx()
    ax2.scatter(
        x,
        y_r,
        c=beta_c,
        cmap=cmap,
        norm=norm,
        s=70,
        alpha=0.85,
        marker="s",
        edgecolors="#1c4d8f",
        linewidths=0.8,
        label=fr"IQP $R(Q={int(q_eval)})$ (median)",
    )
    ax2.set_ylabel(fr"$R(Q={int(q_eval)})$ (higher better)", color="#1c4d8f")
    ax2.tick_params(axis="y", labelcolor="#1c4d8f")
    ax.set_title("Discovery link (global random holdout)")

    # Combined legend.
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best", frameon=True, framealpha=0.95)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.018, fraction=0.025)
    cbar.set_label(r"$\beta$")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    betas = _parse_list_floats(args.betas)
    seeds = _parse_list_ints(args.seeds)

    rep_rows: List[Dict[str, object]] = []
    agg_rows: List[Dict[str, object]] = []

    total = len(betas) * len(seeds) * int(args.holdout_reps)
    done = 0

    for beta in betas:
        for seed in seeds:
            bits_table = hv.make_bits_table(int(args.n))
            p_star, support, scores = hv.build_target_distribution_paper(int(args.n), float(beta))
            roi_mask = support.astype(bool)  # Global ROI by design.
            roi_idx = np.where(roi_mask)[0]
            roi_size = int(roi_idx.size)
            if int(args.holdout_k) > roi_size:
                raise RuntimeError(f"holdout_k={int(args.holdout_k)} exceeds roi_size={roi_size}.")

            # Holdout-free predictor from ROI (fixed per beta/seed).
            alphas_ref = hv.sample_alphas(int(args.n), float(args.sigma), int(args.K), seed=int(seed) + 222)
            P_ref = hv.build_parity_matrix(alphas_ref, bits_table)
            z_ref = P_ref @ p_star
            q_lin_ref = hv.linear_band_reconstruction(P_ref, z_ref, int(args.n))
            q_lin_pred = predicted_q_lin_holdout_from_roi(
                P=P_ref,
                z=z_ref,
                roi_mask=roi_mask,
                holdout_k=int(args.holdout_k),
                n=int(args.n),
            )

            q80_vals: List[float] = []
            r_vals: List[float] = []
            qlin_vals: List[float] = []

            for rep in range(int(args.holdout_reps)):
                done += 1
                rng = np.random.default_rng(int(seed) * 1000 + int(round(beta * 100.0)) * 31 + 17 * (rep + 1))
                holdout_mask = sample_uniform_holdout_mask(
                    rng=rng,
                    roi_indices=roi_idx,
                    holdout_k=int(args.holdout_k),
                    N=p_star.size,
                )

                cfg_seed = int(seed) * 100 + (rep + 1)
                cfg_args = argparse.Namespace(**vars(args))
                cfg_args.beta = float(beta)
                cfg = _build_cfg(cfg_args, seed=cfg_seed)

                art = hv.rerun_single_setting(
                    cfg=cfg,
                    p_star=p_star,
                    holdout_mask=holdout_mask,
                    bits_table=bits_table,
                    sigma=float(args.sigma),
                    K=int(args.K),
                    return_hist=False,
                    iqp_loss="parity_mse",
                )
                q_iqp = art["q_iqp"]
                if not isinstance(q_iqp, np.ndarray):
                    raise RuntimeError("IQP distribution missing in rerun artifact.")

                q80_iqp = float(
                    hv.find_Q_threshold(
                        probs=q_iqp,
                        mask=holdout_mask,
                        thr=float(args.q80_thr),
                        Qmax=int(args.q80_search_max),
                    )
                )
                r_qeval = float(
                    hv.expected_unique_fraction(
                        probs=q_iqp,
                        mask=holdout_mask,
                        Q_vals=np.array([int(args.q_eval)], dtype=int),
                    )[0]
                )
                qlin_h_ref = float(np.sum(q_lin_ref[holdout_mask]))

                q80_vals.append(q80_iqp)
                r_vals.append(r_qeval)
                qlin_vals.append(qlin_h_ref)

                rep_rows.append(
                    {
                        "beta": float(beta),
                        "seed": int(seed),
                        "rep": int(rep),
                        "n": int(args.n),
                        "train_m": int(args.train_m),
                        "K": int(args.K),
                        "sigma": float(args.sigma),
                        "holdout_k": int(args.holdout_k),
                        "roi_mode": "global",
                        "roi_size": int(roi_size),
                        "q_lin_pred_from_roi": float(q_lin_pred),
                        "q_lin_ref_holdout": float(qlin_h_ref),
                        "Q80_iqp": float(q80_iqp),
                        f"R_Q{int(args.q_eval)}_iqp": float(r_qeval),
                    }
                )
                print(
                    f"[{done}/{total}] beta={beta:g} seed={seed} rep={rep} "
                    f"pred={q_lin_pred:.4e} qlin(H)={qlin_h_ref:.4e} Q80={q80_iqp:.0f}"
                )

            q80_arr = np.asarray(q80_vals, dtype=np.float64)
            r_arr = np.asarray(r_vals, dtype=np.float64)
            qlin_arr = np.asarray(qlin_vals, dtype=np.float64)
            q80_finite = q80_arr[np.isfinite(q80_arr)]
            agg_rows.append(
                {
                    "beta": float(beta),
                    "seed": int(seed),
                    "q_lin_pred_from_roi": float(q_lin_pred),
                    "q_lin_ref_holdout_mean": float(np.mean(qlin_arr)),
                    "q_lin_ref_holdout_median": float(np.median(qlin_arr)),
                    "Q80_iqp_mean": float(np.mean(q80_arr)),
                    "Q80_iqp_median": float(np.median(q80_arr)),
                    "Q80_iqp_std": float(np.std(q80_finite, ddof=1)) if q80_finite.size > 1 else float("nan"),
                    "R_qeval_iqp_mean": float(np.mean(r_arr)),
                    "R_qeval_iqp_median": float(np.median(r_arr)),
                    "R_qeval_iqp_std": float(np.std(r_arr, ddof=1)) if r_arr.size > 1 else 0.0,
                    "holdout_reps": int(args.holdout_reps),
                    "q_eval": int(args.q_eval),
                }
            )

    reps_csv = outdir / "global_visibility_discovery_points.csv"
    summary_csv = outdir / "global_visibility_discovery_summary.csv"
    stats_txt = outdir / "global_visibility_discovery_stats.txt"
    plot_pdf = outdir / "global_visibility_discovery_2panel.pdf"
    plot_png = outdir / "global_visibility_discovery_2panel.png"

    _write_csv(reps_csv, rep_rows)
    _write_csv(summary_csv, agg_rows)

    rep_df = pd.DataFrame(rep_rows)
    agg_df = pd.DataFrame(agg_rows)
    rho_calib = _spearman(
        rep_df["q_lin_pred_from_roi"].to_numpy(np.float64),
        rep_df["q_lin_ref_holdout"].to_numpy(np.float64),
    )
    rho_q80 = _spearman(
        agg_df["q_lin_pred_from_roi"].to_numpy(np.float64),
        -agg_df["Q80_iqp_median"].to_numpy(np.float64),
    )
    rho_r = _spearman(
        agg_df["q_lin_pred_from_roi"].to_numpy(np.float64),
        agg_df["R_qeval_iqp_median"].to_numpy(np.float64),
    )

    stats_lines = [
        "Global ROI visibility predictor vs discovery",
        (
            f"n={int(args.n)}, train_m={int(args.train_m)}, holdout_k={int(args.holdout_k)}, "
            f"K={int(args.K)}, sigma={float(args.sigma)}, q_eval={int(args.q_eval)}"
        ),
        (
            f"betas={min(_parse_list_floats(args.betas)):g}..{max(_parse_list_floats(args.betas)):g}, "
            f"seeds={len(_parse_list_ints(args.seeds))}, holdout_reps={int(args.holdout_reps)}"
        ),
        "",
        "Correlations:",
        f"spearman(pred_q_lin_H, realized_q_lin_H)={rho_calib:.6f}",
        f"spearman(pred_q_lin_H, -Q80_iqp_median)={rho_q80:.6f}",
        f"spearman(pred_q_lin_H, R_qeval_iqp_median)={rho_r:.6f}",
        f"std(pred_q_lin_H)={float(np.std(agg_df['q_lin_pred_from_roi'].to_numpy(np.float64))):.6e}",
        "",
        "Medians:",
        f"median(Q80_iqp_median)={float(agg_df['Q80_iqp_median'].median()):.6g}",
        f"median(R_qeval_iqp_median)={float(agg_df['R_qeval_iqp_median'].median()):.6g}",
    ]
    stats_txt.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    _plot_predictor_vs_discovery(
        rep_df=rep_df,
        agg_df=agg_df,
        out_pdf=plot_pdf,
        out_png=plot_png,
        q_eval=int(args.q_eval),
        dpi=int(args.dpi),
    )

    print(f"[saved] {reps_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {stats_txt}")
    print(f"[saved] {plot_pdf}")
    print(f"[saved] {plot_png}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Global ROI visibility predictor vs holdout discovery.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "48_global_visibility_predicts_discovery"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--betas", type=str, default="0.6,0.8,1.0,1.2,1.4")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-reps", type=int, default=3)

    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--good-frac", type=float, default=0.05)

    ap.add_argument("--iqp-steps", type=int, default=300)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)

    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    ap.add_argument("--q-eval", type=int, default=2000)
    ap.add_argument("--dpi", type=int, default=420)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
