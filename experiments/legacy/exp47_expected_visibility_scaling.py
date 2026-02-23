#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expected visibility scaling for random holdouts inside a fixed ROI.

Verifies numerically (Monte Carlo over random H subset G) the identities:
  E_H[1hat_H(alpha)] = (|H|/|G|) * 1hat_G(alpha)
  E_H[Vis_B(H)]      = (|H|/|G|) * Vis_B(G)
  E_H[q_lin(H)]      = |H|/2^n + (|H|/|G|) * Vis_B(G)

Default configuration matches the agreed medium budget.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def _roi_mask(roi_mode: str, support: np.ndarray, good_mask: np.ndarray) -> np.ndarray:
    mode = str(roi_mode).strip().lower()
    if mode == "global":
        return support.astype(bool)
    if mode == "high_value":
        return good_mask.astype(bool)
    raise ValueError(f"Unsupported roi_mode: {roi_mode}")


def expected_rhs_from_roi(
    P: np.ndarray,
    z: np.ndarray,
    roi_mask: np.ndarray,
    h: int,
    n: int,
) -> Dict[str, object]:
    """Compute holdout-free RHS quantities implied by the proposition."""
    N = 2 ** int(n)
    roi_size = int(np.sum(roi_mask))
    if roi_size <= 0:
        raise ValueError("ROI is empty.")
    if h <= 0 or h > roi_size:
        raise ValueError(f"Invalid holdout size h={h} for roi_size={roi_size}.")

    hat1_g = hv.indicator_walsh_coeffs(P=P, holdout_mask=roi_mask, n=int(n))
    vis_g = float(np.dot(np.asarray(z, dtype=np.float64), np.asarray(hat1_g, dtype=np.float64)))
    scale = float(h) / float(roi_size)

    rhs_hat1 = scale * np.asarray(hat1_g, dtype=np.float64)
    rhs_vis = float(scale * vis_g)
    rhs_q_lin_h = float(float(h) / float(N) + rhs_vis)

    return {
        "hat1_g": hat1_g,
        "vis_g": float(vis_g),
        "scale_h_over_roi": float(scale),
        "rhs_hat1": rhs_hat1,
        "rhs_vis": float(rhs_vis),
        "rhs_q_lin_h": float(rhs_q_lin_h),
    }


def monte_carlo_lhs_for_random_holdouts(
    P: np.ndarray,
    z: np.ndarray,
    q_lin: np.ndarray,
    roi_indices: np.ndarray,
    h: int,
    n: int,
    trials: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Monte Carlo estimates for random holdouts H uniformly sampled from ROI."""
    N = 2 ** int(n)
    K = int(P.shape[0])
    if h <= 0 or h > int(roi_indices.size):
        raise ValueError(f"Invalid h={h} for roi_indices size={roi_indices.size}.")
    if trials <= 0:
        raise ValueError("trials must be > 0.")

    acc_hat1 = np.zeros(K, dtype=np.float64)
    vis_vals = np.zeros(int(trials), dtype=np.float64)
    qlin_vals = np.zeros(int(trials), dtype=np.float64)

    zf = np.asarray(z, dtype=np.float64)
    qlin = np.asarray(q_lin, dtype=np.float64)
    Pf = np.asarray(P, dtype=np.float64)
    roi_idx = np.asarray(roi_indices, dtype=int)

    for t in range(int(trials)):
        pick = rng.choice(roi_idx, size=int(h), replace=False)
        hat1_h = np.sum(Pf[:, pick], axis=1) / float(N)
        vis_h = float(np.dot(zf, hat1_h))
        qlin_h = float(np.sum(qlin[pick]))

        acc_hat1 += hat1_h
        vis_vals[t] = vis_h
        qlin_vals[t] = qlin_h

    lhs_hat1 = acc_hat1 / float(trials)
    lhs_vis = float(np.mean(vis_vals))
    lhs_q_lin_h = float(np.mean(qlin_vals))

    return {
        "lhs_hat1": lhs_hat1,
        "lhs_vis": float(lhs_vis),
        "lhs_q_lin_h": float(lhs_q_lin_h),
        "vis_std": float(np.std(vis_vals, ddof=1)) if trials > 1 else 0.0,
        "q_lin_h_std": float(np.std(qlin_vals, ddof=1)) if trials > 1 else 0.0,
    }


def _plot_scaling_checks(
    alpha_df: pd.DataFrame,
    run_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    roi_mode: str,
    h: int,
    trials: int,
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

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), dpi=int(dpi))
    betas = run_df["beta"].astype(float).to_numpy()
    bmin = float(np.min(betas))
    bmax = float(np.max(betas))
    if bmax <= bmin:
        bmax = bmin + 1e-6
    norm = Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap("Reds")

    # Panel A: per-alpha expected indicator scaling.
    ax = axes[0]
    ax.scatter(
        alpha_df["rhs_hat1"].astype(float).to_numpy(),
        alpha_df["lhs_hat1_mean"].astype(float).to_numpy(),
        c=alpha_df["beta"].astype(float).to_numpy(),
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.35,
        linewidths=0.0,
    )
    xvals = np.concatenate(
        [
            alpha_df["rhs_hat1"].astype(float).to_numpy(),
            alpha_df["lhs_hat1_mean"].astype(float).to_numpy(),
        ]
    )
    finite = np.isfinite(xvals)
    lo = float(np.min(xvals[finite])) if finite.any() else -1.0
    hi = float(np.max(xvals[finite])) if finite.any() else 1.0
    if hi <= lo:
        hi = lo + 1e-6
    ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"RHS $(|H|/|G|)\,\hat 1_G(\alpha)$")
    ax.set_ylabel(r"MC mean $\mathbb{E}_H[\hat 1_H(\alpha)]$")
    ax.set_title("Indicator scaling")
    ax.grid(True, alpha=0.20, linestyle="--")

    # Panel B: visibility scaling.
    ax = axes[1]
    ax.scatter(
        run_df["rhs_vis"].astype(float).to_numpy(),
        run_df["lhs_vis"].astype(float).to_numpy(),
        c=run_df["beta"].astype(float).to_numpy(),
        cmap=cmap,
        norm=norm,
        s=60,
        alpha=0.9,
        edgecolors="#3a3a3a",
        linewidths=0.6,
    )
    x = run_df["rhs_vis"].astype(float).to_numpy()
    y = run_df["lhs_vis"].astype(float).to_numpy()
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if hi <= lo:
        hi = lo + 1e-6
    ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"RHS $(|H|/|G|)\,\mathrm{Vis}_B(G)$")
    ax.set_ylabel(r"MC mean $\mathbb{E}_H[\mathrm{Vis}_B(H)]$")
    ax.set_title("Visibility scaling")
    ax.grid(True, alpha=0.20, linestyle="--")

    # Panel C: q_lin(H) scaling.
    ax = axes[2]
    ax.scatter(
        run_df["rhs_q_lin_h"].astype(float).to_numpy(),
        run_df["lhs_q_lin_h"].astype(float).to_numpy(),
        c=run_df["beta"].astype(float).to_numpy(),
        cmap=cmap,
        norm=norm,
        s=60,
        alpha=0.9,
        edgecolors="#3a3a3a",
        linewidths=0.6,
    )
    x = run_df["rhs_q_lin_h"].astype(float).to_numpy()
    y = run_df["lhs_q_lin_h"].astype(float).to_numpy()
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if hi <= lo:
        hi = lo + 1e-6
    ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"RHS $|H|/2^n + (|H|/|G|)\,\mathrm{Vis}_B(G)$")
    ax.set_ylabel(r"MC mean $\mathbb{E}_H[q_{\mathrm{lin}}(H)]$")
    ax.set_title("Linear completion mass scaling")
    ax.grid(True, alpha=0.20, linestyle="--")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.012, fraction=0.025)
    cbar.set_label(r"$\beta$")

    fig.suptitle(
        fr"Expected visibility scaling | ROI={roi_mode}, $|H|$={int(h)}, trials={int(trials)}",
        fontsize=14,
        y=1.03,
    )
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
    n = int(args.n)
    h = int(args.holdout_k)
    K = int(args.K)
    sigma = float(args.sigma)
    trials = int(args.trials)
    force_alpha_all_ones = bool(int(args.force_alpha_all_ones))

    alpha_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    total = len(betas) * len(seeds)
    done = 0
    for beta in betas:
        for seed in seeds:
            done += 1
            bits_table = hv.make_bits_table(n)
            p_star, support, scores = hv.build_target_distribution_paper(n, float(beta))
            good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))
            roi_mask = _roi_mask(str(args.roi_mode), support, good_mask)
            roi_idx = np.where(roi_mask)[0]
            roi_size = int(roi_idx.size)
            if h > roi_size:
                raise RuntimeError(f"holdout_k={h} exceeds roi_size={roi_size}.")

            alphas = hv.sample_alphas(n, sigma, K, seed=int(seed) + 222)
            if force_alpha_all_ones:
                if K <= 0:
                    raise RuntimeError("K must be >= 1 when forcing alpha=all-ones.")
                # Inject the global parity mode explicitly (important for global ROI on S_even).
                alphas[0, :] = 1
            P = hv.build_parity_matrix(alphas, bits_table)
            z = P @ p_star
            q_lin = hv.linear_band_reconstruction(P, z, n)

            rhs = expected_rhs_from_roi(P=P, z=z, roi_mask=roi_mask, h=h, n=n)
            rng = np.random.default_rng(int(seed) + int(round(1000.0 * beta)) + 991)
            lhs = monte_carlo_lhs_for_random_holdouts(
                P=P,
                z=z,
                q_lin=q_lin,
                roi_indices=roi_idx,
                h=h,
                n=n,
                trials=trials,
                rng=rng,
            )

            rhs_hat1 = np.asarray(rhs["rhs_hat1"], dtype=np.float64)
            lhs_hat1 = np.asarray(lhs["lhs_hat1"], dtype=np.float64)
            abs_err_hat1 = np.abs(lhs_hat1 - rhs_hat1)
            denom_hat1 = np.maximum(1e-15, np.abs(rhs_hat1))
            rel_err_hat1 = abs_err_hat1 / denom_hat1

            for k in range(K):
                alpha_rows.append(
                    {
                        "beta": float(beta),
                        "seed": int(seed),
                        "alpha_idx": int(k),
                        "rhs_hat1": float(rhs_hat1[k]),
                        "lhs_hat1_mean": float(lhs_hat1[k]),
                        "abs_err_hat1": float(abs_err_hat1[k]),
                        "rel_err_hat1": float(rel_err_hat1[k]),
                    }
                )

            lhs_vis = float(lhs["lhs_vis"])
            rhs_vis = float(rhs["rhs_vis"])
            lhs_q = float(lhs["lhs_q_lin_h"])
            rhs_q = float(rhs["rhs_q_lin_h"])

            abs_err_vis = float(abs(lhs_vis - rhs_vis))
            abs_err_q = float(abs(lhs_q - rhs_q))
            rel_err_vis = float("nan")
            if abs(rhs_vis) > 1e-10:
                rel_err_vis = float(abs_err_vis / abs(rhs_vis))

            rel_err_q = float("nan")
            if abs(rhs_q) > 1e-10:
                rel_err_q = float(abs_err_q / abs(rhs_q))

            run_rows.append(
                {
                    "beta": float(beta),
                    "seed": int(seed),
                    "n": int(n),
                    "K": int(K),
                    "sigma": float(sigma),
                    "holdout_k": int(h),
                    "roi_mode": str(args.roi_mode),
                    "roi_size": int(roi_size),
                    "trials": int(trials),
                    "scale_h_over_roi": float(rhs["scale_h_over_roi"]),
                    "vis_g": float(rhs["vis_g"]),
                    "lhs_vis": float(lhs_vis),
                    "rhs_vis": float(rhs_vis),
                    "abs_err_vis": float(abs_err_vis),
                    "rel_err_vis": float(rel_err_vis),
                    "lhs_q_lin_h": float(lhs_q),
                    "rhs_q_lin_h": float(rhs_q),
                    "abs_err_q_lin_h": float(abs_err_q),
                    "rel_err_q_lin_h": float(rel_err_q),
                    "mean_abs_err_hat1": float(np.mean(abs_err_hat1)),
                    "median_abs_err_hat1": float(np.median(abs_err_hat1)),
                    "mean_rel_err_hat1": float(np.mean(rel_err_hat1)),
                    "median_rel_err_hat1": float(np.median(rel_err_hat1)),
                    "vis_std_mc": float(lhs["vis_std"]),
                    "q_lin_h_std_mc": float(lhs["q_lin_h_std"]),
                }
            )

            print(
                f"[{done}/{total}] beta={beta:g} seed={seed} "
                f"vis_err={abs_err_vis:.3e} qlin_err={abs_err_q:.3e}"
            )

    alpha_csv = outdir / "expected_visibility_scaling_points.csv"
    run_csv = outdir / "expected_visibility_scaling_summary.csv"
    stats_txt = outdir / "expected_visibility_scaling_stats.txt"
    plot_pdf = outdir / "expected_visibility_scaling_3panel.pdf"
    plot_png = outdir / "expected_visibility_scaling_3panel.png"

    _write_csv(alpha_csv, alpha_rows)
    _write_csv(run_csv, run_rows)

    alpha_df = pd.DataFrame(alpha_rows)
    run_df = pd.DataFrame(run_rows)
    rho_vis = _spearman(run_df["rhs_vis"].to_numpy(np.float64), run_df["lhs_vis"].to_numpy(np.float64))
    rho_q = _spearman(run_df["rhs_q_lin_h"].to_numpy(np.float64), run_df["lhs_q_lin_h"].to_numpy(np.float64))

    finite_rel_vis = run_df["rel_err_vis"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_rel_q = run_df["rel_err_q_lin_h"].replace([np.inf, -np.inf], np.nan).dropna()

    lines = [
        "Expected visibility scaling diagnostics",
        (
            f"n={n}, K={K}, sigma={sigma}, holdout_k={h}, trials={trials}, roi_mode={args.roi_mode}, "
            f"force_alpha_all_ones={int(force_alpha_all_ones)}"
        ),
        f"n_points_alpha={len(alpha_rows)}, n_runs={len(run_rows)}",
        "",
        "Aggregate errors:",
        f"median(mean_abs_err_hat1)={float(run_df['mean_abs_err_hat1'].median()):.6e}",
        f"median(abs_err_vis)={float(run_df['abs_err_vis'].median()):.6e}",
        f"median(abs_err_q_lin_h)={float(run_df['abs_err_q_lin_h'].median()):.6e}",
        (
            f"median(rel_err_vis | rhs!=0)={float(finite_rel_vis.median()):.6e}"
            if len(finite_rel_vis)
            else "median(rel_err_vis | rhs!=0)=nan"
        ),
        (
            f"median(rel_err_q_lin_h | rhs!=0)={float(finite_rel_q.median()):.6e}"
            if len(finite_rel_q)
            else "median(rel_err_q_lin_h | rhs!=0)=nan"
        ),
        "",
        "Rank correlations (lhs vs rhs):",
        f"spearman(lhs_vis,rhs_vis)={rho_vis:.6f}",
        f"spearman(lhs_q_lin_h,rhs_q_lin_h)={rho_q:.6f}",
    ]
    stats_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _plot_scaling_checks(
        alpha_df=alpha_df,
        run_df=run_df,
        out_pdf=plot_pdf,
        out_png=plot_png,
        dpi=int(args.dpi),
        roi_mode=str(args.roi_mode),
        h=int(h),
        trials=int(trials),
    )

    print(f"[saved] {alpha_csv}")
    print(f"[saved] {run_csv}")
    print(f"[saved] {stats_txt}")
    print(f"[saved] {plot_pdf}")
    print(f"[saved] {plot_png}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Expected visibility scaling for random holdouts.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "47_expected_visibility_scaling_global"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--betas", type=str, default="0.6,0.8,1.0,1.2,1.4")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--roi-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--force-alpha-all-ones", type=int, default=0, choices=[0, 1])
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--dpi", type=int, default=420)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
