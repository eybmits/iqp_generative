#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create quick plots for the n=12, beta=0.9 D3PM vs IQP comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _load_iqp_seed_values(iqp_csv: Path, beta: float) -> pd.DataFrame:
    df = pd.read_csv(iqp_csv)
    df = df[
        (df["model_key"] == "iqp_parity_mse")
        & np.isclose(df["beta"].astype(float).to_numpy(), float(beta))
    ].copy()
    need = ["seed", "TV_score", "BSHS", "Composite"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"IQP CSV missing required columns: {missing}")
    return df[need].copy()


def _load_iqp_recovery_from_fig6(fig6_npz: Path, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(fig6_npz, allow_pickle=True) as z:
        q = np.asarray(z["Q"], dtype=np.float64)
        betas = np.asarray(z["betas"], dtype=np.float64)
        bidx = int(np.argmin(np.abs(betas - float(beta))))
        if not np.isclose(float(betas[bidx]), float(beta), atol=1e-9, rtol=0.0):
            raise ValueError(f"beta={beta} not present in {fig6_npz}; available={betas.tolist()}")

        order = [str(x) for x in np.asarray(z["model_order"], dtype=object).tolist()]
        if "iqp_parity_mse" not in order:
            raise ValueError("Model key 'iqp_parity_mse' not found in fig6 model_order.")
        midx = order.index("iqp_parity_mse")

        y_iqp = np.asarray(z["y_models"][bidx, midx], dtype=np.float64)
        y_target = np.asarray(z["y_target"][bidx], dtype=np.float64)
        y_unif = np.asarray(z["y_unif"][bidx], dtype=np.float64)
    return q, y_iqp, y_target, y_unif


def _boxplot_main(
    d3pm_df: pd.DataFrame,
    iqp_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    metrics = [
        ("TV_score", "TV_score (lower better)"),
        ("BSHS", "BSHS (higher better)"),
        ("Composite", "Composite (higher better)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, (col, title) in zip(axes, metrics):
        d_vals = d3pm_df[col].astype(float).to_numpy()
        i_vals = iqp_df[col].astype(float).to_numpy()
        bp = ax.boxplot(
            [d_vals, i_vals],
            tick_labels=["D3PM", "IQP parity"],
            patch_artist=True,
            showmeans=True,
            meanline=False,
            widths=0.58,
        )
        colors = ["#2ca02c", "#c40000"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.23)
            patch.set_edgecolor(c)
            patch.set_linewidth(1.4)
        for median in bp["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.4)
        for mean in bp["means"]:
            mean.set_marker("o")
            mean.set_markerfacecolor("#111111")
            mean.set_markeredgecolor("white")
            mean.set_markersize(5.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.18, linestyle="--")

    fig.suptitle("n=12, beta=0.9: D3PM vs IQP parity (seed-wise)", fontsize=12)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _d3pm_holdout_plot(
    d3pm_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    d = d3pm_df.sort_values("seed").copy()
    seeds = d["seed"].astype(int).to_numpy()
    qh = d["q_holdout"].astype(float).to_numpy()
    r10k = d["R_Q10000"].astype(float).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    axes[0].plot(seeds, qh, color="#2ca02c", marker="o", linewidth=1.8)
    axes[0].axhline(float(np.mean(qh)), color="#2ca02c", linestyle="--", linewidth=1.3, alpha=0.85)
    axes[0].set_title("D3PM holdout mass q(H)")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("q(H)")
    axes[0].grid(True, alpha=0.18, linestyle="--")

    axes[1].plot(seeds, r10k, color="#2ca02c", marker="o", linewidth=1.8)
    axes[1].axhline(float(np.mean(r10k)), color="#2ca02c", linestyle="--", linewidth=1.3, alpha=0.85)
    axes[1].set_title("D3PM recovery R(Q=10000)")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("R_Q10000")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.18, linestyle="--")

    fig.suptitle("n=12, beta=0.9: D3PM holdout metrics", fontsize=12)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _recovery_overlay_plot(
    *,
    d3pm_curve_npz: Path,
    fig6_npz: Path,
    beta: float,
    out_pdf: Path,
    out_png: Path,
) -> None:
    with np.load(d3pm_curve_npz, allow_pickle=False) as zd:
        q_d3 = np.asarray(zd["Q"], dtype=np.float64)
        y_d3 = np.asarray(zd["d3pm_curve_mean"], dtype=np.float64)
        s_d3 = np.asarray(zd["d3pm_curve_std"], dtype=np.float64)

    q_iqp, y_iqp, y_target, y_unif = _load_iqp_recovery_from_fig6(fig6_npz=fig6_npz, beta=beta)

    if q_d3.shape != q_iqp.shape or not np.allclose(q_d3, q_iqp):
        y_d3 = np.interp(q_iqp, q_d3, y_d3)
        s_d3 = np.interp(q_iqp, q_d3, s_d3)
        q_plot = q_iqp
    else:
        q_plot = q_iqp

    lo = np.clip(y_d3 - s_d3, 0.0, 1.0)
    hi = np.clip(y_d3 + s_d3, 0.0, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(5.1, 3.4))
    ax.plot(q_plot, y_target, color="#111111", linewidth=2.0, label=r"Target $p^*$", zorder=5)
    ax.plot(q_plot, y_iqp, color="#C40000", linewidth=2.0, label="IQP parity", zorder=6)
    ax.plot(q_plot, y_d3, color="#2ca02c", linewidth=2.0, label="D3PM (mean over seeds)", zorder=7)
    ax.fill_between(q_plot, lo, hi, color="#2ca02c", alpha=0.22, linewidth=0.0, label="D3PM ±1 std", zorder=4)
    ax.plot(q_plot, y_unif, color="#8A8A8A", linestyle="--", linewidth=1.4, label="Uniform", zorder=3)

    ax.set_xlim(0.0, float(np.max(q_plot)))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_title(rf"n=12, $\beta={beta:g}$: Recovery vs IQP parity")
    ax.grid(True, alpha=0.18, linestyle="--")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#bfbfbf")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot D3PM vs IQP (n=12, beta=0.9) outputs.")
    ap.add_argument(
        "--indir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "d3pm_vs_iqp_n12_beta0p9"),
    )
    ap.add_argument(
        "--iqp-csv",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "final_plots"
            / "fig3_tv_bshs_seedmean_scatter"
            / "tv_bshs_points_multiseed_beta_q1000_no_iqp_mse_beta0p9_newseeds12.csv"
        ),
    )
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument(
        "--fig6-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig6_beta_sweep_recovery_grid" / "fig6_data_default.npz"),
    )
    args = ap.parse_args()

    _style()
    indir = Path(args.indir)
    d3pm_csv = indir / "d3pm_seed_metrics_n12_beta0p9.csv"
    if not d3pm_csv.exists():
        raise FileNotFoundError(f"Missing D3PM metrics CSV: {d3pm_csv}")

    d3pm_df = pd.read_csv(d3pm_csv)
    iqp_df = _load_iqp_seed_values(Path(args.iqp_csv), beta=float(args.beta))

    out1_pdf = indir / "d3pm_vs_iqp_boxplot_beta0p9_n12.pdf"
    out1_png = indir / "d3pm_vs_iqp_boxplot_beta0p9_n12.png"
    _boxplot_main(d3pm_df=d3pm_df, iqp_df=iqp_df, out_pdf=out1_pdf, out_png=out1_png)

    out2_pdf = indir / "d3pm_holdout_metrics_beta0p9_n12.pdf"
    out2_png = indir / "d3pm_holdout_metrics_beta0p9_n12.png"
    _d3pm_holdout_plot(d3pm_df=d3pm_df, out_pdf=out2_pdf, out_png=out2_png)

    d3_curve_npz = indir / "d3pm_recovery_curves_n12_beta0p9.npz"
    if d3_curve_npz.exists():
        out3_pdf = indir / "d3pm_vs_iqp_recovery_beta0p9_n12.pdf"
        out3_png = indir / "d3pm_vs_iqp_recovery_beta0p9_n12.png"
        _recovery_overlay_plot(
            d3pm_curve_npz=d3_curve_npz,
            fig6_npz=Path(args.fig6_npz),
            beta=float(args.beta),
            out_pdf=out3_pdf,
            out_png=out3_png,
        )
        print(f"[saved] {out3_pdf}")
        print(f"[saved] {out3_png}")
    else:
        print(f"[skip] recovery overlay not generated; missing {d3_curve_npz}")

    print(f"[saved] {out1_pdf}")
    print(f"[saved] {out1_png}")
    print(f"[saved] {out2_pdf}")
    print(f"[saved] {out2_png}")


if __name__ == "__main__":
    main()
