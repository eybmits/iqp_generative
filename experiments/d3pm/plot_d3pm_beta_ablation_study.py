#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot D3PM ablation study (base vs strong) against IQP parity over beta values."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _beta_tag(beta: float) -> str:
    s = f"{float(beta):.1f}"
    return s.replace(".", "p")


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


def _load_fig6_curves(fig6_npz: Path) -> dict:
    with np.load(fig6_npz, allow_pickle=True) as z:
        q = np.asarray(z["Q"], dtype=np.float64)
        betas = np.asarray(z["betas"], dtype=np.float64)
        y_models = np.asarray(z["y_models"], dtype=np.float64)
        y_target = np.asarray(z["y_target"], dtype=np.float64)
        y_unif = np.asarray(z["y_unif"], dtype=np.float64)
        order = [str(x) for x in np.asarray(z["model_order"], dtype=object).tolist()]
    if "iqp_parity_mse" not in order:
        raise ValueError("iqp_parity_mse not found in fig6 model_order.")
    iqp_idx = order.index("iqp_parity_mse")
    return {
        "Q": q,
        "betas": betas,
        "y_iqp": y_models[:, iqp_idx, :],
        "y_target": y_target,
        "y_unif": y_unif,
    }


def _load_d3pm_curves(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = run_dir / "d3pm_recovery_curves_n12_beta0p9.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Missing D3PM curve file: {npz}")
    with np.load(npz, allow_pickle=False) as z:
        q = np.asarray(z["Q"], dtype=np.float64)
        m = np.asarray(z["d3pm_curve_mean"], dtype=np.float64)
        s = np.asarray(z["d3pm_curve_std"], dtype=np.float64)
    return q, m, s


def _interp_if_needed(q_src: np.ndarray, y_src: np.ndarray, q_tgt: np.ndarray) -> np.ndarray:
    if q_src.shape == q_tgt.shape and np.allclose(q_src, q_tgt):
        return y_src
    return np.interp(q_tgt, q_src, y_src)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create D3PM ablation study plots over beta.")
    ap.add_argument("--betas", type=str, default="0.6,0.8,1.0,1.2")
    ap.add_argument(
        "--base-root",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "d3pm_ablation_beta_sweep"),
    )
    ap.add_argument(
        "--fig6-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig6_beta_sweep_recovery_grid" / "fig6_data_default.npz"),
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "d3pm_ablation_beta_sweep"),
    )
    args = ap.parse_args()

    betas = _parse_floats(args.betas)
    if not betas:
        raise ValueError("No betas provided.")

    _style()
    fig6 = _load_fig6_curves(Path(args.fig6_npz))
    q_fig6 = fig6["Q"]
    fig6_betas = fig6["betas"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base_root = Path(args.base_root)

    summary_rows: list[dict] = []

    # Plot 1: grid of recovery curves over betas.
    ncols = 2
    nrows = int(np.ceil(len(betas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 3.1 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for i, beta in enumerate(betas):
        ax = axes[i]
        tag = _beta_tag(beta)
        base_dir = base_root / f"base_beta{tag}"
        strong_dir = base_root / f"strong_beta{tag}"
        q_base, yb_m, yb_s = _load_d3pm_curves(base_dir)
        q_str, ys_m, ys_s = _load_d3pm_curves(strong_dir)
        yb_m = _interp_if_needed(q_base, yb_m, q_fig6)
        yb_s = _interp_if_needed(q_base, yb_s, q_fig6)
        ys_m = _interp_if_needed(q_str, ys_m, q_fig6)
        ys_s = _interp_if_needed(q_str, ys_s, q_fig6)

        bidx = int(np.argmin(np.abs(fig6_betas - beta)))
        if not np.isclose(fig6_betas[bidx], beta, atol=1e-9, rtol=0.0):
            raise ValueError(f"beta={beta} missing in fig6 npz.")
        y_iqp = fig6["y_iqp"][bidx]
        y_target = fig6["y_target"][bidx]
        y_unif = fig6["y_unif"][bidx]

        ax.plot(q_fig6, y_target, color="#111111", linewidth=1.8, label="Target", zorder=5)
        ax.plot(q_fig6, y_iqp, color="#C40000", linewidth=1.9, label="IQP parity", zorder=6)

        ax.plot(q_fig6, yb_m, color="#1f77b4", linewidth=1.7, linestyle="--", label="D3PM base", zorder=7)
        ax.fill_between(q_fig6, np.clip(yb_m - yb_s, 0.0, 1.0), np.clip(yb_m + yb_s, 0.0, 1.0), color="#1f77b4", alpha=0.18, zorder=4)

        ax.plot(q_fig6, ys_m, color="#2ca02c", linewidth=1.9, linestyle="-", label="D3PM strong", zorder=8)
        ax.fill_between(q_fig6, np.clip(ys_m - ys_s, 0.0, 1.0), np.clip(ys_m + ys_s, 0.0, 1.0), color="#2ca02c", alpha=0.20, zorder=3)

        ax.plot(q_fig6, y_unif, color="#8A8A8A", linewidth=1.2, linestyle=":", label="Uniform", zorder=2)
        ax.set_xlim(0.0, float(np.max(q_fig6)))
        ax.set_ylim(0.0, 1.02)
        ax.set_title(rf"$\beta={beta:g}$")
        ax.set_xlabel(r"$Q$")
        ax.set_ylabel(r"$R(Q)$")
        ax.grid(True, alpha=0.15, linestyle="--")

        i_q10000 = int(np.argmin(np.abs(q_fig6 - 10000.0)))
        summary_rows.extend(
            [
                {
                    "beta": beta,
                    "model": "d3pm_base",
                    "R_Q10000_mean": float(yb_m[i_q10000]),
                    "R_Q10000_std": float(yb_s[i_q10000]),
                },
                {
                    "beta": beta,
                    "model": "d3pm_strong",
                    "R_Q10000_mean": float(ys_m[i_q10000]),
                    "R_Q10000_std": float(ys_s[i_q10000]),
                },
                {
                    "beta": beta,
                    "model": "iqp_parity",
                    "R_Q10000_mean": float(y_iqp[i_q10000]),
                    "R_Q10000_std": float("nan"),
                },
            ]
        )

    for j in range(len(betas), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Ablation Study: D3PM base vs strong vs IQP parity (n=12, Global+Smart, m=200)", y=1.06, fontsize=12)
    out_grid_pdf = outdir / "ablation_recovery_grid_d3pm_vs_iqp_n12.pdf"
    out_grid_png = outdir / "ablation_recovery_grid_d3pm_vs_iqp_n12.png"
    fig.savefig(out_grid_pdf)
    fig.savefig(out_grid_png, dpi=300)
    plt.close(fig)

    # Plot 2: R_Q10000 summary bars.
    sum_df = pd.DataFrame(summary_rows)
    sum_df = sum_df.sort_values(["beta", "model"]).reset_index(drop=True)
    out_csv = outdir / "ablation_summary_rq10000.csv"
    sum_df.to_csv(out_csv, index=False)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7.4, 3.5))
    x = np.arange(len(betas), dtype=np.float64)
    w = 0.24

    def _vals(model: str) -> tuple[np.ndarray, np.ndarray]:
        d = sum_df[sum_df["model"] == model].sort_values("beta")
        return d["R_Q10000_mean"].to_numpy(dtype=np.float64), d["R_Q10000_std"].to_numpy(dtype=np.float64)

    yb, sb = _vals("d3pm_base")
    ys, ss = _vals("d3pm_strong")
    yi, _ = _vals("iqp_parity")

    ax2.bar(x - w, yb, width=w, color="#1f77b4", alpha=0.85, label="D3PM base")
    ax2.errorbar(x - w, yb, yerr=sb, fmt="none", ecolor="#133d66", capsize=3.0, linewidth=1.1)
    ax2.bar(x, ys, width=w, color="#2ca02c", alpha=0.85, label="D3PM strong")
    ax2.errorbar(x, ys, yerr=ss, fmt="none", ecolor="#1a6a1e", capsize=3.0, linewidth=1.1)
    ax2.bar(x + w, yi, width=w, color="#c40000", alpha=0.85, label="IQP parity")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b:g}" for b in betas])
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel(r"$R(10000)$")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_title(r"$R(10000)$ summary (IQP std unavailable in snapshot)")
    ax2.grid(True, axis="y", alpha=0.15, linestyle="--")
    ax2.legend(loc="lower left", frameon=True)

    out_bar_pdf = outdir / "ablation_rq10000_d3pm_vs_iqp_n12.pdf"
    out_bar_png = outdir / "ablation_rq10000_d3pm_vs_iqp_n12.png"
    fig2.savefig(out_bar_pdf)
    fig2.savefig(out_bar_png, dpi=300)
    plt.close(fig2)

    print(f"[saved] {out_grid_pdf}")
    print(f"[saved] {out_grid_png}")
    print(f"[saved] {out_bar_pdf}")
    print(f"[saved] {out_bar_png}")
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()

