#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot strong Transformer baseline vs IQP over beta values."""

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
    return f"{float(beta):.1f}".replace(".", "p")


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


def _load_fig6(fig6_npz: Path) -> dict:
    with np.load(fig6_npz, allow_pickle=True) as z:
        q = np.asarray(z["Q"], dtype=np.float64)
        betas = np.asarray(z["betas"], dtype=np.float64)
        y_models = np.asarray(z["y_models"], dtype=np.float64)
        y_target = np.asarray(z["y_target"], dtype=np.float64)
        y_unif = np.asarray(z["y_unif"], dtype=np.float64)
        order = [str(x) for x in np.asarray(z["model_order"], dtype=object).tolist()]
    if "iqp_parity_mse" not in order:
        raise ValueError("iqp_parity_mse missing in fig6 model_order")
    iqp_idx = order.index("iqp_parity_mse")
    return {
        "Q": q,
        "betas": betas,
        "y_iqp": y_models[:, iqp_idx, :],
        "y_target": y_target,
        "y_unif": y_unif,
    }


def _load_transformer_curve(outdir: Path, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tag = _beta_tag(beta)
    npz = outdir / f"transformer_recovery_curves_n12_beta{tag}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Missing curve file: {npz}")
    with np.load(npz, allow_pickle=False) as z:
        q = np.asarray(z["Q"], dtype=np.float64)
        m = np.asarray(z["transformer_curve_mean"], dtype=np.float64)
        s = np.asarray(z["transformer_curve_std"], dtype=np.float64)
    return q, m, s


def _interp_if_needed(q_src: np.ndarray, y_src: np.ndarray, q_tgt: np.ndarray) -> np.ndarray:
    if q_src.shape == q_tgt.shape and np.allclose(q_src, q_tgt):
        return y_src
    return np.interp(q_tgt, q_src, y_src)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Transformer-vs-IQP ablation.")
    ap.add_argument("--betas", type=str, default="0.6,0.8,1.0,1.2")
    ap.add_argument(
        "--indir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "transformer_vs_iqp_ablation"),
    )
    ap.add_argument(
        "--fig6-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig6_beta_sweep_recovery_grid" / "fig6_data_default.npz"),
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "transformer_vs_iqp_ablation"),
    )
    args = ap.parse_args()

    betas = _parse_floats(args.betas)
    if not betas:
        raise ValueError("No betas provided.")

    _style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    indir = Path(args.indir)
    fig6 = _load_fig6(Path(args.fig6_npz))
    q = fig6["Q"]
    fig6_betas = fig6["betas"]

    ncols = 2
    nrows = int(np.ceil(len(betas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, 3.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    summary_rows: list[dict] = []
    i_q10000 = int(np.argmin(np.abs(q - 10000.0)))

    for i, beta in enumerate(betas):
        ax = axes[i]
        qt, ym, ys = _load_transformer_curve(indir, beta)
        ym = _interp_if_needed(qt, ym, q)
        ys = _interp_if_needed(qt, ys, q)

        bidx = int(np.argmin(np.abs(fig6_betas - beta)))
        if not np.isclose(fig6_betas[bidx], beta, atol=1e-9, rtol=0.0):
            raise ValueError(f"beta={beta} missing in fig6.")
        yi = fig6["y_iqp"][bidx]
        yt = fig6["y_target"][bidx]
        yu = fig6["y_unif"][bidx]

        ax.plot(q, yt, color="#111111", linewidth=1.8, label="Target", zorder=5)
        ax.plot(q, yi, color="#C40000", linewidth=1.9, label="IQP parity", zorder=6)
        ax.plot(q, ym, color="#0b6e4f", linewidth=2.0, label="Transformer strong", zorder=7)
        ax.fill_between(q, np.clip(ym - ys, 0.0, 1.0), np.clip(ym + ys, 0.0, 1.0), color="#0b6e4f", alpha=0.22, zorder=4)
        ax.plot(q, yu, color="#8A8A8A", linewidth=1.2, linestyle=":", label="Uniform", zorder=2)

        ax.set_xlim(0.0, float(np.max(q)))
        ax.set_ylim(0.0, 1.02)
        ax.set_title(rf"$\beta={beta:g}$")
        ax.set_xlabel(r"$Q$")
        ax.set_ylabel(r"$R(Q)$")
        ax.grid(True, alpha=0.15, linestyle="--")

        summary_rows.extend(
            [
                {"beta": beta, "model": "transformer_strong", "R_Q10000_mean": float(ym[i_q10000]), "R_Q10000_std": float(ys[i_q10000])},
                {"beta": beta, "model": "iqp_parity", "R_Q10000_mean": float(yi[i_q10000]), "R_Q10000_std": float("nan")},
            ]
        )

    for j in range(len(betas), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Strong Transformer vs IQP parity: recovery ablation (n=12)", y=1.06, fontsize=12)
    out_grid_pdf = outdir / "transformer_vs_iqp_recovery_ablation_n12.pdf"
    out_grid_png = outdir / "transformer_vs_iqp_recovery_ablation_n12.png"
    fig.savefig(out_grid_pdf)
    fig.savefig(out_grid_png, dpi=300)
    plt.close(fig)

    # R(10000) summary bar plot.
    sdf = pd.DataFrame(summary_rows).sort_values(["beta", "model"]).reset_index(drop=True)
    out_csv = outdir / "transformer_vs_iqp_rq10000_summary.csv"
    sdf.to_csv(out_csv, index=False)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7.2, 3.4))
    x = np.arange(len(betas), dtype=np.float64)
    w = 0.30
    st = sdf[sdf["model"] == "transformer_strong"].sort_values("beta")
    si = sdf[sdf["model"] == "iqp_parity"].sort_values("beta")

    y_t = st["R_Q10000_mean"].to_numpy(dtype=np.float64)
    s_t = st["R_Q10000_std"].to_numpy(dtype=np.float64)
    y_i = si["R_Q10000_mean"].to_numpy(dtype=np.float64)

    ax2.bar(x - w / 2.0, y_t, width=w, color="#0b6e4f", alpha=0.88, label="Transformer strong")
    ax2.errorbar(x - w / 2.0, y_t, yerr=s_t, fmt="none", ecolor="#084c38", capsize=3.0, linewidth=1.1)
    ax2.bar(x + w / 2.0, y_i, width=w, color="#C40000", alpha=0.88, label="IQP parity")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b:g}" for b in betas])
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel(r"$R(10000)$")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(True, axis="y", alpha=0.15, linestyle="--")
    ax2.set_title(r"$R(10000)$ summary (Transformer mean$\pm$std)")
    ax2.legend(loc="lower left", frameon=True)

    out_bar_pdf = outdir / "transformer_vs_iqp_rq10000_ablation_n12.pdf"
    out_bar_png = outdir / "transformer_vs_iqp_rq10000_ablation_n12.png"
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

