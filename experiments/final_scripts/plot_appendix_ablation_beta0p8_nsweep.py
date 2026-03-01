#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final appendix plot script: beta=0.8 n-sweep ablation (data-driven)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

PNG_DPI = 300
FIG_W = 6.95
FIG_H = 2.60

MODEL_STYLES: Dict[str, Dict[str, object]] = {
    "iqp_parity": {"label": "IQP (parity)", "color": "#C40000", "marker": "o"},
    "iqp_mse": {"label": "IQP (MSE)", "color": "#1F77B4", "marker": "s"},
}


def apply_final_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.8,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#bfbfbf",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.03,
        }
    )


def _draw_panel(
    ax: plt.Axes,
    n_values: np.ndarray,
    model_keys: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    ylabel: str,
    show_seed_points: bool,
    seed_values: np.ndarray,
    seed_data: np.ndarray,
) -> None:
    for mi, key in enumerate(model_keys):
        st = MODEL_STYLES.get(key, {"label": key, "color": "#444444", "marker": "o"})
        x = n_values.astype(np.float64)
        y = means[mi]
        yerr = stds[mi]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=str(st["color"]),
            marker=str(st["marker"]),
            linewidth=2.0,
            capsize=3.0,
            elinewidth=1.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=str(st["label"]),
            zorder=8,
        )

        if show_seed_points:
            for ni, n in enumerate(n_values.tolist()):
                vals = seed_data[mi, ni]
                if vals.size == 0:
                    continue
                jitter = np.linspace(-0.12, 0.12, num=vals.size)
                ax.scatter(
                    np.full(vals.size, float(n)) + jitter,
                    vals,
                    color=str(st["color"]),
                    alpha=0.35,
                    s=18,
                    linewidths=0.0,
                    zorder=6,
                )

    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_xticks(n_values.tolist())
    ax.grid(True, alpha=0.16, linestyle="--")


def run() -> None:
    ap = argparse.ArgumentParser(description="Final appendix ablation plot (beta=0.8, n-sweep, data-driven).")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig7_appendix_ablation_beta0p8_nsweep"),
    )
    ap.add_argument(
        "--data-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig7_appendix_ablation_beta0p8_nsweep" / "fig7_data_default.npz"),
    )
    ap.add_argument("--show-seed-points", type=int, default=0, choices=[0, 1])
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_npz = Path(args.data_npz)
    if not data_npz.exists():
        raise FileNotFoundError(f"Missing data file: {data_npz}")

    with np.load(data_npz, allow_pickle=True) as z:
        n_values = np.asarray(z["n_values"], dtype=np.int64)
        model_keys = [str(x) for x in z["model_keys"].tolist()]
        beta = float(z["beta"])
        seed_values = np.asarray(z["seed_values"], dtype=np.int64)

        q_holdout_seed = np.asarray(z["q_holdout_seed"], dtype=np.float64)
        r_q10000_seed = np.asarray(z["r_q10000_seed"], dtype=np.float64)
        q_holdout_mean = np.asarray(z["q_holdout_mean"], dtype=np.float64)
        q_holdout_std = np.asarray(z["q_holdout_std"], dtype=np.float64)
        r_q10000_mean = np.asarray(z["r_q10000_mean"], dtype=np.float64)
        r_q10000_std = np.asarray(z["r_q10000_std"], dtype=np.float64)

        eval_mode_by_n = [str(x) for x in z["eval_mode_by_n"].tolist()]
        shots_n16 = int(z["shots_n16"]) if "shots_n16" in z else 0

    apply_final_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), constrained_layout=True)
    ax_l, ax_r = axes

    _draw_panel(
        ax=ax_l,
        n_values=n_values,
        model_keys=model_keys,
        means=q_holdout_mean,
        stds=q_holdout_std,
        ylabel=r"Holdout mass $q(H)$",
        show_seed_points=bool(int(args.show_seed_points)),
        seed_values=seed_values,
        seed_data=q_holdout_seed,
    )

    _draw_panel(
        ax=ax_r,
        n_values=n_values,
        model_keys=model_keys,
        means=r_q10000_mean,
        stds=r_q10000_std,
        ylabel=r"Recovery $R(10000)$",
        show_seed_points=bool(int(args.show_seed_points)),
        seed_values=seed_values,
        seed_data=r_q10000_seed,
    )
    ax_r.set_ylim(0.0, 1.02)

    ax_l.legend(
        loc="upper right",
        frameon=True,
        fontsize=8.0,
        facecolor="white",
        edgecolor="#bfbfbf",
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.8,
    )

    n_shot = [int(n_values[i]) for i, mode in enumerate(eval_mode_by_n) if mode.startswith("shot")]
    shot_note = ""
    if n_shot:
        shot_note = f"exact for n<=14; shot-based ({shots_n16:,} shots) for n={','.join(map(str, n_shot))}"
    else:
        shot_note = "exact evaluation for all n"

    fig.text(
        0.5,
        0.01,
        fr"$\beta={beta:g}$, mean$\pm$std over {seed_values.size} seeds; {shot_note}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    out_pdf = outdir / "fig7_appendix_ablation_beta0p8_nsweep.pdf"
    out_png = outdir / "fig7_appendix_ablation_beta0p8_nsweep.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    run()
