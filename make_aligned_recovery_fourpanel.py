#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the four recovery-curve panels as one LaTeX-sized textwidth figure.

The output is a rerender-only composition of the exact recovery plots used in
the manuscript draft:

1. Experiment 4 IQP-parity versus spectral proxy,
2. Experiment 4 parity IQP family versus IQP-MSE,
3. Experiment 4 spectral family sweep,
4. Experiment 14 hardware/simulation recovery comparison.

The PDF natural width is fixed at 516 pt so it can be included via
``\\includegraphics[width=1.0\\textwidth]{...}`` without downstream scaling
changing the intended font-size relationship.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

from final_plot_style import (  # noqa: E402
    MSE_COLOR,
    TARGET_COLOR,
    TEXT_DARK,
    UNIFORM_COLOR,
    apply_ieee_latex_style,
    save_exact_figure,
)
from model_labels import IQP_MSE_LABEL, IQP_PARITY_LABEL  # noqa: E402


ROOT = Path(__file__).resolve().parent
OUTPUT_STEM = "fig_recovery_fourpanel_aligned_textwidth"
DEFAULT_OUTDIR = ROOT / "plots" / "aligned_recovery_fourpanel"

EXP4_DIR = ROOT / "plots" / "experiment_4_recovery_sigmak_triplet_seed118"
EXP4_NPZ = EXP4_DIR / "coverage_sigmak_triplet_data.npz"
EXP4_RUN_CONFIG = EXP4_DIR / "RUN_CONFIG.json"
EXP16_NPZ = (
    ROOT
    / "plots"
    / "experiment_16_seedwise_best_hardware_recovery_curve"
    / "experiment_16_seedwise_best_hardware_recovery_curve_avg_common_seeds_no_spectral_paper.npz"
)

TEXTWIDTH_PT = 516.0
TEX_PT_PER_IN = 72.27
FIG_W_IN = TEXTWIDTH_PT / TEX_PT_PER_IN
FIG_H_IN = 145.0 / TEX_PT_PER_IN

PARITY_COLOR = "#DC2626"
PARITY_CONTEXT_COLOR = "#F4A5A5"
SPECTRAL_COLOR = "#565656"
SPECTRAL_CONTEXT_COLOR = "#BDBDBD"
MSE_PANEL_COLOR = "#1F77B4"
GRID_COLOR = "#D8D8D8"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_exp4() -> dict[str, object]:
    with np.load(EXP4_NPZ, allow_pickle=True) as z:
        q = np.asarray(z["Q"], dtype=np.float64)
        parity_labels = [str(x) for x in z["parity_labels"].tolist()]
        spectral_labels = [str(x) for x in z["spectral_labels"].tolist()]
        payload: dict[str, object] = {
            "Q": q,
            "target_curve": np.asarray(z["target_curve"], dtype=np.float64),
            "uniform_curve": np.asarray(z["uniform_curve"], dtype=np.float64),
            "iqp_mse_curve": np.asarray(z["iqp_mse_curve"], dtype=np.float64),
            "parity_by_key": {
                key: np.asarray(curve, dtype=np.float64)
                for key, curve in zip(parity_labels, np.asarray(z["parity_curves"], dtype=np.float64))
            },
            "spectral_by_key": {
                key: np.asarray(curve, dtype=np.float64)
                for key, curve in zip(spectral_labels, np.asarray(z["spectral_curves"], dtype=np.float64))
            },
            "npz_best_parity_key": str(z["best_parity_key"].item()),
            "npz_best_spectral_key": str(z["best_spectral_key"].item()),
            "best_selection_budget_q": int(np.asarray(z["best_selection_budget_q"]).ravel()[0]),
            "seed": int(np.asarray(z["seed"]).ravel()[0]),
            "beta": float(np.asarray(z["beta"]).ravel()[0]),
        }

    run_config = json.loads(EXP4_RUN_CONFIG.read_text(encoding="utf-8"))
    payload["parity_key"] = str(run_config.get("best_parity_key", "sigma=1, K=512"))
    payload["spectral_key"] = str(run_config.get("comparison_spectral_key", "sigma=1, K=512"))
    return payload


def _load_exp16() -> dict[str, np.ndarray | int | float | list[int]]:
    with np.load(EXP16_NPZ, allow_pickle=True) as z:
        return {
            "q_grid": np.asarray(z["q_grid"], dtype=np.float64),
            "target_curve": np.asarray(z["target_mean"], dtype=np.float64),
            "target_std": np.asarray(z["target_std"], dtype=np.float64),
            "uniform_curve": np.asarray(z["uniform_mean"], dtype=np.float64),
            "uniform_std": np.asarray(z["uniform_std"], dtype=np.float64),
            "parity_sim_curve": np.asarray(z["parity_sim_mean"], dtype=np.float64),
            "parity_sim_std": np.asarray(z["parity_sim_std"], dtype=np.float64),
            "parity_hw_curve": np.asarray(z["parity_hw_mean"], dtype=np.float64),
            "parity_hw_std": np.asarray(z["parity_hw_std"], dtype=np.float64),
            "mse_sim_curve": np.asarray(z["mse_sim_mean"], dtype=np.float64),
            "mse_sim_std": np.asarray(z["mse_sim_std"], dtype=np.float64),
            "mse_hw_curve": np.asarray(z["mse_hw_mean"], dtype=np.float64),
            "mse_hw_std": np.asarray(z["mse_hw_std"], dtype=np.float64),
            "selected_seeds": [int(seed) for seed in np.asarray(z["selected_seeds"], dtype=np.int64).tolist()],
        }


def _red_shades(n: int) -> list[tuple[float, float, float, float]]:
    if n <= 0:
        return []
    start = np.array([0.98, 0.82, 0.82])
    stop = np.array([0.78, 0.12, 0.12])
    out = []
    for idx in range(n):
        t = idx / max(1, n - 1)
        c = (1.0 - t) * start + t * stop
        out.append((float(c[0]), float(c[1]), float(c[2]), 0.98))
    return out


def _gray_shades_by_budget(
    keys: list[str],
    curves_by_key: dict[str, np.ndarray],
    q: np.ndarray,
    budget_q: int,
) -> dict[str, tuple[float, float, float, float]]:
    if not keys:
        return {}
    values = []
    for key in keys:
        values.append((key, float(np.interp(float(budget_q), q, curves_by_key[key]))))
    values.sort(key=lambda item: item[1])
    start = np.array([0.87, 0.87, 0.87])
    stop = np.array([0.18, 0.18, 0.18])
    out: dict[str, tuple[float, float, float, float]] = {}
    for idx, (key, _value) in enumerate(values):
        t = idx / max(1, len(values) - 1)
        c = (1.0 - t) * start + t * stop
        out[key] = (float(c[0]), float(c[1]), float(c[2]), 0.98)
    return out


def _style_ax(ax: plt.Axes, qmax: float) -> None:
    ax.set_xlim(0.0, qmax)
    ax.set_ylim(0.0, 0.9)
    ax.set_xlabel("Number of samples", labelpad=1.0)
    ax.set_ylabel(r"$R(Q)$", labelpad=1.2)
    ax.set_xticks([0.0, qmax / 2.0, qmax])
    ax.set_yticks([0.0, 0.5, 0.9])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.45, alpha=0.38)
    ax.tick_params(axis="both", which="major", pad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.set_box_aspect(1.0)


def _legend(ax: plt.Axes, handles: list[Line2D], *, ncol: int = 1) -> None:
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#888888",
        framealpha=0.94,
        borderpad=0.15,
        labelspacing=0.07,
        handlelength=1.08,
        handletextpad=0.28,
        borderaxespad=0.20,
        columnspacing=0.42,
        ncol=ncol,
        fontsize=6.35 if ncol == 1 else 5.85,
    )
    leg.get_frame().set_linewidth(0.45)
    leg.set_zorder(100)


def _panel_caption(fig: plt.Figure, x: float, text: str) -> None:
    fig.text(x, 0.070, text, ha="center", va="center", fontsize=7.65, color=TEXT_DARK)


def _render_panel_a(ax: plt.Axes, exp4: dict[str, object]) -> None:
    q = np.asarray(exp4["Q"], dtype=np.float64)
    target = np.asarray(exp4["target_curve"], dtype=np.float64)
    uniform = np.asarray(exp4["uniform_curve"], dtype=np.float64)
    parity_by_key = exp4["parity_by_key"]
    spectral_by_key = exp4["spectral_by_key"]
    parity_key = str(exp4["parity_key"])
    spectral_key = str(exp4["spectral_key"])

    ax.plot(q, target, color=TARGET_COLOR, lw=2.05, zorder=5)
    ax.plot(q, uniform, color=UNIFORM_COLOR, lw=1.55, ls=":", zorder=3)
    ax.plot(q, spectral_by_key[spectral_key], color=SPECTRAL_COLOR, lw=1.90, ls="-.", zorder=4)
    ax.plot(q, parity_by_key[parity_key], color=PARITY_COLOR, lw=2.45, zorder=6)
    _style_ax(ax, float(q[-1]))
    _legend(
        ax,
        [
            Line2D([0], [0], color=TARGET_COLOR, lw=2.05, label=r"Target $p^*$"),
            Line2D([0], [0], color=PARITY_COLOR, lw=2.45, label=IQP_PARITY_LABEL),
            Line2D([0], [0], color=SPECTRAL_COLOR, lw=1.90, ls="-.", label="Spectral"),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.55, ls=":", label="Uniform"),
        ],
    )


def _render_panel_b(ax: plt.Axes, exp4: dict[str, object]) -> None:
    q = np.asarray(exp4["Q"], dtype=np.float64)
    target = np.asarray(exp4["target_curve"], dtype=np.float64)
    uniform = np.asarray(exp4["uniform_curve"], dtype=np.float64)
    iqp_mse = np.asarray(exp4["iqp_mse_curve"], dtype=np.float64)
    parity_by_key = exp4["parity_by_key"]
    parity_key = str(exp4["parity_key"])
    other_keys = [key for key in parity_by_key if key != parity_key]
    shades = _red_shades(len(other_keys))

    ax.plot(q, target, color=TARGET_COLOR, lw=2.05, zorder=6)
    ax.plot(q, uniform, color=UNIFORM_COLOR, lw=1.55, ls=":", zorder=3)
    for key, color in zip(other_keys, shades):
        ax.plot(q, parity_by_key[key], color=color, lw=1.05, alpha=0.82, zorder=2)
    ax.plot(q, parity_by_key[parity_key], color=PARITY_COLOR, lw=2.45, zorder=7)
    ax.plot(q, iqp_mse, color=MSE_PANEL_COLOR, lw=2.05, zorder=5)
    _style_ax(ax, float(q[-1]))
    _legend(
        ax,
        [
            Line2D([0], [0], color=TARGET_COLOR, lw=2.05, label=r"Target $p^*$"),
            Line2D([0], [0], color=PARITY_COLOR, lw=2.45, label=IQP_PARITY_LABEL),
            Line2D([0], [0], color=PARITY_CONTEXT_COLOR, lw=1.10, label="other parity"),
            Line2D([0], [0], color=MSE_PANEL_COLOR, lw=2.05, label=IQP_MSE_LABEL),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.55, ls=":", label="Uniform"),
        ],
    )


def _render_panel_c(ax: plt.Axes, exp4: dict[str, object]) -> None:
    q = np.asarray(exp4["Q"], dtype=np.float64)
    target = np.asarray(exp4["target_curve"], dtype=np.float64)
    uniform = np.asarray(exp4["uniform_curve"], dtype=np.float64)
    spectral_by_key = exp4["spectral_by_key"]
    spectral_key = str(exp4["spectral_key"])
    budget_q = int(exp4["best_selection_budget_q"])
    other_keys = [key for key in spectral_by_key if key != spectral_key]
    spectral_colors = _gray_shades_by_budget(list(spectral_by_key), spectral_by_key, q, budget_q)

    ax.plot(q, target, color=TARGET_COLOR, lw=2.05, zorder=6)
    ax.plot(q, uniform, color=UNIFORM_COLOR, lw=1.55, ls=":", zorder=3)
    for key in other_keys:
        ax.plot(q, spectral_by_key[key], color=spectral_colors[key], lw=1.05, alpha=0.83, zorder=2)
    ax.plot(q, spectral_by_key[spectral_key], color=SPECTRAL_COLOR, lw=2.05, ls="-.", zorder=7)
    _style_ax(ax, float(q[-1]))
    _legend(
        ax,
        [
            Line2D([0], [0], color=TARGET_COLOR, lw=2.05, label=r"Target $p^*$"),
            Line2D([0], [0], color=SPECTRAL_COLOR, lw=2.05, ls="-.", label="Spectral"),
            Line2D([0], [0], color=SPECTRAL_CONTEXT_COLOR, lw=1.10, label="other spectral"),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.55, ls=":", label="Uniform"),
        ],
    )


def _std_band(ax: plt.Axes, q: np.ndarray, mean: np.ndarray, std: np.ndarray, color: str, alpha: float, zorder: int) -> None:
    if np.any(std > 0.0):
        ax.fill_between(
            q,
            np.maximum(0.0, mean - std),
            mean + std,
            color=color,
            alpha=alpha,
            linewidth=0.0,
            zorder=zorder,
        )


def _render_panel_d(ax: plt.Axes, exp16: dict[str, np.ndarray | int | float | list[int]]) -> None:
    q = np.asarray(exp16["q_grid"], dtype=np.float64)
    parity_hw = np.asarray(exp16["parity_hw_curve"], dtype=np.float64)
    mse_hw = np.asarray(exp16["mse_hw_curve"], dtype=np.float64)
    parity_sim = np.asarray(exp16["parity_sim_curve"], dtype=np.float64)
    mse_sim = np.asarray(exp16["mse_sim_curve"], dtype=np.float64)

    _std_band(ax, q, parity_hw, np.asarray(exp16["parity_hw_std"], dtype=np.float64), PARITY_COLOR, 0.12, 1)
    _std_band(ax, q, mse_hw, np.asarray(exp16["mse_hw_std"], dtype=np.float64), MSE_COLOR, 0.12, 1)
    _std_band(ax, q, parity_sim, np.asarray(exp16["parity_sim_std"], dtype=np.float64), PARITY_COLOR, 0.07, 1)
    _std_band(ax, q, mse_sim, np.asarray(exp16["mse_sim_std"], dtype=np.float64), MSE_COLOR, 0.07, 1)

    ax.plot(q, np.asarray(exp16["target_curve"], dtype=np.float64), color=TARGET_COLOR, lw=2.05, zorder=7)
    ax.plot(q, np.asarray(exp16["uniform_curve"], dtype=np.float64), color=UNIFORM_COLOR, lw=1.55, ls=":", zorder=3)
    ax.plot(q, parity_hw, color=PARITY_COLOR, lw=2.45, zorder=6)
    ax.plot(q, mse_hw, color=MSE_COLOR, lw=2.25, zorder=5)
    ax.plot(
        q,
        parity_sim,
        color=PARITY_COLOR,
        lw=1.80,
        ls="--",
        alpha=0.78,
        zorder=5,
    )
    ax.plot(
        q,
        mse_sim,
        color=MSE_COLOR,
        lw=1.70,
        ls="--",
        alpha=0.78,
        zorder=4,
    )
    _style_ax(ax, float(q[-1]))
    _legend(
        ax,
        [
            Line2D([0], [0], color=TARGET_COLOR, lw=2.05, label=r"Target $p^*$"),
            Line2D([0], [0], color=PARITY_COLOR, lw=2.45, label=f"{IQP_PARITY_LABEL} hw"),
            Line2D([0], [0], color=MSE_COLOR, lw=2.25, label=f"{IQP_MSE_LABEL} hw"),
            Line2D([0], [0], color=PARITY_COLOR, lw=1.80, ls="--", label=f"{IQP_PARITY_LABEL} sim"),
            Line2D([0], [0], color=MSE_COLOR, lw=1.70, ls="--", label=f"{IQP_MSE_LABEL} sim"),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.55, ls=":", label="Uniform"),
        ],
    )


def render(outdir: Path, *, use_tex: bool) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    exp4 = _load_exp4()
    exp16 = _load_exp16()

    apply_ieee_latex_style(use_tex=use_tex)
    plt.rcParams.update(
        {
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.35,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.45,
        }
    )

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=False)
    axes = [
        fig.add_axes([0.052, 0.285, 0.190, 0.676]),
        fig.add_axes([0.299, 0.285, 0.190, 0.676]),
        fig.add_axes([0.546, 0.285, 0.190, 0.676]),
        fig.add_axes([0.793, 0.285, 0.190, 0.676]),
    ]

    _render_panel_a(axes[0], exp4)
    _render_panel_b(axes[1], exp4)
    _render_panel_c(axes[2], exp4)
    _render_panel_d(axes[3], exp16)

    _panel_caption(fig, 0.147, r"(a) IQP-parity vs. spectral proxy")
    _panel_caption(fig, 0.394, r"(b) IQP-parity family vs. IQP-MSE")
    _panel_caption(fig, 0.641, r"(c) Spectral family sweep")
    _panel_caption(fig, 0.888, r"(d) HW: IQP-parity vs. IQP-MSE")

    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}.png"
    save_exact_figure(fig, out_pdf)
    fig.savefig(out_png, format="png", dpi=500, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

    metadata = {
        "script": Path(__file__).name,
        "output_pdf": _rel(out_pdf),
        "output_png": _rel(out_png),
        "natural_width_pt": TEXTWIDTH_PT,
        "natural_height_pt": FIG_H_IN * TEX_PT_PER_IN,
        "intended_latex_include": r"\includegraphics[width=1.0\textwidth]{...}",
        "use_tex": bool(use_tex),
        "source_panel_a_pdf": _rel(EXP4_DIR / "experiment_4_recovery_best_iqp_vs_best_spectral.pdf"),
        "source_panel_b_pdf": _rel(EXP4_DIR / "experiment_4_recovery_parity_sigmak_vs_iqp_mse.pdf"),
        "source_panel_c_pdf": _rel(EXP4_DIR / "experiment_4_recovery_spectral_sigmak_only.pdf"),
        "source_panel_d_pdf": _rel(EXP16_NPZ.with_suffix(".pdf")),
        "source_exp4_npz": _rel(EXP4_NPZ),
        "source_exp4_run_config": _rel(EXP4_RUN_CONFIG),
        "source_exp16_npz": _rel(EXP16_NPZ),
        "exp4_parity_key_used": str(exp4["parity_key"]),
        "exp4_spectral_key_used": str(exp4["spectral_key"]),
        "exp16_selected_seeds": exp16["selected_seeds"],
        "legend_simplification": "Removed sigma/K parentheticals, enlarged legends, and standardized IQP-parity/IQP-MSE hardware-simulation labels.",
    }
    (outdir / f"{OUTPUT_STEM}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Render aligned textwidth recovery four-panel figure.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--use-tex", type=int, default=1)
    args = ap.parse_args()
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    out_pdf = render(outdir, use_tex=bool(int(args.use_tex)))
    print(f"[saved] {out_pdf}")


if __name__ == "__main__":
    main()
