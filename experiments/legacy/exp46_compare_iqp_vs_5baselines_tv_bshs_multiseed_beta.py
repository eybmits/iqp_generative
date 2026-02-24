#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TV-vs-BSHS comparison plots over seeds and betas.

Produces:
  1) Raw scatter over (beta, seed) points.
  2) Seed-mean scatter (one point per model+beta).
  3) Single-beta seed-mean scatter (e.g. beta=0.9).
  4) Bucket profile plot (target vs train-with-holdout-removed vs models).

Supports reusing an existing points CSV to avoid retraining all runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.legacy import exp11_beta_sweep_global_holdout as exp11  # noqa: E402
from experiments.legacy import exp45_compare_iqp_vs_5baselines_composite as exp45  # noqa: E402
from iqp_generative import core as hv  # noqa: E402


MODEL_MARKERS: Dict[str, str] = {
    "iqp_parity_mse": "o",
    "iqp_prob_mse": "o",
    "classical_nnn_fields_parity": "s",
    "classical_dense_fields_xent": "D",
    "classical_transformer_mle": "^",
    "classical_maxent_parity": "X",
}


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _parse_list_strs(s: str) -> List[str]:
    return [str(x.strip()) for x in s.split(",") if str(x).strip()]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _fmt_beta_tag(beta: float) -> str:
    return f"{float(beta):.2f}".replace("-", "m").replace(".", "p")


def _with_tag(stem: str, tag: str) -> str:
    t = str(tag).strip()
    return f"{stem}_{t}" if t else stem


def _model_spec_maps(selected_specs: Sequence[Tuple[str, str, str, object, float]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    label_by_key = {k: lbl for (k, lbl, _c, _ls, _lw) in selected_specs}
    color_by_key = {k: c for (k, _lbl, c, _ls, _lw) in selected_specs}
    return label_by_key, color_by_key


def _resolve_model_specs(include_keys_raw: str, exclude_keys_raw: str) -> List[Tuple[str, str, str, object, float]]:
    all_specs = list(exp11.MODEL_SPECS)
    all_keys = [k for (k, _lbl, _c, _ls, _lw) in all_specs]

    include_keys = _parse_list_strs(include_keys_raw)
    exclude_keys = set(_parse_list_strs(exclude_keys_raw))

    if include_keys:
        include_set = set(include_keys)
    else:
        include_set = set(all_keys)

    selected = [s for s in all_specs if s[0] in include_set and s[0] not in exclude_keys]
    if not selected:
        raise ValueError("No models selected after include/exclude filtering.")

    return selected


def _score_levels(scores: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    s_int = scores.astype(np.int64)
    return np.sort(np.unique(s_int[support_mask]))


def _mass_by_level(probs: np.ndarray, scores: np.ndarray, levels: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    s_int = scores.astype(np.int64)
    out = np.zeros(levels.shape[0], dtype=np.float64)
    for i, lv in enumerate(levels):
        m = support_mask & (s_int == int(lv))
        out[i] = float(np.sum(probs[m]))
    return out


def _normalized(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    s = float(np.sum(v))
    if s <= eps:
        return np.zeros_like(v, dtype=np.float64)
    return v / s


def _train_one_beta_seed(args: argparse.Namespace, beta: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    p_star, scores, holdout_mask, model_rows, _ = exp11._train_models_for_beta(
        holdout_mode=str(args.holdout_mode),
        n=int(args.n),
        beta=float(beta),
        seed=int(seed),
        train_m=int(args.train_m),
        sigma=float(args.sigma),
        K=int(args.K),
        layers=int(args.layers),
        holdout_k=int(args.holdout_k),
        holdout_pool=int(args.holdout_pool),
        holdout_m_train=int(args.holdout_m_train),
        good_frac=float(args.good_frac),
        iqp_steps=int(args.iqp_steps),
        iqp_lr=float(args.iqp_lr),
        iqp_eval_every=int(args.iqp_eval_every),
        q80_thr=float(args.q80_thr),
        q80_search_max=int(args.q80_search_max),
        artr_epochs=int(args.artr_epochs),
        artr_d_model=int(args.artr_d_model),
        artr_heads=int(args.artr_heads),
        artr_layers=int(args.artr_layers),
        artr_ff=int(args.artr_ff),
        artr_lr=float(args.artr_lr),
        artr_batch_size=int(args.artr_batch_size),
        maxent_steps=int(args.maxent_steps),
        maxent_lr=float(args.maxent_lr),
    )
    return p_star, scores, holdout_mask, model_rows


def _collect_bucket_rows_for_run(
    p_star: np.ndarray,
    scores: np.ndarray,
    holdout_mask: np.ndarray,
    model_rows: List[Dict[str, object]],
    beta: float,
    seed: int,
    selected_keys: set[str],
    label_by_key: Dict[str, str],
) -> List[Dict[str, object]]:
    support_mask = p_star > 0.0
    levels = _score_levels(scores, support_mask)

    p_train = p_star.copy()
    p_train[holdout_mask] = 0.0
    p_train /= max(1e-15, float(np.sum(p_train)))

    series_probs: Dict[str, np.ndarray] = {
        "target": p_star,
        "p_train_removed": p_train,
    }
    series_labels: Dict[str, str] = {
        "target": r"Target $p^*$",
        "p_train_removed": r"Train $p_{train}$ (holdout removed)",
    }

    for mr in model_rows:
        key = str(mr["key"])
        if key not in selected_keys:
            continue
        q = np.asarray(mr["q"], dtype=np.float64)
        series_probs[key] = q
        series_labels[key] = label_by_key.get(key, key)

    rows: List[Dict[str, object]] = []
    for key, probs in series_probs.items():
        level_mass = _mass_by_level(probs=probs, scores=scores, levels=levels, support_mask=support_mask)
        level_share = _normalized(level_mass)
        support_mass_total = float(np.sum(level_mass))
        for i, lv in enumerate(levels):
            rows.append(
                {
                    "beta": float(beta),
                    "seed": int(seed),
                    "series_key": str(key),
                    "series_label": str(series_labels[key]),
                    "score_level": int(lv),
                    "mass_in_support": float(level_mass[i]),
                    "share_in_support": float(level_share[i]),
                    "support_mass_total": float(support_mass_total),
                }
            )
    return rows


def _plot_tv_bshs_budgetlaw_style(
    rows: List[Dict[str, object]],
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    title_suffix: str,
    selected_specs: Sequence[Tuple[str, str, str, object, float]],
    show_title: bool = True,
) -> None:
    if not rows:
        raise RuntimeError("No rows to plot.")

    df = pd.DataFrame(rows)
    model_specs = {k: (lbl, color, ls, lw) for (k, lbl, color, ls, lw) in selected_specs}

    betas = df["beta"].astype(float).to_numpy()
    bmin = float(np.min(betas))
    bmax = float(np.max(betas))
    if bmax <= bmin:
        bmax = bmin + 1e-6

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.3, 6.1), dpi=dpi)
    ax.set_facecolor("#f7f7f7")

    norm = Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap("Reds")

    for model_key, model_df in df.groupby("model_key", sort=False):
        if model_key not in model_specs:
            continue
        _label, edge_color, _ls, _lw = model_specs.get(model_key, (model_key, "#444444", "-", 1.5))
        marker = MODEL_MARKERS.get(str(model_key), "o")

        x = model_df["TV_score"].astype(float).to_numpy()
        y = model_df["BSHS"].astype(float).to_numpy()
        c = model_df["beta"].astype(float).to_numpy()

        ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            norm=norm,
            s=95,
            marker=marker,
            edgecolors=edge_color,
            linewidths=1.4,
            alpha=0.88,
            zorder=4,
        )

    ax.grid(which="major", linestyle="-", color="#d8d8d8", alpha=0.6, linewidth=0.65)
    ax.grid(which="minor", linestyle="-", color="#e6e6e6", alpha=0.45, linewidth=0.45)
    ax.minorticks_on()

    ax.set_xlabel(r"TV$_{score}$ (lower better)")
    ax.set_ylabel(r"BSHS$(Q)$ (higher better)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)

    if show_title and title_suffix:
        ax.set_title(f"IQP parity vs 5 baselines | {title_suffix}")

    legend_handles: List[Line2D] = []
    present_keys = set(df["model_key"].astype(str).tolist())
    for key, label, color, _ls, _lw in selected_specs:
        if key not in present_keys:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS.get(key, "o"),
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.8,
                markersize=8.5,
                label=label,
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Model",
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#c0c0c0",
            framealpha=0.98,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\beta$")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=dpi)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_tv_bshs_seedmean_by_beta(
    df_mean: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    selected_specs: Sequence[Tuple[str, str, str, object, float]],
    q_eval: int,
    holdout_mode: str,
    train_m: int,
    draw_pareto_front: bool = False,
    pareto_only: bool = False,
    draw_model_trends: bool = True,
    show_title: bool = True,
) -> None:
    if df_mean.empty:
        raise RuntimeError("No seed-mean rows to plot.")

    model_specs = {k: (lbl, color, ls, lw) for (k, lbl, color, ls, lw) in selected_specs}
    x_all = df_mean["TV_score_mean"].astype(float).to_numpy()
    y_all = df_mean["BSHS_mean"].astype(float).to_numpy()
    pareto_mask = _pareto_mask_minx_maxy(x_all, y_all)
    df_plot = df_mean.loc[pareto_mask].copy() if pareto_only else df_mean.copy()

    betas = df_plot["beta"].astype(float).to_numpy()
    bmin = float(np.min(betas))
    bmax = float(np.max(betas))
    if bmax <= bmin:
        bmax = bmin + 1e-6

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.3, 6.0), dpi=dpi)
    ax.set_facecolor("white")

    norm = Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap("Reds")

    for model_key, g in df_plot.groupby("model_key", sort=False):
        if model_key not in model_specs:
            continue
        _lbl, edge_color, _ls, _lw = model_specs[model_key]
        marker = MODEL_MARKERS.get(str(model_key), "o")

        g = g.sort_values("beta")
        x = g["TV_score_mean"].astype(float).to_numpy()
        y = g["BSHS_mean"].astype(float).to_numpy()
        c = g["beta"].astype(float).to_numpy()

        if draw_model_trends:
            # Thin trend line to make beta progression per model readable.
            ax.plot(x, y, color=edge_color, linewidth=1.1, alpha=0.45, zorder=2)

        ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            norm=norm,
            s=135,
            marker=marker,
            edgecolors=edge_color,
            linewidths=1.8,
            alpha=0.95,
            zorder=4,
        )

    if draw_pareto_front:
        df_pf = df_mean.loc[pareto_mask].copy().sort_values("TV_score_mean")
        if len(df_pf) >= 1:
            ax.plot(
                df_pf["TV_score_mean"].astype(float).to_numpy(),
                df_pf["BSHS_mean"].astype(float).to_numpy(),
                color="#202020",
                linewidth=2.2,
                linestyle="-",
                alpha=0.90,
                zorder=6,
                label="Pareto front",
            )

    ax.grid(True, which="major", linestyle="-", color="#d5d5d5", alpha=0.65, linewidth=0.65)
    ax.grid(True, which="minor", linestyle="-", color="#ececec", alpha=0.6, linewidth=0.45)
    ax.minorticks_on()

    ax.set_xlabel(r"TV$_{score}$ (lower better)")
    ax.set_ylabel(r"BSHS$(Q)$ (higher better)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    if show_title:
        title = f"Seed-mean model comparison | holdout={holdout_mode}, m={train_m}, Q={q_eval}"
        if pareto_only:
            title += " | Pareto-only"
        ax.set_title(title)

    legend_handles: List[Line2D] = []
    present_keys = set(df_plot["model_key"].astype(str).tolist())
    for key, label, color, _ls, _lw in selected_specs:
        if key not in present_keys:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS.get(key, "o"),
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2.0,
                markersize=8.8,
                label=label,
            )
        )
    if draw_pareto_front:
        legend_handles.append(Line2D([0], [0], color="#202020", lw=2.2, linestyle="-", label="Pareto front"))

    ax.legend(
        handles=legend_handles,
        title="Model",
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#bfbfbf",
        framealpha=0.98,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\beta$")

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=dpi)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_tv_bshs_single_beta_seedmean(
    df_single: pd.DataFrame,
    beta: float,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    selected_specs: Sequence[Tuple[str, str, str, object, float]],
    q_eval: int,
    draw_pareto_front: bool = False,
    pareto_only: bool = False,
    show_title: bool = True,
) -> None:
    if df_single.empty:
        raise RuntimeError("No rows for single-beta seed-mean plot.")

    model_specs = {k: (lbl, color, ls, lw) for (k, lbl, color, ls, lw) in selected_specs}
    x_all = df_single["TV_score_mean"].astype(float).to_numpy()
    y_all = df_single["BSHS_mean"].astype(float).to_numpy()
    pareto_mask = _pareto_mask_minx_maxy(x_all, y_all)
    df_plot = df_single.loc[pareto_mask].copy() if pareto_only else df_single.copy()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=dpi)
    ax.set_facecolor("white")

    short_by_key = dict(exp11.MODEL_LABEL_SHORT)

    for _, r in df_plot.iterrows():
        key = str(r["model_key"])
        if key not in model_specs:
            continue
        _lbl, edge_color, _ls, _lw = model_specs[key]
        marker = MODEL_MARKERS.get(key, "o")

        x = float(r["TV_score_mean"])
        y = float(r["BSHS_mean"])

        ax.scatter(
            [x],
            [y],
            s=180,
            marker=marker,
            facecolors="white",
            edgecolors=edge_color,
            linewidths=2.4,
            zorder=5,
        )
        ax.annotate(
            short_by_key.get(key, key),
            (x, y),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=10,
            color=edge_color,
        )

    if draw_pareto_front:
        df_pf = df_single.loc[pareto_mask].copy().sort_values("TV_score_mean")
        if len(df_pf) >= 1:
            ax.plot(
                df_pf["TV_score_mean"].astype(float).to_numpy(),
                df_pf["BSHS_mean"].astype(float).to_numpy(),
                color="#202020",
                linewidth=2.2,
                linestyle="-",
                alpha=0.92,
                zorder=4,
                label="Pareto front",
            )

    ax.grid(True, which="major", linestyle="-", color="#d7d7d7", alpha=0.65, linewidth=0.65)
    ax.grid(True, which="minor", linestyle="-", color="#ededed", alpha=0.6, linewidth=0.45)
    ax.minorticks_on()

    ax.set_xlabel(r"TV$_{score}$ (lower better)")
    ax.set_ylabel(r"BSHS$(Q)$ (higher better)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    if show_title:
        title = fr"Single-$\beta$ seed-mean comparison ($\beta$={float(beta):g}, Q={int(q_eval)})"
        if pareto_only:
            title += " | Pareto-only"
        ax.set_title(title)

    legend_handles: List[Line2D] = []
    present_keys = set(df_plot["model_key"].astype(str).tolist())
    for key, label, color, _ls, _lw in selected_specs:
        if key not in present_keys:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS.get(key, "o"),
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2.0,
                markersize=8.8,
                label=label,
            )
        )
    if draw_pareto_front:
        legend_handles.append(Line2D([0], [0], color="#202020", lw=2.2, linestyle="-", label="Pareto front"))

    ax.legend(
        handles=legend_handles,
        title="Model",
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#bfbfbf",
        framealpha=0.98,
    )

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=dpi)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_bucket_profile_seedmean(
    df_bucket_mean: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    selected_specs: Sequence[Tuple[str, str, str, object, float]],
    beta: float,
    seed_count: int,
    show_title: bool = True,
) -> None:
    if df_bucket_mean.empty:
        raise RuntimeError("No bucket rows to plot.")

    color_by_key = {k: c for (k, _lbl, c, _ls, _lw) in selected_specs}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(10.2, 6.1), dpi=dpi)
    ax.set_facecolor("white")

    order: List[str] = ["target", "p_train_removed"] + [k for (k, _lbl, _c, _ls, _lw) in selected_specs]
    levels = sorted(df_bucket_mean["score_level"].astype(int).unique().tolist())
    x_base = np.arange(len(levels), dtype=np.float64)

    series_keys: List[str] = [k for k in order if k in set(df_bucket_mean["series_key"].astype(str).tolist())]
    n_series = max(1, len(series_keys))
    group_width = 0.86
    bar_w = group_width / float(n_series)
    offsets = (np.arange(n_series, dtype=np.float64) - 0.5 * (n_series - 1)) * bar_w

    for j, key in enumerate(series_keys):
        g = df_bucket_mean[df_bucket_mean["series_key"] == key].sort_values("score_level")
        if g.empty:
            continue

        label = str(g["series_label"].iloc[0])
        g_idx = g.set_index(g["score_level"].astype(int))

        y = np.array([float(g_idx.loc[lv, "share_mean"]) if lv in g_idx.index else 0.0 for lv in levels], dtype=np.float64)
        ylo = np.array([float(g_idx.loc[lv, "share_ci_lo"]) if lv in g_idx.index else 0.0 for lv in levels], dtype=np.float64)
        yhi = np.array([float(g_idx.loc[lv, "share_ci_hi"]) if lv in g_idx.index else 0.0 for lv in levels], dtype=np.float64)

        if key == "target":
            color = "#222222"
            alpha = 0.96
            z = 7
        elif key == "p_train_removed":
            color = "#8a8a8a"
            alpha = 0.92
            z = 6
        else:
            color = color_by_key.get(key, "#666666")
            alpha = 0.84
            z = 5

        x = x_base + offsets[j]
        ax.bar(
            x,
            y,
            width=bar_w * 0.94,
            color=color,
            alpha=alpha,
            label=label,
            edgecolor="white",
            linewidth=0.30,
            zorder=z,
        )

        # Keep uncertainty visible but unobtrusive.
        err_lo = np.clip(y - ylo, 0.0, None)
        err_hi = np.clip(yhi - y, 0.0, None)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([err_lo, err_hi]),
            fmt="none",
            ecolor=color,
            elinewidth=0.65,
            capsize=1.4,
            alpha=0.36,
            zorder=max(1, z - 1),
        )

    ax.grid(True, which="major", linestyle="-", color="#d7d7d7", alpha=0.65, linewidth=0.65)
    ax.grid(True, which="minor", linestyle="-", color="#ededed", alpha=0.6, linewidth=0.45)
    ax.minorticks_on()

    ax.set_xlabel("Score level s")
    ax.set_ylabel("Share inside support")
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlim(left=-0.6, right=float(len(levels) - 1) + 0.6)
    ax.set_ylim(0.0, max(0.12, float(df_bucket_mean["share_ci_hi"].max()) * 1.12))
    if show_title:
        ax.set_title(fr"Bucket profile vs target/train ($\beta$={float(beta):g}, seeds={int(seed_count)})")

    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#bfbfbf", ncol=2)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=dpi)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _aggregate_seed_mean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["model_key", "model_label", "beta"], as_index=False)
        .agg(
            seeds_n=("seed", "nunique"),
            TV_score_mean=("TV_score", "mean"),
            TV_score_std=("TV_score", "std"),
            BSHS_mean=("BSHS", "mean"),
            BSHS_std=("BSHS", "std"),
            Composite_mean=("Composite", "mean"),
            Composite_std=("Composite", "std"),
        )
        .sort_values(["model_key", "beta"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return out


def _nearest_beta(available: Sequence[float], target: float) -> float:
    arr = np.asarray(list(available), dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError("No available beta values.")
    i = int(np.argmin(np.abs(arr - float(target))))
    return float(arr[i])


def _pareto_mask_minx_maxy(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Non-dominated mask for min-x / max-y objectives."""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    n = int(xv.size)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        xi = float(xv[i])
        yi = float(yv[i])
        dominated = False
        for j in range(n):
            if i == j:
                continue
            xj = float(xv[j])
            yj = float(yv[j])
            if (xj <= xi + eps) and (yj >= yi - eps) and ((xj < xi - eps) or (yj > yi + eps)):
                dominated = True
                break
        keep[i] = not dominated
    return keep


def _collect_rows_and_optional_bucket(
    args: argparse.Namespace,
    betas: Sequence[float],
    seeds: Sequence[int],
    selected_keys: set[str],
    label_by_key: Dict[str, str],
    bucket_beta: Optional[float],
    bucket_tol: float,
    collect_bucket: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    bucket_rows: List[Dict[str, object]] = []

    total_runs = len(betas) * len(seeds)
    run_idx = 0
    for beta in betas:
        for seed in seeds:
            run_idx += 1
            print(f"[run {run_idx}/{total_runs}] beta={beta:g} seed={seed}")

            p_star, scores, holdout_mask, model_rows = _train_one_beta_seed(args, beta=beta, seed=seed)
            support_mask = p_star > 0.0
            q_eval = int(args.q_eval)

            for mr in model_rows:
                key = str(mr["key"])
                if key not in selected_keys:
                    continue
                q = np.asarray(mr["q"], dtype=np.float64)
                m = exp45.compute_support_bucket_metrics(
                    p_star=p_star,
                    q=q,
                    support_mask=support_mask,
                    scores=scores,
                    q_eval=q_eval,
                )
                rows.append(
                    {
                        "model_key": key,
                        "model_label": str(mr["label"]),
                        "TV_score": float(m["TV_score"]),
                        "BSHS": float(m["BSHS"]),
                        "Composite": float(m["Composite"]),
                        "beta": float(beta),
                        "seed": int(seed),
                        "q_eval": int(q_eval),
                        "holdout_mode": str(args.holdout_mode),
                        "train_m": int(args.train_m),
                        "sigma": float(args.sigma),
                        "K": int(args.K),
                    }
                )

            if collect_bucket and (bucket_beta is not None) and (abs(float(beta) - float(bucket_beta)) <= float(bucket_tol)):
                bucket_rows.extend(
                    _collect_bucket_rows_for_run(
                        p_star=p_star,
                        scores=scores,
                        holdout_mask=holdout_mask,
                        model_rows=model_rows,
                        beta=float(beta),
                        seed=int(seed),
                        selected_keys=selected_keys,
                        label_by_key=label_by_key,
                    )
                )

    return rows, bucket_rows


def _collect_bucket_rows_for_beta_only(
    args: argparse.Namespace,
    beta: float,
    seeds: Sequence[int],
    selected_keys: set[str],
    label_by_key: Dict[str, str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i, seed in enumerate(seeds, start=1):
        print(f"[bucket {i}/{len(seeds)}] beta={beta:g} seed={seed}")
        p_star, scores, holdout_mask, model_rows = _train_one_beta_seed(args, beta=float(beta), seed=int(seed))
        out.extend(
            _collect_bucket_rows_for_run(
                p_star=p_star,
                scores=scores,
                holdout_mask=holdout_mask,
                model_rows=model_rows,
                beta=float(beta),
                seed=int(seed),
                selected_keys=selected_keys,
                label_by_key=label_by_key,
            )
        )
    return out


def run(args: argparse.Namespace) -> None:
    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")
    if not exp11.exp10.HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    hv.set_style(base=8)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_specs = _resolve_model_specs(
        include_keys_raw=str(args.include_model_keys),
        exclude_keys_raw=str(args.exclude_model_keys),
    )
    selected_keys = {k for (k, _lbl, _c, _ls, _lw) in selected_specs}
    label_by_key, _color_by_key = _model_spec_maps(selected_specs)

    betas = _parse_list_floats(args.betas)
    seeds = _parse_list_ints(args.seeds)

    bucket_beta_req = float(args.bucket_beta) if args.bucket_beta is not None else float(args.single_beta)
    bucket_tol = float(args.single_beta_tol)

    rows: List[Dict[str, object]] = []
    bucket_rows: List[Dict[str, object]] = []

    points_csv_in = str(args.points_csv).strip()
    used_points_csv: Optional[Path] = None
    if points_csv_in:
        p = Path(points_csv_in)
        if not p.exists():
            raise FileNotFoundError(f"points CSV not found: {p}")
        print(f"[load] using points CSV: {p}")
        rows_loaded = _read_csv_rows(p)
        # Keep only selected model keys.
        rows = [r for r in rows_loaded if str(r.get("model_key", "")) in selected_keys]
        used_points_csv = p
        if not rows:
            raise RuntimeError("No rows left after model filtering on loaded points CSV.")
    else:
        rows, bucket_rows = _collect_rows_and_optional_bucket(
            args=args,
            betas=betas,
            seeds=seeds,
            selected_keys=selected_keys,
            label_by_key=label_by_key,
            bucket_beta=bucket_beta_req,
            bucket_tol=bucket_tol,
            collect_bucket=bool(args.make_bucket_profile),
        )

    if not rows:
        raise RuntimeError("No rows available for plotting.")

    # Ensure numeric columns are numeric for robust grouping.
    df = pd.DataFrame(rows).copy()
    for col in ["TV_score", "BSHS", "Composite", "beta", "seed", "q_eval", "train_m", "sigma", "K"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["TV_score", "BSHS", "Composite", "beta", "seed"]).reset_index(drop=True)

    # Save/refresh points CSV in outdir for reproducibility of postprocessed plots.
    points_csv = outdir / _with_tag("tv_bshs_points_multiseed_beta", str(args.tag))
    points_csv = points_csv.with_suffix(".csv")
    _write_csv(points_csv, [dict(r) for r in df.to_dict(orient="records")])

    # Model summary over all rows.
    summary_df = (
        df.groupby(["model_key", "model_label"], as_index=False)
        .agg(
            n=("Composite", "count"),
            TV_score_mean=("TV_score", "mean"),
            TV_score_std=("TV_score", "std"),
            BSHS_mean=("BSHS", "mean"),
            BSHS_std=("BSHS", "std"),
            Composite_mean=("Composite", "mean"),
            Composite_std=("Composite", "std"),
        )
        .sort_values("Composite_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary_csv = outdir / _with_tag("tv_bshs_summary_by_model", str(args.tag))
    summary_csv = summary_csv.with_suffix(".csv")
    summary_df.to_csv(summary_csv, index=False)

    # Raw scatter (all points) for compatibility.
    q_eval_ref = int(float(df["q_eval"].iloc[0])) if "q_eval" in df.columns else int(args.q_eval)
    betas_available = sorted(df["beta"].astype(float).unique().tolist())
    seeds_available = sorted(df["seed"].astype(int).unique().tolist())

    outputs: Dict[str, str] = {
        "points_csv": str(points_csv),
        "summary_csv": str(summary_csv),
    }

    if bool(args.make_raw_scatter):
        raw_pdf = outdir / _with_tag("tv_bshs_scatter_budgetlaw_style_multiseed_beta", str(args.tag))
        raw_png = raw_pdf.with_suffix(".png")
        raw_pdf = raw_pdf.with_suffix(".pdf")

        title_suffix = (
            f"global={args.holdout_mode}, m={int(args.train_m)}, Q={int(q_eval_ref)}, "
            f"betas={min(betas_available):g}..{max(betas_available):g}, seeds={len(seeds_available)}"
        )
        _plot_tv_bshs_budgetlaw_style(
            rows=[dict(r) for r in df.to_dict(orient="records")],
            out_pdf=raw_pdf,
            out_png=raw_png,
            dpi=int(args.dpi),
            title_suffix=title_suffix,
            selected_specs=selected_specs,
            show_title=not bool(args.hide_titles),
        )
        outputs["raw_scatter_pdf"] = str(raw_pdf)
        outputs["raw_scatter_png"] = str(raw_png)

    # Seed-mean aggregation by (model, beta).
    df_mean = _aggregate_seed_mean(df)
    seedmean_csv = outdir / _with_tag("tv_bshs_seedmean_by_model_beta", str(args.tag))
    seedmean_csv = seedmean_csv.with_suffix(".csv")
    df_mean.to_csv(seedmean_csv, index=False)
    outputs["seedmean_csv"] = str(seedmean_csv)

    if bool(args.make_seedmean_scatter):
        mean_pdf = outdir / _with_tag("tv_bshs_scatter_budgetlaw_style_multiseed_beta_seedmean", str(args.tag))
        mean_png = mean_pdf.with_suffix(".png")
        mean_pdf = mean_pdf.with_suffix(".pdf")

        _plot_tv_bshs_seedmean_by_beta(
            df_mean=df_mean,
            out_pdf=mean_pdf,
            out_png=mean_png,
            dpi=int(args.dpi),
            selected_specs=selected_specs,
            q_eval=int(q_eval_ref),
            holdout_mode=str(args.holdout_mode),
            train_m=int(args.train_m),
            draw_pareto_front=bool(args.draw_pareto_front),
            pareto_only=bool(args.pareto_only),
            draw_model_trends=bool(args.draw_model_trends),
            show_title=not bool(args.hide_titles),
        )
        outputs["seedmean_scatter_pdf"] = str(mean_pdf)
        outputs["seedmean_scatter_png"] = str(mean_png)

    # Single-beta seed-mean plot.
    beta_single_req = float(args.single_beta)
    beta_single = _nearest_beta(df_mean["beta"].astype(float).unique().tolist(), target=beta_single_req)
    if abs(beta_single - beta_single_req) > float(args.single_beta_tol):
        print(f"[info] requested single beta {beta_single_req:g}, using nearest available beta {beta_single:g}")

    df_single = df_mean[np.isclose(df_mean["beta"].astype(float), float(beta_single), atol=float(args.single_beta_tol), rtol=0.0)].copy()
    if df_single.empty:
        # robust fallback if strict atol does not catch floating representation
        df_single = df_mean[df_mean["beta"].astype(float) == float(beta_single)].copy()

    single_csv = outdir / _with_tag(f"tv_bshs_seedmean_beta_{_fmt_beta_tag(beta_single)}", str(args.tag))
    single_csv = single_csv.with_suffix(".csv")
    df_single.to_csv(single_csv, index=False)
    outputs["single_beta_seedmean_csv"] = str(single_csv)

    if bool(args.make_single_beta_scatter):
        single_pdf = outdir / _with_tag(f"tv_bshs_scatter_budgetlaw_style_beta_{_fmt_beta_tag(beta_single)}_seedmean", str(args.tag))
        single_png = single_pdf.with_suffix(".png")
        single_pdf = single_pdf.with_suffix(".pdf")

        _plot_tv_bshs_single_beta_seedmean(
            df_single=df_single,
            beta=float(beta_single),
            out_pdf=single_pdf,
            out_png=single_png,
            dpi=int(args.dpi),
            selected_specs=selected_specs,
            q_eval=int(q_eval_ref),
            draw_pareto_front=bool(args.draw_pareto_front),
            pareto_only=bool(args.pareto_only),
            show_title=not bool(args.hide_titles),
        )
        outputs["single_beta_scatter_pdf"] = str(single_pdf)
        outputs["single_beta_scatter_png"] = str(single_png)

    # Bucket profile plot at single beta (or explicit bucket beta).
    if bool(args.make_bucket_profile):
        if not bucket_rows:
            # If points were loaded from CSV, collect only the requested beta bucket rows now.
            bucket_rows = _collect_bucket_rows_for_beta_only(
                args=args,
                beta=float(bucket_beta_req),
                seeds=seeds,
                selected_keys=selected_keys,
                label_by_key=label_by_key,
            )

        df_bucket = pd.DataFrame(bucket_rows).copy()
        if not df_bucket.empty:
            for col in ["beta", "seed", "score_level", "share_in_support"]:
                df_bucket[col] = pd.to_numeric(df_bucket[col], errors="coerce")
            df_bucket = df_bucket.dropna(subset=["beta", "seed", "score_level", "share_in_support"]).reset_index(drop=True)

            beta_bucket = _nearest_beta(df_bucket["beta"].astype(float).unique().tolist(), target=float(bucket_beta_req))
            df_bucket = df_bucket[np.isclose(df_bucket["beta"].astype(float), float(beta_bucket), atol=float(args.single_beta_tol), rtol=0.0)].copy()

            if df_bucket.empty:
                raise RuntimeError("Bucket rows are empty after beta filtering.")

            bucket_raw_csv = outdir / _with_tag(f"tv_bshs_bucket_profiles_raw_beta_{_fmt_beta_tag(beta_bucket)}", str(args.tag))
            bucket_raw_csv = bucket_raw_csv.with_suffix(".csv")
            df_bucket.to_csv(bucket_raw_csv, index=False)

            def _q_lo(x: pd.Series) -> float:
                return float(np.quantile(x.to_numpy(np.float64), 0.16))

            def _q_hi(x: pd.Series) -> float:
                return float(np.quantile(x.to_numpy(np.float64), 0.84))

            df_bucket_mean = (
                df_bucket.groupby(["series_key", "series_label", "score_level"], as_index=False)
                .agg(
                    share_mean=("share_in_support", "mean"),
                    share_std=("share_in_support", "std"),
                    share_ci_lo=("share_in_support", _q_lo),
                    share_ci_hi=("share_in_support", _q_hi),
                    seeds_n=("seed", "nunique"),
                )
                .sort_values(["series_key", "score_level"], ascending=[True, True])
                .reset_index(drop=True)
            )

            bucket_mean_csv = outdir / _with_tag(f"tv_bshs_bucket_profiles_seedmean_beta_{_fmt_beta_tag(beta_bucket)}", str(args.tag))
            bucket_mean_csv = bucket_mean_csv.with_suffix(".csv")
            df_bucket_mean.to_csv(bucket_mean_csv, index=False)

            bucket_pdf = outdir / _with_tag(f"tv_bshs_bucket_profiles_beta_{_fmt_beta_tag(beta_bucket)}_seedmean", str(args.tag))
            bucket_png = bucket_pdf.with_suffix(".png")
            bucket_pdf = bucket_pdf.with_suffix(".pdf")

            _plot_bucket_profile_seedmean(
                df_bucket_mean=df_bucket_mean,
                out_pdf=bucket_pdf,
                out_png=bucket_png,
                dpi=int(args.dpi),
                selected_specs=selected_specs,
                beta=float(beta_bucket),
                seed_count=int(df_bucket["seed"].nunique()),
                show_title=not bool(args.hide_titles),
            )

            outputs["bucket_raw_csv"] = str(bucket_raw_csv)
            outputs["bucket_seedmean_csv"] = str(bucket_mean_csv)
            outputs["bucket_plot_pdf"] = str(bucket_pdf)
            outputs["bucket_plot_png"] = str(bucket_png)

    best = summary_df.iloc[0].to_dict() if len(summary_df) else {}
    summary_json = outdir / _with_tag("tv_bshs_summary_multiseed_beta", str(args.tag))
    summary_json = summary_json.with_suffix(".json")

    payload = {
        "config": {
            "n": int(args.n),
            "holdout_mode": str(args.holdout_mode),
            "train_m": int(args.train_m),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "betas_requested": [float(b) for b in betas],
            "betas_available": [float(b) for b in betas_available],
            "seeds_requested": [int(s) for s in seeds],
            "seeds_available": [int(s) for s in seeds_available],
            "q_eval": int(q_eval_ref),
            "q80_thr": float(args.q80_thr),
            "single_beta_requested": float(args.single_beta),
            "single_beta_used": float(beta_single),
            "bucket_beta_requested": float(bucket_beta_req),
            "models_included": [k for (k, _lbl, _c, _ls, _lw) in selected_specs],
            "models_excluded": _parse_list_strs(str(args.exclude_model_keys)),
            "draw_pareto_front": bool(args.draw_pareto_front),
            "pareto_only": bool(args.pareto_only),
            "points_csv_input": str(used_points_csv) if used_points_csv is not None else None,
        },
        "best_model_by_composite_mean": best,
        "outputs": outputs,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[saved] {points_csv}")
    print(f"[saved] {summary_csv}")
    for k in sorted(outputs.keys()):
        if k.endswith("_pdf") or k.endswith("_csv") or k.endswith("_png"):
            print(f"[saved] {outputs[k]}")
    print(f"[saved] {summary_json}")
    if best:
        print(
            "[best-mean]"
            f" {best.get('model_key')} | Composite_mean={float(best.get('Composite_mean', float('nan'))):.4f}"
            f" | BSHS_mean={float(best.get('BSHS_mean', float('nan'))):.4f}"
            f" | TV_mean={float(best.get('TV_score_mean', float('nan'))):.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TV-vs-BSHS multi-seed/beta comparison with seed-mean and bucket diagnostics.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "46_compare_iqp_vs_5baselines_tv_bshs_multiseed_beta"),
    )

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)

    ap.add_argument("--betas", type=str, default="0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--q-eval", type=int, default=2000)

    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)

    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)

    ap.add_argument("--iqp-steps", type=int, default=400)
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

    ap.add_argument("--points-csv", type=str, default="", help="Reuse existing points CSV instead of retraining all (beta,seed) runs.")
    ap.add_argument("--include-model-keys", type=str, default="", help="Comma-separated model keys to include (default: all).")
    ap.add_argument("--exclude-model-keys", type=str, default="", help="Comma-separated model keys to exclude.")

    ap.add_argument("--single-beta", type=float, default=0.9, help="Beta used for the single-beta seed-mean scatter.")
    ap.add_argument("--bucket-beta", type=float, default=None, help="Optional beta override for bucket-profile plot.")
    ap.add_argument("--single-beta-tol", type=float, default=1e-6)

    ap.add_argument("--make-raw-scatter", type=int, default=1, choices=[0, 1])
    ap.add_argument("--make-seedmean-scatter", type=int, default=1, choices=[0, 1])
    ap.add_argument("--make-single-beta-scatter", type=int, default=1, choices=[0, 1])
    ap.add_argument("--make-bucket-profile", type=int, default=1, choices=[0, 1])
    ap.add_argument("--draw-pareto-front", type=int, default=0, choices=[0, 1], help="Draw Pareto front on seed-mean plots.")
    ap.add_argument("--pareto-only", type=int, default=0, choices=[0, 1], help="Show only Pareto-nondominated points on seed-mean plots.")
    ap.add_argument("--draw-model-trends", type=int, default=1, choices=[0, 1], help="Draw per-model beta trend lines in seed-mean plot.")
    ap.add_argument("--hide-titles", type=int, default=0, choices=[0, 1], help="Hide plot titles.")

    ap.add_argument("--tag", type=str, default="", help="Optional filename suffix tag.")
    ap.add_argument("--dpi", type=int, default=420)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
