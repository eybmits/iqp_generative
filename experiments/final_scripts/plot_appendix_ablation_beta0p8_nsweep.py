#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final appendix plot script: beta=0.8 n-sweep ablation (data-driven).

The committed default NPZ is a historical 5-seed frozen snapshot; benchmark-standard
20-seed reruns belong in the analysis pipeline rather than the frozen final package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

PNG_DPI = 300
FIG_W = 6.95
FIG_H = 2.60
SINGLE_FIG_W = 243.12 / 72.0
SINGLE_FIG_H = 185.52 / 72.0

MODEL_STYLES: Dict[str, Dict[str, object]] = {
    "iqp_parity": {"label": "IQP (parity)", "color": "#C40000", "marker": "o"},
    "iqp_mse": {"label": "IQP (MSE)", "color": "#1F77B4", "marker": "s"},
    "classical_nnn_fields_parity": {
        "label": "Ising+fields (NN+NNN)",
        "color": "#4C78A8",
        "marker": "^",
    },
    "classical_dense_fields_xent": {
        "label": "Dense Ising+fields (xent)",
        "color": "#9C755F",
        "marker": "D",
    },
    "classical_transformer_mle": {
        "label": "AR Transformer (MLE)",
        "color": "#2CBCCB",
        "marker": "v",
    },
    "classical_maxent_parity": {
        "label": "MaxEnt parity (P,z)",
        "color": "#8F63B8",
        "marker": "X",
    },
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
        x_full = n_values.astype(np.float64)
        y_full = means[mi]
        yerr_full = stds[mi]
        valid = np.isfinite(y_full) & np.isfinite(yerr_full)
        if not np.any(valid):
            continue
        x = x_full[valid]
        y = y_full[valid]
        yerr = yerr_full[valid]

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
                vals = vals[np.isfinite(vals)]
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


def _legend_handles(model_keys: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for key in model_keys:
        st = MODEL_STYLES.get(key, {"label": key, "color": "#444444", "marker": "o"})
        handles.append(
            Line2D(
                [0],
                [0],
                color=str(st["color"]),
                marker=str(st["marker"]),
                linewidth=2.0,
                markersize=5.6,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=str(st["label"]),
            )
        )
    return handles


def _add_legend(
    ax: plt.Axes,
    model_keys: list[str],
    loc: str = "upper right",
    bbox_to_anchor: tuple[float, float] | None = None,
    framealpha: float = 1.0,
) -> None:
    leg = ax.legend(
        handles=_legend_handles(model_keys),
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        fontsize=6.9,
        facecolor="white",
        edgecolor="#bfbfbf",
        framealpha=framealpha,
        borderpad=0.20,
        labelspacing=0.18,
        handlelength=1.55,
        handletextpad=0.45,
        borderaxespad=0.20,
    )
    # Force legend above all plotted curves so it visibly overlays them.
    leg.set_zorder(100)


def _q80_from_qh_values(qh: np.ndarray, holdout_size: int, thr: float) -> np.ndarray:
    qh_arr = np.asarray(qh, dtype=np.float64)
    out = np.full_like(qh_arr, np.nan, dtype=np.float64)
    if holdout_size <= 0:
        return out
    h = float(holdout_size)
    q_state = np.clip(qh_arr / h, 0.0, 1.0 - 1e-15)
    num = float(np.log(max(1e-15, 1.0 - float(thr))))
    valid = np.isfinite(q_state) & (q_state > 0.0)
    if np.any(valid):
        den = np.log1p(-q_state[valid])  # negative
        out[valid] = num / den
    return out


def _reduce_seed_stats(seed_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(seed_arr.shape[:2], np.nan, dtype=np.float64)
    std = np.full(seed_arr.shape[:2], np.nan, dtype=np.float64)
    for i in range(seed_arr.shape[0]):
        for j in range(seed_arr.shape[1]):
            vals = seed_arr[i, j]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean[i, j] = float(np.mean(vals))
            std[i, j] = float(np.std(vals, ddof=0))
    return mean, std


def _merge_optional_baselines(
    *,
    include_classical_baselines: bool,
    baseline_csv: Path,
    n_values: np.ndarray,
    seed_values: np.ndarray,
    model_keys: list[str],
    q_holdout_seed: np.ndarray,
    r_q10000_seed: np.ndarray,
    q_holdout_mean: np.ndarray,
    q_holdout_std: np.ndarray,
    r_q10000_mean: np.ndarray,
    r_q10000_std: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not include_classical_baselines:
        return (
            model_keys,
            q_holdout_seed,
            r_q10000_seed,
            q_holdout_mean,
            q_holdout_std,
            r_q10000_mean,
            r_q10000_std,
        )

    if not baseline_csv.exists():
        raise FileNotFoundError(
            f"Missing baseline CSV: {baseline_csv}. "
            "Provide --baseline-csv or set --include-classical-baselines=0."
        )

    df = pd.read_csv(baseline_csv)
    required_cols = {"n", "model_key", "q_holdout", "R_Q10000"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"Baseline CSV is missing columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[df["n"].isin(n_values.tolist())].copy()
    if df.empty:
        raise ValueError(f"No baseline rows match n_values={n_values.tolist()} in {baseline_csv}")

    base_keys_all = [str(k) for k in df["model_key"].dropna().unique().tolist()]
    base_keys = [k for k in base_keys_all if k not in model_keys]
    if not base_keys:
        return (
            model_keys,
            q_holdout_seed,
            r_q10000_seed,
            q_holdout_mean,
            q_holdout_std,
            r_q10000_mean,
            r_q10000_std,
        )

    extra_q_seed = np.full((len(base_keys), n_values.size, seed_values.size), np.nan, dtype=np.float64)
    extra_r_seed = np.full((len(base_keys), n_values.size, seed_values.size), np.nan, dtype=np.float64)
    extra_q_mean = np.full((len(base_keys), n_values.size), np.nan, dtype=np.float64)
    extra_q_std = np.full((len(base_keys), n_values.size), np.nan, dtype=np.float64)
    extra_r_mean = np.full((len(base_keys), n_values.size), np.nan, dtype=np.float64)
    extra_r_std = np.full((len(base_keys), n_values.size), np.nan, dtype=np.float64)
    n_index = {int(n): i for i, n in enumerate(n_values.tolist())}

    for bi, key in enumerate(base_keys):
        dmk = df[df["model_key"] == key]
        for n in n_values.tolist():
            dn = dmk[dmk["n"] == n]
            if dn.empty:
                continue
            ni = n_index[int(n)]
            q_vals = dn["q_holdout"].astype(float).to_numpy()
            r_vals = dn["R_Q10000"].astype(float).to_numpy()
            extra_q_mean[bi, ni] = float(np.nanmean(q_vals))
            extra_q_std[bi, ni] = float(np.nanstd(q_vals, ddof=0))
            extra_r_mean[bi, ni] = float(np.nanmean(r_vals))
            extra_r_std[bi, ni] = float(np.nanstd(r_vals, ddof=0))

            # Keep as many replicate points as the existing jitter capacity.
            kq = min(seed_values.size, q_vals.size)
            kr = min(seed_values.size, r_vals.size)
            if kq > 0:
                extra_q_seed[bi, ni, :kq] = q_vals[:kq]
            if kr > 0:
                extra_r_seed[bi, ni, :kr] = r_vals[:kr]

    return (
        model_keys + base_keys,
        np.concatenate([q_holdout_seed, extra_q_seed], axis=0),
        np.concatenate([r_q10000_seed, extra_r_seed], axis=0),
        np.concatenate([q_holdout_mean, extra_q_mean], axis=0),
        np.concatenate([q_holdout_std, extra_q_std], axis=0),
        np.concatenate([r_q10000_mean, extra_r_mean], axis=0),
        np.concatenate([r_q10000_std, extra_r_std], axis=0),
    )


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
    ap.add_argument(
        "--include-classical-baselines",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, merge additional classical models from --baseline-csv into both Fig7 panels.",
    )
    ap.add_argument(
        "--baseline-csv",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "final_plots"
            / "fig7_appendix_ablation_beta0p8_nsweep"
            / "fig7_baselines_seed_table.csv"
        ),
        help=(
            "CSV with columns: n, model_key, q_holdout, R_Q10000 and optional seed. "
            "Used only when --include-classical-baselines=1."
        ),
    )
    ap.add_argument("--show-seed-points", type=int, default=0, choices=[0, 1])
    ap.add_argument(
        "--right-metric",
        type=str,
        default="rq10000",
        choices=["rq10000", "q80_from_qh"],
        help=(
            "Right panel metric: rq10000 (default) or q80_from_qh "
            "(Q80 estimated from q(H) via equal-mass holdout approximation)."
        ),
    )
    ap.add_argument("--q80-thr", type=float, default=0.8, help="Recovery threshold used for Q80 mode.")
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
        holdout_size = int(z["holdout_k"]) if "holdout_k" in z else 20

        eval_mode_by_n = [str(x) for x in z["eval_mode_by_n"].tolist()]
        if "shots_budget" in z:
            shots_budget = int(z["shots_budget"])
        elif "shots_n16" in z:
            shots_budget = int(z["shots_n16"])
        else:
            shots_budget = 0

    (
        model_keys,
        q_holdout_seed,
        r_q10000_seed,
        q_holdout_mean,
        q_holdout_std,
        r_q10000_mean,
        r_q10000_std,
    ) = _merge_optional_baselines(
        include_classical_baselines=bool(int(args.include_classical_baselines)),
        baseline_csv=Path(args.baseline_csv),
        n_values=n_values,
        seed_values=seed_values,
        model_keys=model_keys,
        q_holdout_seed=q_holdout_seed,
        r_q10000_seed=r_q10000_seed,
        q_holdout_mean=q_holdout_mean,
        q_holdout_std=q_holdout_std,
        r_q10000_mean=r_q10000_mean,
        r_q10000_std=r_q10000_std,
    )

    right_seed = r_q10000_seed
    right_mean = r_q10000_mean
    right_std = r_q10000_std
    right_ylabel = r"Recovery $R(10000)$"
    right_is_log = False
    right_legend_loc = "upper left"
    if str(args.right_metric).lower() == "q80_from_qh":
        right_seed = _q80_from_qh_values(q_holdout_seed, holdout_size=holdout_size, thr=float(args.q80_thr))
        right_mean, right_std = _reduce_seed_stats(right_seed)
        right_ylabel = r"$Q_{80}$ (lower better)"
        right_is_log = True
        right_legend_loc = "upper left"

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
        means=right_mean,
        stds=right_std,
        ylabel=right_ylabel,
        show_seed_points=bool(int(args.show_seed_points)),
        seed_values=seed_values,
        seed_data=right_seed,
    )
    if right_is_log:
        finite_vals = right_seed[np.isfinite(right_seed) & (right_seed > 0.0)]
        if finite_vals.size > 0:
            y_lo = float(np.min(finite_vals) * 0.8)
            y_hi = float(np.max(finite_vals) * 1.2)
            ax_r.set_yscale("log")
            ax_r.set_ylim(y_lo, y_hi)
    else:
        ax_r.set_ylim(0.0, 1.02)

    _add_legend(ax_l, model_keys)
    _add_legend(ax_r, model_keys, loc=right_legend_loc)

    n_shot = [int(n_values[i]) for i, mode in enumerate(eval_mode_by_n) if mode.startswith("shot")]
    shot_note = ""
    if n_shot:
        shot_note = f"exact for n<=14; shot-based ({shots_budget:,} shots) for n={','.join(map(str, n_shot))}"
    else:
        shot_note = "exact evaluation for all n"
    if right_is_log:
        shot_note += "; Q80 estimated from q(H)"

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

    # Export both panels as separate compact figures.
    fig_q, ax_q = plt.subplots(figsize=(SINGLE_FIG_W, SINGLE_FIG_H), constrained_layout=True)
    _draw_panel(
        ax=ax_q,
        n_values=n_values,
        model_keys=model_keys,
        means=q_holdout_mean,
        stds=q_holdout_std,
        ylabel=r"Holdout mass $q(H)$",
        show_seed_points=bool(int(args.show_seed_points)),
        seed_values=seed_values,
        seed_data=q_holdout_seed,
    )
    _add_legend(ax_q, model_keys, loc="upper right")
    out_q_pdf = outdir / "fig7_appendix_ablation_beta0p8_nsweep_qholdout_vs_n.pdf"
    out_q_png = outdir / "fig7_appendix_ablation_beta0p8_nsweep_qholdout_vs_n.png"
    fig_q.savefig(out_q_pdf)
    fig_q.savefig(out_q_png, dpi=int(args.dpi))
    plt.close(fig_q)

    fig_r, ax_r2 = plt.subplots(figsize=(SINGLE_FIG_W, SINGLE_FIG_H), constrained_layout=True)
    _draw_panel(
        ax=ax_r2,
        n_values=n_values,
        model_keys=model_keys,
        means=right_mean,
        stds=right_std,
        ylabel=right_ylabel,
        show_seed_points=bool(int(args.show_seed_points)),
        seed_values=seed_values,
        seed_data=right_seed,
    )
    if right_is_log:
        finite_vals = right_seed[np.isfinite(right_seed) & (right_seed > 0.0)]
        if finite_vals.size > 0:
            y_lo = float(np.min(finite_vals) * 0.8)
            y_hi = float(np.max(finite_vals) * 1.2)
            ax_r2.set_yscale("log")
            ax_r2.set_ylim(y_lo, y_hi)
    else:
        ax_r2.set_ylim(0.0, 1.02)
    _add_legend(
        ax_r2,
        model_keys,
        loc="upper left",
        bbox_to_anchor=None,
        framealpha=0.95,
    )
    out_r_pdf = outdir / "fig7_appendix_ablation_beta0p8_nsweep_rq10000_vs_n.pdf"
    out_r_png = outdir / "fig7_appendix_ablation_beta0p8_nsweep_rq10000_vs_n.png"
    fig_r.savefig(out_r_pdf)
    fig_r.savefig(out_r_png, dpi=int(args.dpi))
    plt.close(fig_r)

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_q_pdf}")
    print(f"[saved] {out_q_png}")
    print(f"[saved] {out_r_pdf}")
    print(f"[saved] {out_r_png}")


if __name__ == "__main__":
    run()
