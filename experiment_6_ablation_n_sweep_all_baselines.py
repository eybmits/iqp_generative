#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 6: fixed-beta n-sweep with 10 matched seeds and medium Transformer.

This driver follows the same active reporting protocol as Experiments 2 and 3:

- 10 matched data seeds
- training budget 600 for all trained model families
- Transformer baseline fixed to the medium configuration selected in Experiment 7

The primary output is a single PDF that summarizes exact forward KL
``D_KL(p* || q)`` versus system size ``n`` over a fixed beta slice.
The plotted line uses the seedwise median and the band shows the interquartile
range. Mean, standard deviation, and 95% CI are also stored for later rerender.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from training_protocol import STANDARD_SEED_IDS_CSV, write_training_protocol

from experiment_2_beta_kl_summary import (
    BAND_ALPHA,
    CLASSICAL_DENSE_INIT_OFFSET,
    CLASSICAL_NNN_INIT_OFFSET,
    HAS_PENNYLANE,
    HAS_TORCH,
    IQP_INIT_OFFSET,
    LEGEND_FONTSIZE,
    MAXENT_INIT_OFFSET,
    MAX_LABELED_X_TICKS,
    MEDIUM_TRANSFORMER,
    MODEL_ORDER,
    MODEL_STYLE,
    PARITY_BAND_OFFSET,
    TRAIN_SAMPLE_OFFSET,
    TRANSFORMER_INIT_OFFSET,
    _kl_pstar_to_q,
    _legend_handles,
    _parse_int_list,
    _reduce_seed_stats,
    _train_classical_boltzmann,
    _train_maxent_parity,
    _train_transformer_autoregressive,
    _try_rel,
    _write_csv,
    _write_json,
    apply_final_style,
    build_parity_matrix,
    build_target_distribution_paper,
    empirical_dist,
    make_bits_table,
    sample_alphas,
    sample_indices,
    train_iqp_qcbm,
)


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_6_ablation_n_sweep_all_baselines.py"
OUTPUT_STEM = "experiment_6_ablation_n_sweep_all_baselines"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM
DEFAULT_N_VALUES = ",".join(str(n) for n in range(8, 21))

FIG_W = 270.0 / 72.0
FIG_H = 185.52 / 72.0


def _major_n_ticks(n_values: Sequence[int]) -> List[int]:
    vals = sorted(int(x) for x in n_values)
    if len(vals) <= MAX_LABELED_X_TICKS:
        return vals
    step = max(2, int(np.ceil(len(vals) / float(MAX_LABELED_X_TICKS))))
    ticks = vals[::step]
    if vals[-1] not in ticks:
        ticks.append(vals[-1])
    return ticks


def _load_series_csv(series_csv: Path) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    with series_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "n",
            "model_key",
            "model_label",
            "n_seeds",
            "min",
            "q1",
            "median",
            "mean",
            "std",
            "ci95",
            "q3",
            "max",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Series CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if model_key not in MODEL_ORDER:
                continue
            rows_out.append(
                {
                    "n": int(float(row["n"])),
                    "model_key": model_key,
                    "model_label": str(row["model_label"]),
                    "n_seeds": int(float(row["n_seeds"])),
                    "min": float(row["min"]),
                    "q1": float(row["q1"]),
                    "median": float(row["median"]),
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "ci95": float(row["ci95"]),
                    "q3": float(row["q3"]),
                    "max": float(row["max"]),
                }
            )
    if not rows_out:
        raise ValueError(f"No usable rows found in {series_csv}")
    return rows_out


def _load_metrics_csv(metrics_csv: Path) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "beta",
            "n",
            "seed",
            "model_key",
            "model_label",
            "KL_pstar_to_q",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Metrics CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if model_key not in MODEL_ORDER:
                continue
            rows_out.append(
                {
                    "beta": float(row["beta"]),
                    "n": int(float(row["n"])),
                    "seed": int(float(row["seed"])),
                    "model_key": model_key,
                    "model_label": str(row["model_label"]),
                    "train_m": int(float(row["train_m"])) if str(row.get("train_m", "")).strip() else 0,
                    "sigma": float(row["sigma"]) if str(row.get("sigma", "")).strip() else float("nan"),
                    "K": int(float(row["K"])) if str(row.get("K", "")).strip() else 0,
                    "iqp_steps": int(float(row["iqp_steps"])) if str(row.get("iqp_steps", "")).strip() else 0,
                    "classical_steps": int(float(row["classical_steps"])) if str(row.get("classical_steps", "")).strip() else 0,
                    "transformer_variant": str(row.get("transformer_variant", "")),
                    "transformer_d_model": int(float(row["transformer_d_model"])) if str(row.get("transformer_d_model", "")).strip() else 0,
                    "transformer_heads": int(float(row["transformer_heads"])) if str(row.get("transformer_heads", "")).strip() else 0,
                    "transformer_layers": int(float(row["transformer_layers"])) if str(row.get("transformer_layers", "")).strip() else 0,
                    "transformer_dim_ff": int(float(row["transformer_dim_ff"])) if str(row.get("transformer_dim_ff", "")).strip() else 0,
                    "transformer_epochs": int(float(row["transformer_epochs"])) if str(row.get("transformer_epochs", "")).strip() else 0,
                    "maxent_steps": int(float(row["maxent_steps"])) if str(row.get("maxent_steps", "")).strip() else 0,
                    "KL_pstar_to_q": float(row["KL_pstar_to_q"]),
                }
            )
    return rows_out


def _series_rows_from_metrics(metric_rows: Sequence[Dict[str, object]], *, beta: float) -> List[Dict[str, object]]:
    grouped_vals: Dict[tuple[int, str], List[float]] = defaultdict(list)
    for row in metric_rows:
        if abs(float(row["beta"]) - float(beta)) > 1e-12:
            continue
        grouped_vals[(int(row["n"]), str(row["model_key"]))].append(float(row["KL_pstar_to_q"]))

    summary_rows: List[Dict[str, object]] = []
    for n in sorted({int(row["n"]) for row in metric_rows if abs(float(row["beta"]) - float(beta)) <= 1e-12}):
        for model_key in MODEL_ORDER:
            vals = np.asarray(grouped_vals.get((int(n), str(model_key)), []), dtype=np.float64)
            if vals.size == 0:
                continue
            stats = _reduce_seed_stats(vals)
            summary_rows.append(
                {
                    "n": int(n),
                    "beta": float(beta),
                    "model_key": str(model_key),
                    "model_label": str(MODEL_STYLE[model_key]["label"]),
                    **stats,
                }
            )
    return summary_rows


def _group_series(series_rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, np.ndarray]]:
    n_values = sorted({int(row["n"]) for row in series_rows})
    grouped: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    for row in series_rows:
        grouped[str(row["model_key"])][int(row["n"])] = {
            "median": float(row["median"]),
            "q1": float(row["q1"]),
            "q3": float(row["q3"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "ci95": float(row["ci95"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
        }

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for model_key in MODEL_ORDER:
        median = np.full(len(n_values), np.nan, dtype=np.float64)
        q1 = np.full(len(n_values), np.nan, dtype=np.float64)
        q3 = np.full(len(n_values), np.nan, dtype=np.float64)
        mean = np.full(len(n_values), np.nan, dtype=np.float64)
        std = np.full(len(n_values), np.nan, dtype=np.float64)
        ci95 = np.full(len(n_values), np.nan, dtype=np.float64)
        vmin = np.full(len(n_values), np.nan, dtype=np.float64)
        vmax = np.full(len(n_values), np.nan, dtype=np.float64)
        for idx, n in enumerate(n_values):
            if n not in grouped.get(model_key, {}):
                continue
            rec = grouped[model_key][n]
            median[idx] = float(rec["median"])
            q1[idx] = float(rec["q1"])
            q3[idx] = float(rec["q3"])
            mean[idx] = float(rec["mean"])
            std[idx] = float(rec["std"])
            ci95[idx] = float(rec["ci95"])
            vmin[idx] = float(rec["min"])
            vmax[idx] = float(rec["max"])
        out[model_key] = {
            "n_values": np.asarray(n_values, dtype=np.int64),
            "median": median,
            "q1": q1,
            "q3": q3,
            "mean": mean,
            "std": std,
            "ci95": ci95,
            "min": vmin,
            "max": vmax,
        }
    return out


def _render_pointcloud(metrics_rows: Sequence[Dict[str, object]], out_pdf: Path) -> None:
    apply_final_style()
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_n = sorted({int(row["n"]) for row in metrics_rows})
    for row in metrics_rows:
        grouped[str(row["model_key"])][int(row["n"])].append(float(row["KL_pstar_to_q"]))

    major_xticks = _major_n_ticks(all_n)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    rng = np.random.default_rng(0)
    ymin = float("inf")
    ymax = float("-inf")

    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        color = str(style["color"])
        mean_x: List[float] = []
        mean_y: List[float] = []
        std_y: List[float] = []
        for n in all_n:
            vals = np.asarray(grouped.get(model_key, {}).get(int(n), []), dtype=np.float64)
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.10, 0.10, size=vals.size)
            ax.scatter(
                np.full(vals.size, float(n)) + jitter,
                vals,
                s=16,
                color=color,
                alpha=0.18,
                edgecolors="none",
                zorder=2,
            )
            mean_x.append(float(n))
            mean_y.append(float(np.mean(vals)))
            std_y.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            ymin = min(ymin, float(np.min(vals)))
            ymax = max(ymax, float(np.max(vals)))

        if mean_x:
            x_arr = np.asarray(mean_x, dtype=np.float64)
            y_arr = np.asarray(mean_y, dtype=np.float64)
            sd_arr = np.asarray(std_y, dtype=np.float64)
            ax.errorbar(
                x_arr,
                y_arr,
                yerr=sd_arr,
                fmt="none",
                ecolor=color,
                elinewidth=1.15,
                capsize=2.6,
                capthick=1.15,
                alpha=0.55,
                zorder=3,
            )
            ax.plot(
                x_arr,
                y_arr,
                color=color,
                ls=style["ls"],
                lw=float(style["lw"]) * 0.85,
                alpha=0.9,
                zorder=4,
            )
            ax.scatter(
                x_arr,
                y_arr,
                s=28,
                color=color,
                alpha=0.95,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )

    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p^* \parallel q)$")
    if len(all_n) == 1:
        x0 = float(all_n[0])
        ax.set_xlim(x0 - 0.5, x0 + 0.5)
    else:
        ax.set_xlim(float(min(all_n)), float(max(all_n)))
    ax.set_xticks(major_xticks)
    ax.set_xticklabels([str(int(tick)) for tick in major_xticks])
    if np.isfinite(ymin) and np.isfinite(ymax):
        pad = 0.08 * max(ymax - ymin, 1e-6)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.grid(True, ls="--", lw=0.5, alpha=0.25)

    legend = ax.legend(
        handles=_legend_handles(),
        loc="upper right",
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.65,
        handletextpad=0.5,
        fontsize=LEGEND_FONTSIZE,
    )
    legend.set_zorder(100)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def _render_plot(series_rows: Sequence[Dict[str, object]], out_pdf: Path) -> None:
    apply_final_style()
    grouped = _group_series(series_rows)
    sample_ns = grouped[MODEL_ORDER[0]]["n_values"]
    major_xticks = _major_n_ticks(sample_ns.tolist())

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    ymin = float("inf")
    ymax = float("-inf")

    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        grp = grouped[model_key]
        x = grp["n_values"].astype(np.float64)
        y = grp["median"]
        q1 = grp["q1"]
        q3 = grp["q3"]
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(q1) & np.isfinite(q3)
        if not np.any(mask):
            continue
        ax.fill_between(x[mask], q1[mask], q3[mask], color=str(style["color"]), alpha=BAND_ALPHA, lw=0.0)
        ax.plot(x[mask], y[mask], color=str(style["color"]), ls=style["ls"], lw=float(style["lw"]))
        ymin = min(ymin, float(np.nanmin(q1[mask])))
        ymax = max(ymax, float(np.nanmax(q3[mask])))

    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p^* \parallel q)$")
    if sample_ns.size == 1:
        x0 = float(sample_ns[0])
        ax.set_xlim(x0 - 0.5, x0 + 0.5)
    else:
        ax.set_xlim(float(sample_ns.min()), float(sample_ns.max()))
    ax.set_xticks(major_xticks)
    ax.set_xticklabels([str(int(tick)) for tick in major_xticks])
    if np.isfinite(ymin) and np.isfinite(ymax):
        pad = 0.10 * max(ymax - ymin, 1e-6)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.grid(True, ls="--", lw=0.5, alpha=0.25)

    legend = ax.legend(
        handles=_legend_handles(),
        loc="upper right",
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.65,
        handletextpad=0.5,
        fontsize=LEGEND_FONTSIZE,
    )
    legend.set_zorder(100)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def _write_readme(
    path: Path,
    *,
    beta: float,
    n_values: np.ndarray,
    seed_values: np.ndarray,
    outdir: Path,
    metrics_csv: Path,
    series_csv: Path,
    data_npz: Path,
    out_pdf: Path,
    out_pointcloud_pdf: Path,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Experiment 6: Fixed-Beta n-Sweep",
        "",
        "This directory contains the final fixed-beta n-sweep KL summary over the full baseline set.",
        "",
        "Protocol:",
        "",
        f"- fixed beta: `{float(beta):.1f}`",
        f"- n values: `{','.join(str(int(x)) for x in n_values.tolist())}`",
        f"- matched seeds: `{','.join(str(int(x)) for x in seed_values.tolist())}`",
        "- seed count: `10` under the active standard protocol",
        "- target distribution: even-parity score-tilted family",
        "- train sample count: `m=200`",
        f"- parity band: `sigma={float(args.sigma):g}`, `K={int(args.K)}`",
        f"- IQP parity budget: `steps={int(args.iqp_steps)}`, `lr={float(args.iqp_lr):g}`",
        f"- Ising+fields budgets: `steps={int(args.classical_steps)}`, `lr={float(args.classical_lr):g}`",
        f"- MaxEnt parity budget: `steps={int(args.maxent_steps)}`, `lr={float(args.maxent_lr):g}`",
        (
            "- Transformer baseline: "
            f"`variant={MEDIUM_TRANSFORMER['variant']}`, `d_model={MEDIUM_TRANSFORMER['d_model']}`, "
            f"`layers={MEDIUM_TRANSFORMER['num_layers']}`, `heads={MEDIUM_TRANSFORMER['nhead']}`, "
            f"`dim_ff={MEDIUM_TRANSFORMER['dim_ff']}`, `epochs={int(args.transformer_epochs)}`, "
            f"`lr={float(args.transformer_lr):g}`, `batch_size={int(args.transformer_batch_size)}`"
        ),
        "",
        "Plot semantics:",
        "",
        "- line: seedwise median KL over the matched-seed pool",
        "- band: interquartile range (Q1 to Q3)",
        "- saved artifacts additionally include mean, standard deviation, and 95% CI for each n/model pair",
        "- rerun the script with a new `--n-values` slice (for example `22`) and keep `--append-existing 1` to merge the new n-values into the existing metrics/series/npz artifacts before rerendering",
        "",
        "Saved artifacts:",
        "",
        f"- per-seed KL metrics: `{_try_rel(metrics_csv)}`",
        f"- aggregated n-series: `{_try_rel(series_csv)}`",
        f"- saved data cube: `{_try_rel(data_npz)}`",
        f"- final PDF: `{_try_rel(out_pdf)}`",
        f"- pointcloud PDF: `{_try_rel(out_pointcloud_pdf)}`",
        "- local protocol doc: `TRAINING_PROTOCOL.md`",
        "",
        f"- source driver: `{SCRIPT_REL}`",
        f"- outdir: `{_try_rel(outdir)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recompute_series(
    *,
    beta: float,
    n_values: np.ndarray,
    seed_values: np.ndarray,
    metrics_csv: Path,
    series_csv: Path,
    data_npz: Path,
    args: argparse.Namespace,
    append_existing: bool,
) -> List[Dict[str, object]]:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for Experiment 6 recomputation.")
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for Experiment 6 recomputation.")

    kl_cube = np.full((n_values.size, len(MODEL_ORDER), seed_values.size), np.nan, dtype=np.float64)
    metric_rows_new: List[Dict[str, object]] = []

    print(f"[experiment6] recomputing {n_values.size} n values x {seed_values.size} seeds", flush=True)
    for ni, n in enumerate(n_values.tolist()):
        print(f"[experiment6] n={int(n)} ({ni + 1}/{n_values.size}) beta={float(beta):.1f}", flush=True)
        bits_table = make_bits_table(int(n))
        p_star, _support, _scores = build_target_distribution_paper(int(n), float(beta))

        for si, seed in enumerate(seed_values.tolist()):
            print(f"[experiment6]   seed={int(seed)} ({si + 1}/{seed_values.size})", flush=True)
            idxs_train = sample_indices(p_star, int(args.train_m), seed=int(seed) + TRAIN_SAMPLE_OFFSET)
            emp = empirical_dist(idxs_train, p_star.size)

            alphas = sample_alphas(int(n), float(args.sigma), int(args.K), seed=int(seed) + PARITY_BAND_OFFSET)
            P = build_parity_matrix(alphas, bits_table)
            z_data = P @ emp

            q_by_key = {
                "iqp_parity_mse": train_iqp_qcbm(
                    n=int(n),
                    layers=int(args.layers),
                    steps=int(args.iqp_steps),
                    lr=float(args.iqp_lr),
                    P=P,
                    z_data=z_data,
                    seed_init=int(seed) + IQP_INIT_OFFSET + 7 * int(args.K),
                    eval_every=int(args.iqp_eval_every),
                ),
                "classical_nnn_fields_parity": _train_classical_boltzmann(
                    n=int(n),
                    steps=int(args.classical_steps),
                    lr=float(args.classical_lr),
                    seed_init=int(seed) + CLASSICAL_NNN_INIT_OFFSET,
                    P=P,
                    z_data=z_data,
                    loss_mode="parity_mse",
                    emp_dist=emp,
                    topology="nn_nnn",
                    include_fields=True,
                ),
                "classical_dense_fields_xent": _train_classical_boltzmann(
                    n=int(n),
                    steps=int(args.classical_steps),
                    lr=float(args.classical_lr),
                    seed_init=int(seed) + CLASSICAL_DENSE_INIT_OFFSET,
                    P=P,
                    z_data=z_data,
                    loss_mode="xent",
                    emp_dist=emp,
                    topology="dense",
                    include_fields=True,
                ),
                "classical_transformer_mle": _train_transformer_autoregressive(
                    bits_table=bits_table,
                    idxs_train=idxs_train,
                    n=int(n),
                    seed=int(seed) + TRANSFORMER_INIT_OFFSET,
                    epochs=int(args.transformer_epochs),
                    d_model=int(MEDIUM_TRANSFORMER["d_model"]),
                    nhead=int(MEDIUM_TRANSFORMER["nhead"]),
                    num_layers=int(MEDIUM_TRANSFORMER["num_layers"]),
                    dim_ff=int(MEDIUM_TRANSFORMER["dim_ff"]),
                    lr=float(args.transformer_lr),
                    batch_size=int(args.transformer_batch_size),
                ),
                "classical_maxent_parity": _train_maxent_parity(
                    P=P,
                    z_data=z_data,
                    seed=int(seed) + MAXENT_INIT_OFFSET,
                    steps=int(args.maxent_steps),
                    lr=float(args.maxent_lr),
                ),
            }

            for mi, model_key in enumerate(MODEL_ORDER):
                q_model = np.asarray(q_by_key[model_key], dtype=np.float64)
                kl = float(_kl_pstar_to_q(p_star, q_model))
                kl_cube[ni, mi, si] = kl
                metric_rows_new.append(
                    {
                        "beta": float(beta),
                        "n": int(n),
                        "seed": int(seed),
                        "model_key": str(model_key),
                        "model_label": str(MODEL_STYLE[model_key]["label"]),
                        "train_m": int(args.train_m),
                        "sigma": float(args.sigma),
                        "K": int(args.K),
                        "iqp_steps": int(args.iqp_steps),
                        "classical_steps": int(args.classical_steps),
                        "transformer_variant": str(MEDIUM_TRANSFORMER["variant"]),
                        "transformer_d_model": int(MEDIUM_TRANSFORMER["d_model"]),
                        "transformer_heads": int(MEDIUM_TRANSFORMER["nhead"]),
                        "transformer_layers": int(MEDIUM_TRANSFORMER["num_layers"]),
                        "transformer_dim_ff": int(MEDIUM_TRANSFORMER["dim_ff"]),
                        "transformer_epochs": int(args.transformer_epochs),
                        "maxent_steps": int(args.maxent_steps),
                        "KL_pstar_to_q": float(kl),
                    }
                )

    retained_rows: List[Dict[str, object]] = []
    if append_existing and metrics_csv.exists():
        for row in _load_metrics_csv(metrics_csv):
            same_beta = abs(float(row["beta"]) - float(beta)) <= 1e-12
            recomputed_n = int(row["n"]) in set(int(x) for x in n_values.tolist())
            if same_beta and recomputed_n:
                continue
            retained_rows.append(row)

    metric_rows_all = retained_rows + metric_rows_new
    metric_rows_all = sorted(
        metric_rows_all,
        key=lambda row: (float(row["beta"]), int(row["n"]), int(row["seed"]), MODEL_ORDER.index(str(row["model_key"]))),
    )
    summary_rows = _series_rows_from_metrics(metric_rows_all, beta=float(beta))

    merged_n_values = np.asarray(
        sorted({int(row["n"]) for row in metric_rows_all if abs(float(row["beta"]) - float(beta)) <= 1e-12}),
        dtype=np.int64,
    )
    merged_seed_values = np.asarray(
        sorted({int(row["seed"]) for row in metric_rows_all if abs(float(row["beta"]) - float(beta)) <= 1e-12}),
        dtype=np.int64,
    )

    merged_kl_cube = np.full(
        (merged_n_values.size, len(MODEL_ORDER), merged_seed_values.size),
        np.nan,
        dtype=np.float64,
    )
    n_to_idx = {int(n): idx for idx, n in enumerate(merged_n_values.tolist())}
    seed_to_idx = {int(seed): idx for idx, seed in enumerate(merged_seed_values.tolist())}
    model_to_idx = {str(model_key): idx for idx, model_key in enumerate(MODEL_ORDER)}
    for row in metric_rows_all:
        if abs(float(row["beta"]) - float(beta)) > 1e-12:
            continue
        ni = n_to_idx[int(row["n"])]
        mi = model_to_idx[str(row["model_key"])]
        si = seed_to_idx[int(row["seed"])]
        merged_kl_cube[ni, mi, si] = float(row["KL_pstar_to_q"])

    _write_csv(metrics_csv, metric_rows_all)
    _write_csv(series_csv, summary_rows)
    np.savez(
        data_npz,
        beta=np.asarray([float(beta)], dtype=np.float64),
        n_values=merged_n_values,
        seed_values=merged_seed_values,
        model_order=np.asarray(MODEL_ORDER, dtype=object),
        kl_cube=merged_kl_cube,
    )
    return summary_rows


def run() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 6: fixed-beta n-sweep with 10 matched seeds and medium Transformer."
    )
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--recompute", type=int, default=0, help="Set to 1 to recompute the full n-sweep before rendering.")
    ap.add_argument("--series-csv", type=str, default="")
    ap.add_argument("--metrics-csv", type=str, default="")
    ap.add_argument("--data-npz", type=str, default="")
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--n-values", type=str, default=DEFAULT_N_VALUES)
    ap.add_argument("--seeds", type=str, default=STANDARD_SEED_IDS_CSV)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=20)
    ap.add_argument("--classical-steps", type=int, default=600)
    ap.add_argument("--classical-lr", type=float, default=0.05)
    ap.add_argument("--transformer-epochs", type=int, default=600)
    ap.add_argument("--transformer-lr", type=float, default=1e-3)
    ap.add_argument("--transformer-batch-size", type=int, default=256)
    ap.add_argument("--maxent-steps", type=int, default=600)
    ap.add_argument("--maxent-lr", type=float, default=0.05)
    ap.add_argument(
        "--append-existing",
        type=int,
        default=1,
        help="When recomputing, merge new n-values into the existing metrics/series files instead of overwriting unrelated n-values.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    series_csv = (
        Path(args.series_csv).expanduser()
        if str(args.series_csv).strip()
        else (outdir / f"{OUTPUT_STEM}_series.csv")
    )
    metrics_csv = (
        Path(args.metrics_csv).expanduser()
        if str(args.metrics_csv).strip()
        else (outdir / f"{OUTPUT_STEM}_metrics_per_seed.csv")
    )
    data_npz = (
        Path(args.data_npz).expanduser()
        if str(args.data_npz).strip()
        else (outdir / f"{OUTPUT_STEM}_data.npz")
    )
    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_pointcloud_pdf = outdir / f"{OUTPUT_STEM}_pointcloud.pdf"
    summary_json = outdir / f"{OUTPUT_STEM}_summary.json"
    readme_md = outdir / "README.md"

    beta = float(args.beta)
    n_values = _parse_int_list(str(args.n_values))
    seed_values = _parse_int_list(str(args.seeds))

    if int(args.recompute) == 1 or not series_csv.exists():
        series_rows = _recompute_series(
            beta=beta,
            n_values=n_values,
            seed_values=seed_values,
            metrics_csv=metrics_csv,
            series_csv=series_csv,
            data_npz=data_npz,
            args=args,
            append_existing=bool(int(args.append_existing)),
        )
    else:
        series_rows = _load_series_csv(series_csv)

    metrics_rows = _load_metrics_csv(metrics_csv)
    _render_plot(series_rows, out_pdf)
    _render_pointcloud(metrics_rows, out_pointcloud_pdf)

    final_n_values = np.asarray(sorted({int(row["n"]) for row in series_rows}), dtype=np.int64)

    grouped = _group_series(series_rows)
    summary_payload = {
        "script": SCRIPT_REL,
        "outdir": _try_rel(outdir),
        "beta": float(beta),
        "n_values": [int(x) for x in final_n_values.tolist()],
        "seed_values": [int(x) for x in seed_values.tolist()],
        "seed_count": int(seed_values.size),
        "plot_center": "median",
        "plot_band": "iqr",
        "secondary_statistics": "mean_std_ci95",
        "transformer_variant": dict(MEDIUM_TRANSFORMER),
        "models": {
            model_key: {
                "best_n_by_median": int(
                    grouped[model_key]["n_values"][int(np.nanargmin(grouped[model_key]["median"]))]
                ),
                "best_median_kl": float(np.nanmin(grouped[model_key]["median"])),
                "mean_at_best_n": float(
                    grouped[model_key]["mean"][int(np.nanargmin(grouped[model_key]["median"]))]
                ),
                "ci95_at_best_n": float(
                    grouped[model_key]["ci95"][int(np.nanargmin(grouped[model_key]["median"]))]
                ),
            }
            for model_key in MODEL_ORDER
        },
    }
    _write_json(summary_json, summary_payload)
    _write_json(
        outdir / "RUN_CONFIG.json",
        {
            "script": SCRIPT_REL,
            "outdir": _try_rel(outdir),
            "input_mode": "recompute" if int(args.recompute) == 1 or not series_csv.exists() else "series_csv",
            "beta": float(beta),
            "n_values": [int(x) for x in final_n_values.tolist()],
            "seed_values": [int(x) for x in seed_values.tolist()],
            "seed_count": int(seed_values.size),
            "train_m": int(args.train_m),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "layers": int(args.layers),
            "iqp_steps": int(args.iqp_steps),
            "iqp_lr": float(args.iqp_lr),
            "classical_steps": int(args.classical_steps),
            "classical_lr": float(args.classical_lr),
            "transformer_variant": dict(MEDIUM_TRANSFORMER),
            "transformer_epochs": int(args.transformer_epochs),
            "transformer_lr": float(args.transformer_lr),
            "transformer_batch_size": int(args.transformer_batch_size),
            "maxent_steps": int(args.maxent_steps),
            "maxent_lr": float(args.maxent_lr),
            "series_csv": _try_rel(series_csv),
            "metrics_csv": _try_rel(metrics_csv),
            "data_npz": _try_rel(data_npz),
            "summary_json": _try_rel(summary_json),
            "pdf": _try_rel(out_pdf),
            "pointcloud_pdf": _try_rel(out_pointcloud_pdf),
            "plot_center": "median",
            "plot_band": "iqr",
            "secondary_statistics": "mean_std_ci95",
            "append_existing": bool(int(args.append_existing)),
            "pdf_only": True,
        },
    )
    _write_json(
        outdir / "RERENDER_CONFIG.json",
        {
            "script": SCRIPT_REL,
            "series_csv": _try_rel(series_csv),
            "outdir": _try_rel(outdir),
            "pdf": out_pdf.name,
            "pointcloud_pdf": out_pointcloud_pdf.name,
            "plot_center": "median",
            "plot_band": "iqr",
            "secondary_statistics": "mean_std_ci95",
            "rerender_command": f"python {SCRIPT_REL} --outdir {outdir.as_posix()} --series-csv {series_csv.as_posix()}",
        },
    )
    _write_readme(
        readme_md,
        beta=beta,
        n_values=final_n_values,
        seed_values=seed_values,
        outdir=outdir,
        metrics_csv=metrics_csv,
        series_csv=series_csv,
        data_npz=data_npz,
        out_pdf=out_pdf,
        out_pointcloud_pdf=out_pointcloud_pdf,
        args=args,
    )
    write_training_protocol(
        outdir,
        experiment_name="Experiment 6 fixed-beta n-sweep",
        note="This run uses the shared 10-seed / 600-budget analysis standard with the medium Transformer selected in Experiment 7.",
        source_relpath=SCRIPT_REL,
        metrics_note="The plotted curve shows seedwise median KL with an interquartile band; mean and 95% CI are stored alongside the rerender artifacts.",
    )
    print(f"[experiment6] wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    run()
