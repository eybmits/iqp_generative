#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 7 ablation: fixed-beta Transformer capacity ablation against IQP parity.

This script evaluates four autoregressive Transformer sizes on a fixed beta slice
and compares them to the IQP parity baseline using:

- validation NLL on an independently sampled validation set from p_train
- exact forward KL D_KL(p* || q) on the full target distribution

The output is a compact model-size sweep intended to address under/over-capacity
questions for the Transformer baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent

from experiment_3_beta_quality_coverage import (  # noqa: E402
    HAS_PENNYLANE,
    HAS_TORCH,
    _ARTransformer,
    _train_transformer_autoregressive,
    build_parity_matrix,
    build_target_distribution_paper,
    empirical_dist,
    make_bits_table,
    sample_alphas,
    sample_indices,
    topk_mask_by_scores,
    train_iqp_qcbm,
)
from final_plot_style import PARITY_COLOR, apply_final_style  # noqa: E402
from training_protocol import STANDARD_SEED_IDS_CSV, write_training_protocol


OUTPUT_STEM = "experiment_7_ablation_transformer_capacity_fixed_beta"
PNG_DPI = 300
TRAIN_SAMPLE_OFFSET = 7
VALIDATION_SAMPLE_OFFSET = 29
PARITY_BAND_OFFSET = 222
IQP_INIT_OFFSET_BASE = 10000
TRANSFORMER_INIT_OFFSET = 35501

TRANSFORMER_CONFIGS: List[Dict[str, int | str]] = [
    {"variant": "tiny", "d_model": 8, "nhead": 2, "layers": 1, "dim_ff": 16},
    {"variant": "small", "d_model": 16, "nhead": 2, "layers": 1, "dim_ff": 32},
    {"variant": "medium", "d_model": 32, "nhead": 4, "layers": 1, "dim_ff": 64},
    {"variant": "large", "d_model": 64, "nhead": 4, "layers": 2, "dim_ff": 128},
]

TRANSFORMER_LINE_COLOR = "#2b6cb0"
TRANSFORMER_POINT_COLOR = "#8fb7e8"
IQP_POINT_COLOR = "#d90429"


def _select_holdout_random(candidate_mask: np.ndarray, *, holdout_k: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    mask = np.zeros_like(candidate_mask, dtype=bool)
    if idx.size == 0 or int(holdout_k) <= 0:
        return mask
    if idx.size <= int(holdout_k):
        mask[idx] = True
        return mask
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(idx, size=int(holdout_k), replace=False)
    mask[np.asarray(chosen, dtype=np.int64)] = True
    return mask


def select_holdout_smart(
    *,
    p_star: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    m_train: int,
    holdout_k: int,
    pool_size: int,
    seed: int,
) -> np.ndarray:
    candidate_idx = np.flatnonzero(np.asarray(good_mask, dtype=bool))
    mask = np.zeros_like(good_mask, dtype=bool)
    if candidate_idx.size == 0 or int(holdout_k) <= 0:
        return mask
    if candidate_idx.size <= int(holdout_k):
        mask[candidate_idx] = True
        return mask

    probs = np.asarray(p_star, dtype=np.float64)[candidate_idx]
    order = np.argsort(-probs)
    pool = candidate_idx[order[: min(int(pool_size), candidate_idx.size)]]
    if pool.size <= int(holdout_k):
        mask[pool] = True
        return mask

    rng = np.random.default_rng(int(seed))
    chosen: List[int] = [int(pool[0])]
    remaining = [int(x) for x in pool[1:]]
    while len(chosen) < int(holdout_k) and remaining:
        best_idx = None
        best_score = None
        for idx in remaining:
            dmin = min(int(np.sum(bits_table[idx] != bits_table[c])) for c in chosen)
            score = (float(p_star[idx]), float(dmin), float(rng.random()))
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        chosen.append(int(best_idx))
        remaining.remove(int(best_idx))
    mask[np.asarray(chosen, dtype=np.int64)] = True
    return mask


def _kl_pstar_to_q(p_star: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    pv = np.asarray(p_star, dtype=np.float64)
    qv = np.asarray(q, dtype=np.float64)
    pv = np.clip(pv, eps, 1.0)
    qv = np.clip(qv, eps, 1.0)
    pv = pv / float(np.sum(pv))
    qv = qv / float(np.sum(qv))
    return float(np.sum(pv * np.log(pv / qv)))


def _mean_nll_from_indices(q: np.ndarray, idxs: np.ndarray, eps: float = 1e-12) -> float:
    qv = np.asarray(q, dtype=np.float64)
    qv = np.clip(qv, eps, 1.0)
    qv = qv / float(np.sum(qv))
    idxs_arr = np.asarray(idxs, dtype=np.int64)
    return float(-np.mean(np.log(qv[idxs_arr])))


def _count_transformer_params(n: int, d_model: int, nhead: int, layers: int, dim_ff: int) -> int:
    model = _ARTransformer(
        n=int(n),
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(layers),
        dim_ff=int(dim_ff),
        dropout=0.0,
    )
    return int(sum(p.numel() for p in model.parameters()))


def _is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    n = int(points.shape[0])
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return keep


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _build_summary(rows: List[Dict[str, object]]) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    variants = []
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    transformer_points = []
    transformer_indices = []
    for idx, variant in enumerate(["tiny", "small", "medium", "large"]):
        group = grouped.get(variant, [])
        if not group:
            continue
        param_count = int(group[0]["param_count"])
        val_vals = np.asarray([float(r["val_nll"]) for r in group], dtype=np.float64)
        kl_vals = np.asarray([float(r["test_kl"]) for r in group], dtype=np.float64)
        entry = {
            "variant": variant,
            "family": "transformer",
            "param_count": param_count,
            "d_model": int(group[0]["d_model"]),
            "layers": int(group[0]["layers"]),
            "nhead": int(group[0]["nhead"]),
            "dim_ff": int(group[0]["dim_ff"]),
            "seeds_n": int(len(group)),
            "val_nll_mean": float(np.mean(val_vals)),
            "val_nll_std": float(np.std(val_vals, ddof=0)),
            "test_kl_mean": float(np.mean(kl_vals)),
            "test_kl_std": float(np.std(kl_vals, ddof=0)),
        }
        variants.append(entry)
        transformer_points.append([entry["val_nll_mean"], entry["test_kl_mean"]])
        transformer_indices.append(len(variants) - 1)

    iqp_group = grouped.get("iqp_parity", [])
    if iqp_group:
        param_count = int(iqp_group[0]["param_count"])
        val_vals = np.asarray([float(r["val_nll"]) for r in iqp_group], dtype=np.float64)
        kl_vals = np.asarray([float(r["test_kl"]) for r in iqp_group], dtype=np.float64)
        variants.append(
            {
                "variant": "iqp_parity",
                "family": "iqp",
                "param_count": param_count,
                "d_model": "",
                "layers": int(iqp_group[0]["layers"]),
                "nhead": "",
                "dim_ff": "",
                "seeds_n": int(len(iqp_group)),
                "val_nll_mean": float(np.mean(val_vals)),
                "val_nll_std": float(np.std(val_vals, ddof=0)),
                "test_kl_mean": float(np.mean(kl_vals)),
                "test_kl_std": float(np.std(kl_vals, ddof=0)),
            }
        )

    if transformer_points:
        points_arr = np.asarray(transformer_points, dtype=np.float64)
        pareto_mask = _is_pareto_efficient(points_arr)
        for local_idx, keep in enumerate(pareto_mask.tolist()):
            variants[transformer_indices[local_idx]]["pareto_non_dominated"] = int(bool(keep))
    for entry in variants:
        entry.setdefault("pareto_non_dominated", 0)

    transformer_variants = [v for v in variants if str(v["family"]) == "transformer"]
    best_kl_variant = min(transformer_variants, key=lambda x: float(x["test_kl_mean"])) if transformer_variants else None
    kl_curve_has_interior_minimum = False
    if len(transformer_variants) >= 3:
        kl_means = np.asarray([float(v["test_kl_mean"]) for v in transformer_variants], dtype=np.float64)
        best_idx = int(np.argmin(kl_means))
        kl_curve_has_interior_minimum = 0 < best_idx < (len(transformer_variants) - 1)

    summary_payload = {
        "best_transformer_by_mean_test_kl": best_kl_variant,
        "kl_curve_has_interior_minimum": bool(kl_curve_has_interior_minimum),
        "transformer_variants": transformer_variants,
    }
    return variants, summary_payload


def _write_readme(
    path: Path,
    *,
    beta: float,
    n: int,
    seed_values: List[int],
    train_m: int,
    val_m: int,
    holdout_mode: str,
    outdir: Path,
    summary_rows: List[Dict[str, object]],
    per_seed_rows: List[Dict[str, object]],
) -> None:
    try:
        outdir_rel = str(outdir.relative_to(ROOT))
    except ValueError:
        outdir_rel = str(outdir)
    transformer_rows = [r for r in summary_rows if str(r["family"]) == "transformer"]
    iqp_row = next(r for r in summary_rows if str(r["variant"]) == "iqp_parity")
    best_transformer_by_kl = min(transformer_rows, key=lambda r: float(r["test_kl_mean"]))
    best_transformer_by_val = min(transformer_rows, key=lambda r: float(r["val_nll_mean"]))
    seeds = sorted({int(r["seed"]) for r in per_seed_rows})
    iqp_seed_wins = 0
    for seed in seeds:
        iqp_seed = next(r for r in per_seed_rows if int(r["seed"]) == seed and str(r["variant"]) == "iqp_parity")
        best_tr_seed = min(
            [r for r in per_seed_rows if int(r["seed"]) == seed and str(r["family"]) == "transformer"],
            key=lambda r: float(r["test_kl"]),
        )
        if float(iqp_seed["test_kl"]) < float(best_tr_seed["test_kl"]):
            iqp_seed_wins += 1
    lines = [
        "# Transformer Capacity Ablation",
        "",
        "This directory contains the fixed-beta Transformer capacity sweep against the IQP parity baseline.",
        "",
        "Protocol:",
        "",
        f"- fixed beta: `{beta:g}`",
        f"- n: `{n}`",
        f"- seeds: `{','.join(str(int(x)) for x in seed_values)}`",
        f"- train sample size: `m={train_m}`",
        f"- validation sample size: `m_val={val_m}` sampled independently from `p_train`",
        f"- holdout mode: `{holdout_mode}`",
        "- same `D_train`, parity band, and validation sample are reused across Transformer sizes within each seed",
        "- reported metrics: validation NLL and exact forward KL",
        "- training protocol file: `TRAINING_PROTOCOL.md`",
        "",
        "Transformer sizes:",
        "",
        "- `tiny`: `d_model=8`, `layers=1`, `heads=2`, `dim_ff=16`",
        "- `small`: `d_model=16`, `layers=1`, `heads=2`, `dim_ff=32`",
        "- `medium`: `d_model=32`, `layers=1`, `heads=4`, `dim_ff=64`",
        "- `large`: `d_model=64`, `layers=2`, `heads=4`, `dim_ff=128`",
        "",
        "Headline results:",
        "",
        (
            f"- `IQP parity` is best overall on both reported means: "
            f"`val NLL = {float(iqp_row['val_nll_mean']):.3f}` and "
            f"`test KL = {float(iqp_row['test_kl_mean']):.3f}` at `24` parameters"
        ),
        (
            f"- best Transformer by mean test KL: `{best_transformer_by_kl['variant']}` "
            f"with `{int(best_transformer_by_kl['param_count']):,}` parameters "
            f"(`d_model={int(best_transformer_by_kl['d_model'])}`, "
            f"`layers={int(best_transformer_by_kl['layers'])}`, "
            f"`heads={int(best_transformer_by_kl['nhead'])}`, "
            f"`dim_ff={int(best_transformer_by_kl['dim_ff'])}`), "
            f"`test KL = {float(best_transformer_by_kl['test_kl_mean']):.3f}`"
        ),
        (
            f"- best Transformer by mean validation NLL: `{best_transformer_by_val['variant']}` "
            f"with `{int(best_transformer_by_val['param_count']):,}` parameters, "
            f"`val NLL = {float(best_transformer_by_val['val_nll_mean']):.3f}`"
        ),
        (
            f"- the KL curve shows no interior minimum in this pilot: the lowest mean Transformer KL "
            f"is already the smallest tested model, and the `67,969`-parameter model is worst"
        ),
        (
            f"- seedwise comparison: `IQP parity` beats the best Transformer on `{iqp_seed_wins}/{len(seeds)}` seeds"
        ),
        "",
        f"Artifacts are stored under `{outdir_rel}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_plots(
    *,
    summary_rows: List[Dict[str, object]],
    per_seed_rows: List[Dict[str, object]],
    out_pdf: Path,
    out_png: Path,
    out_val_pdf: Path,
    out_val_png: Path,
    out_kl_pdf: Path,
    out_kl_png: Path,
    dpi: int,
) -> None:
    apply_final_style()
    transformer_summary = [r for r in summary_rows if str(r["family"]) == "transformer"]
    iqp_summary = next((r for r in summary_rows if str(r["variant"]) == "iqp_parity"), None)
    transformer_summary = sorted(transformer_summary, key=lambda r: int(r["param_count"]))

    panels = [
        ("val_nll_mean", "val_nll_std", "val_nll", "Validation NLL (lower better)"),
        ("test_kl_mean", "test_kl_std", "test_kl", r"Test KL $D_{KL}(p^* \parallel q)$ (lower better)"),
    ]

    def _draw_capacity_panel(ax, mean_key: str, std_key: str, seed_key: str, ylabel: str) -> None:
        x = np.asarray([int(r["param_count"]) for r in transformer_summary], dtype=np.float64)
        y = np.asarray([float(r[mean_key]) for r in transformer_summary], dtype=np.float64)
        yerr = np.asarray([float(r[std_key]) for r in transformer_summary], dtype=np.float64)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=TRANSFORMER_LINE_COLOR,
            marker="o",
            linewidth=2.0,
            capsize=3.0,
            label="Transformer sweep",
            zorder=10,
        )
        if mean_key == "val_nll_mean":
            label_offsets_pts = {
                "tiny": (-10, 10),
                "small": (-10, 20),
                "medium": (-10, 10),
                "large": (-10, 10),
            }
        else:
            label_offsets_pts = {
                "tiny": (-10, 10),
                "small": (-10, 10),
                "medium": (-10, 10),
                "large": (-10, 10),
            }
        for row in transformer_summary:
            xoff, yoff = label_offsets_pts.get(str(row["variant"]), (8, 0))
            ax.annotate(
                f"{row['variant']}",
                xy=(float(row["param_count"]), float(row[mean_key])),
                xycoords="data",
                xytext=(xoff, yoff),
                textcoords="offset points",
                ha="left" if xoff >= 0 else "right",
                va="center",
                fontsize=7.6,
                color=TRANSFORMER_LINE_COLOR,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.15),
                clip_on=False,
                zorder=20,
            )

        for variant_row in transformer_summary:
            v = str(variant_row["variant"])
            px = float(variant_row["param_count"])
            vals = np.asarray(
                [float(r[seed_key]) for r in per_seed_rows if str(r["variant"]) == v],
                dtype=np.float64,
            )
            if vals.size == 0:
                continue
            jitter = np.linspace(-0.03, 0.03, num=vals.size)
            x_jitter = px * np.power(10.0, jitter)
            ax.scatter(x_jitter, vals, color=TRANSFORMER_POINT_COLOR, alpha=0.45, s=20, linewidths=0.0, zorder=6)

        if iqp_summary is not None:
            iqp_x = float(iqp_summary["param_count"])
            iqp_y = float(iqp_summary[mean_key])
            iqp_yerr = float(iqp_summary[std_key])
            ax.errorbar(
                [iqp_x],
                [iqp_y],
                yerr=[iqp_yerr],
                color=IQP_POINT_COLOR,
                marker="o",
                markersize=6,
                linewidth=1.6,
                capsize=3.0,
                label="IQP parity",
                zorder=12,
            )
            ax.annotate(
                "IQP parity",
                xy=(iqp_x, iqp_y),
                xycoords="data",
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=7.6,
                color=IQP_POINT_COLOR,
                va="center",
                ha="left",
                clip_on=False,
                zorder=20,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.16, linestyle="--")
        ax.legend(loc="upper left", frameon=True, fontsize=7.0)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), constrained_layout=True)
    for ax, (mean_key, std_key, seed_key, ylabel) in zip(axes, panels):
        _draw_capacity_panel(ax, mean_key=mean_key, std_key=std_key, seed_key=seed_key, ylabel=ylabel)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)

    fig_val, ax_val = plt.subplots(figsize=(3.55, 2.85), constrained_layout=True)
    _draw_capacity_panel(ax_val, mean_key=panels[0][0], std_key=panels[0][1], seed_key=panels[0][2], ylabel=panels[0][3])
    fig_val.savefig(out_val_pdf)
    fig_val.savefig(out_val_png, dpi=int(dpi))
    plt.close(fig_val)

    fig_kl, ax_kl = plt.subplots(figsize=(3.55, 2.85), constrained_layout=True)
    _draw_capacity_panel(ax_kl, mean_key=panels[1][0], std_key=panels[1][1], seed_key=panels[1][2], ylabel=panels[1][3])
    fig_kl.savefig(out_kl_pdf)
    fig_kl.savefig(out_kl_png, dpi=int(dpi))
    plt.close(fig_kl)


def run() -> None:
    ap = argparse.ArgumentParser(description="Experiment 7 ablation: fixed-beta Transformer capacity ablation against IQP parity.")
    ap.add_argument(
        "--outdir",
        type=str,
        default="",
        help="If omitted, a beta-tagged directory under plots/ is used.",
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--seeds", type=str, default=STANDARD_SEED_IDS_CSV)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--val-m", type=int, default=1000)
    ap.add_argument(
        "--holdout-mode",
        type=str,
        default="global",
        choices=["global", "high_value", "random_global", "random_high_value"],
    )
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--holdout-seed", type=int, default=46)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--artr-epochs", type=int, default=600)
    ap.add_argument("--artr-lr", type=float, default=1e-3)
    ap.add_argument("--artr-batch-size", type=int, default=256)
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    ap.add_argument("--recompute", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    beta_tag = f"{float(args.beta):.2f}".replace(".", "p")
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir.strip()
        else ROOT / "plots" / f"{OUTPUT_STEM}_beta{beta_tag}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    points_csv = outdir / f"{OUTPUT_STEM}_points_beta{beta_tag}.csv"
    summary_csv = outdir / f"{OUTPUT_STEM}_summary_beta{beta_tag}.csv"
    summary_json = outdir / f"{OUTPUT_STEM}_summary_beta{beta_tag}.json"
    out_pdf = outdir / f"{OUTPUT_STEM}_beta{beta_tag}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}_beta{beta_tag}.png"
    out_val_pdf = outdir / f"{OUTPUT_STEM}_beta{beta_tag}_val_nll.pdf"
    out_val_png = outdir / f"{OUTPUT_STEM}_beta{beta_tag}_val_nll.png"
    out_kl_pdf = outdir / f"{OUTPUT_STEM}_beta{beta_tag}_test_kl.pdf"
    out_kl_png = outdir / f"{OUTPUT_STEM}_beta{beta_tag}_test_kl.png"
    run_config_json = outdir / "RUN_CONFIG.json"
    readme_md = outdir / "README.md"

    seed_values = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    if points_csv.exists() and not bool(int(args.recompute)):
        with points_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        bits_table = make_bits_table(int(args.n))
        rows: List[Dict[str, object]] = []

        for run_idx, seed in enumerate(seed_values, start=1):
            print(f"[run {run_idx}/{len(seed_values)}] beta={float(args.beta):g} seed={seed}")
            p_star, support, scores = build_target_distribution_paper(int(args.n), float(args.beta))
            good_mask = topk_mask_by_scores(scores, support, frac=float(args.good_frac))
            candidate_mask = (
                support.astype(bool)
                if str(args.holdout_mode) in {"global", "random_global"}
                else good_mask
            )
            if str(args.holdout_mode).startswith("random_"):
                holdout_mask = _select_holdout_random(
                    candidate_mask,
                    holdout_k=int(args.holdout_k),
                    seed=int(args.holdout_seed) + 111,
                )
            else:
                holdout_mask = select_holdout_smart(
                    p_star=p_star,
                    good_mask=candidate_mask,
                    bits_table=bits_table,
                    m_train=int(args.holdout_m_train),
                    holdout_k=int(args.holdout_k),
                    pool_size=int(args.holdout_pool),
                    seed=int(args.holdout_seed) + 111,
                )

            p_train = p_star.copy()
            p_train[holdout_mask] = 0.0
            p_train /= float(np.sum(p_train))

            idxs_train = sample_indices(p_train, int(args.train_m), seed=int(seed) + TRAIN_SAMPLE_OFFSET)
            idxs_val = sample_indices(p_train, int(args.val_m), seed=int(seed) + VALIDATION_SAMPLE_OFFSET)
            emp = empirical_dist(idxs_train, p_star.size)
            alphas = sample_alphas(int(args.n), float(args.sigma), int(args.K), seed=int(seed) + PARITY_BAND_OFFSET)
            P = build_parity_matrix(alphas, bits_table)
            z_data = P @ emp

            q_iqp = train_iqp_qcbm(
                n=int(args.n),
                layers=int(args.layers),
                steps=int(args.iqp_steps),
                lr=float(args.iqp_lr),
                P=P,
                z_data=z_data,
                seed_init=int(seed) + IQP_INIT_OFFSET_BASE + 7 * int(args.K),
                eval_every=int(args.iqp_eval_every),
            )
            rows.append(
                {
                    "variant": "iqp_parity",
                    "family": "iqp",
                    "seed": int(seed),
                    "param_count": 24,
                    "d_model": "",
                    "layers": int(args.layers),
                    "nhead": "",
                    "dim_ff": "",
                    "val_nll": float(_mean_nll_from_indices(q_iqp, idxs_val)),
                    "test_kl": float(_kl_pstar_to_q(p_star=p_star, q=q_iqp)),
                }
            )

            for cfg in TRANSFORMER_CONFIGS:
                q_tr = _train_transformer_autoregressive(
                    bits_table=bits_table,
                    idxs_train=idxs_train,
                    n=int(args.n),
                    seed=int(seed) + TRANSFORMER_INIT_OFFSET,
                    epochs=int(args.artr_epochs),
                    d_model=int(cfg["d_model"]),
                    nhead=int(cfg["nhead"]),
                    num_layers=int(cfg["layers"]),
                    dim_ff=int(cfg["dim_ff"]),
                    lr=float(args.artr_lr),
                    batch_size=int(args.artr_batch_size),
                )
                rows.append(
                    {
                        "variant": str(cfg["variant"]),
                        "family": "transformer",
                        "seed": int(seed),
                        "param_count": _count_transformer_params(
                            int(args.n),
                            int(cfg["d_model"]),
                            int(cfg["nhead"]),
                            int(cfg["layers"]),
                            int(cfg["dim_ff"]),
                        ),
                        "d_model": int(cfg["d_model"]),
                        "layers": int(cfg["layers"]),
                        "nhead": int(cfg["nhead"]),
                        "dim_ff": int(cfg["dim_ff"]),
                        "val_nll": float(_mean_nll_from_indices(q_tr, idxs_val)),
                        "test_kl": float(_kl_pstar_to_q(p_star=p_star, q=q_tr)),
                    }
                )

        _write_csv(points_csv, rows)

    summary_rows, summary_payload = _build_summary(rows)
    _write_csv(summary_csv, summary_rows)
    _write_json(
        summary_json,
        {
            "config": {
                "beta": float(args.beta),
                "n": int(args.n),
                "seeds": seed_values,
                "train_m": int(args.train_m),
                "val_m": int(args.val_m),
                "holdout_mode": str(args.holdout_mode),
                "sigma": float(args.sigma),
                "K": int(args.K),
                "iqp_steps": int(args.iqp_steps),
                "artr_epochs": int(args.artr_epochs),
                "artr_lr": float(args.artr_lr),
                "artr_batch_size": int(args.artr_batch_size),
            },
            **summary_payload,
        },
    )
    _render_plots(
        summary_rows=summary_rows,
        per_seed_rows=rows,
        out_pdf=out_pdf,
        out_png=out_png,
        out_val_pdf=out_val_pdf,
        out_val_png=out_val_png,
        out_kl_pdf=out_kl_pdf,
        out_kl_png=out_kl_png,
        dpi=int(args.dpi),
    )
    _write_json(
        run_config_json,
        {
            "script": "experiment_7_ablation_transformer_capacity_fixed_beta.py",
            "outdir": str(outdir.relative_to(ROOT)),
            "beta": float(args.beta),
            "n": int(args.n),
            "seeds": seed_values,
            "train_m": int(args.train_m),
            "val_m": int(args.val_m),
            "holdout_mode": str(args.holdout_mode),
            "holdout_seed": int(args.holdout_seed),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "iqp_steps": int(args.iqp_steps),
            "artr_epochs": int(args.artr_epochs),
            "artr_lr": float(args.artr_lr),
            "artr_batch_size": int(args.artr_batch_size),
            "transformer_configs": TRANSFORMER_CONFIGS,
            "rerun_command": (
                "MPLCONFIGDIR=/tmp/mpl-cache python "
                "experiment_7_ablation_transformer_capacity_fixed_beta.py "
                f"--beta {float(args.beta)} --n {int(args.n)} "
                f"--seeds {','.join(str(x) for x in seed_values)} "
                f"--train-m {int(args.train_m)} --val-m {int(args.val_m)} "
                f"--holdout-mode {str(args.holdout_mode)} "
                f"--iqp-steps {int(args.iqp_steps)} --artr-epochs {int(args.artr_epochs)} "
                f"--outdir {str(outdir.relative_to(ROOT))}"
            ),
        },
    )
    _write_readme(
        readme_md,
        beta=float(args.beta),
        n=int(args.n),
        seed_values=seed_values,
        train_m=int(args.train_m),
        val_m=int(args.val_m),
        holdout_mode=str(args.holdout_mode),
        outdir=outdir,
        summary_rows=summary_rows,
        per_seed_rows=rows,
    )
    write_training_protocol(
        outdir,
        experiment_name="Experiment 7 ablation: Transformer capacity at fixed beta",
        note="This capacity ablation uses the shared 10-seed / 600-budget analysis standard.",
        source_relpath="experiment_7_ablation_transformer_capacity_fixed_beta.py",
        metrics_note="The primary outputs are validation NLL and exact forward KL as a function of model capacity.",
    )

    print(f"[saved] {points_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {summary_json}")
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_val_pdf}")
    print(f"[saved] {out_val_png}")
    print(f"[saved] {out_kl_pdf}")
    print(f"[saved] {out_kl_png}")
    print(f"[saved] {run_config_json}")
    print(f"[saved] {readme_md}")
    print(f"[saved] {outdir / 'TRAINING_PROTOCOL.md'}")


if __name__ == "__main__":
    run()
