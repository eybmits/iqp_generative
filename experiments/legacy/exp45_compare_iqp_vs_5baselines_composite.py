#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare IQP parity vs 5 baselines using bucket-support metrics.

Metrics per model:
  - TV_score: score-bucket mass TV distance on full support (lower better)
  - BSHS(Q): bucket-weighted support-hit at budget Q (higher better)
  - Composite(Q) = BSHS(Q) * (1 - TV_score) (higher better)

Outputs:
  - model_comparison_metrics.csv
  - model_comparison_composite_plot.pdf
  - model_comparison_composite_plot.png
  - model_comparison_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402
from experiments.legacy import exp11_beta_sweep_global_holdout as exp11  # noqa: E402


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


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


def _bucket_hit_curve_component(q: np.ndarray, level_mask: np.ndarray, q_eval: int) -> float:
    y = hv.expected_unique_fraction(q, level_mask, np.array([int(q_eval)], dtype=int))
    return float(y[0]) if y.size else 0.0


def compute_support_bucket_metrics(
    p_star: np.ndarray,
    q: np.ndarray,
    support_mask: np.ndarray,
    scores: np.ndarray,
    q_eval: int,
) -> Dict[str, float]:
    levels = _score_levels(scores, support_mask)
    target_mass = _mass_by_level(p_star, scores, levels, support_mask)
    model_mass = _mass_by_level(q, scores, levels, support_mask)

    target_share = _normalized(target_mass)
    model_share = _normalized(model_mass)

    tv_score = 0.5 * float(np.sum(np.abs(target_share - model_share)))

    s_int = scores.astype(np.int64)
    bucket_hit = np.zeros(levels.shape[0], dtype=np.float64)
    for i, lv in enumerate(levels):
        level_mask = support_mask & (s_int == int(lv))
        bucket_hit[i] = _bucket_hit_curve_component(q, level_mask, q_eval=q_eval)

    bshs = float(np.sum(target_share * bucket_hit))
    composite = float(bshs * (1.0 - tv_score))

    return {
        "TV_score": float(tv_score),
        "BSHS": float(bshs),
        "Composite": float(composite),
    }


def _plot_comparison(rows: List[Dict[str, object]], out_pdf: Path, out_png: Path, q_eval: int) -> None:
    # Keep canonical model order from exp11 so comparisons stay consistent across runs.
    order = [k for k, *_ in exp11.MODEL_SPECS]
    by_key = {str(r["model_key"]): r for r in rows}
    rows_ordered = [by_key[k] for k in order if k in by_key]

    labels = [str(r["model_short"]) for r in rows_ordered]
    comps = np.array([float(r["Composite"]) for r in rows_ordered], dtype=np.float64)
    tvs = np.array([float(r["TV_score"]) for r in rows_ordered], dtype=np.float64)
    bshs = np.array([float(r["BSHS"]) for r in rows_ordered], dtype=np.float64)
    colors = [str(r["color"]) for r in rows_ordered]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 3.5), constrained_layout=True)

    x = np.arange(len(labels))
    bars = ax0.bar(x, comps, color=colors, alpha=0.85)
    # Highlight our model label visually.
    for i, r in enumerate(rows_ordered):
        if str(r["model_key"]) == "iqp_parity_mse":
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(1.4)

    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=28, ha="right")
    ax0.set_ylabel(r"Composite = BSHS$(Q)\cdot(1-\mathrm{TV}_{score})$")
    ax0.set_title(f"Single-score model ranking @ Q={q_eval}")
    ax0.set_ylim(0.0, max(1e-6, float(np.max(comps)) * 1.15))

    ax1.scatter(tvs, bshs, c=colors, s=58, alpha=0.9, edgecolors="black", linewidths=0.5)
    for i, lab in enumerate(labels):
        ax1.annotate(lab, (tvs[i], bshs[i]), textcoords="offset points", xytext=(5, 4), fontsize=7)

    ax1.set_xlabel(r"TV$_{score}$ (lower better)")
    ax1.set_ylabel(r"BSHS$(Q)$ (higher better)")
    ax1.set_title("Mass-fit vs support-hit tradeoff")
    ax1.set_xlim(0.0, max(0.05, float(np.max(tvs)) * 1.15))
    ax1.set_ylim(0.0, 1.02)

    fig.suptitle("IQP parity vs 5 baselines (full-support bucket metrics)", y=1.02)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")
    if not exp11.exp10.HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    hv.set_style(base=8)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p_star, scores, holdout_mask, model_rows, _metrics_rows = exp11._train_models_for_beta(
        holdout_mode=str(args.holdout_mode),
        n=int(args.n),
        beta=float(args.beta),
        seed=int(args.seed),
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

    # Common comparison budget Q: by default use IQP parity Q80 on holdout.
    q_by_key = {str(r["key"]): np.asarray(r["q"], dtype=np.float64) for r in model_rows}
    if str(args.q_eval_mode) == "iqp_q80":
        q80_iqp = float(hv.find_Q_threshold(q_by_key["iqp_parity_mse"], holdout_mask, thr=float(args.q80_thr), Qmax=int(args.q80_search_max)))
        q_eval = int(args.q_eval) if args.q_eval is not None else (int(args.qmax) if not math.isfinite(q80_iqp) else max(1, min(int(math.ceil(q80_iqp)), int(args.qmax))))
    elif str(args.q_eval_mode) == "fixed":
        q_eval = int(args.q_eval) if args.q_eval is not None else int(args.qmax)
    else:
        raise ValueError(f"Unsupported q_eval_mode: {args.q_eval_mode}")

    support_mask = p_star > 0.0

    rows: List[Dict[str, object]] = []
    color_by_key = {k: c for (k, _lbl, c, _ls, _lw) in exp11.MODEL_SPECS}
    short_by_key = {**exp11.MODEL_LABEL_SHORT}

    for row in model_rows:
        key = str(row["key"])
        q = np.asarray(row["q"], dtype=np.float64)
        m = compute_support_bucket_metrics(
            p_star=p_star,
            q=q,
            support_mask=support_mask,
            scores=scores,
            q_eval=q_eval,
        )
        rows.append(
            {
                "model_key": key,
                "model_label": str(row["label"]),
                "model_short": short_by_key.get(key, key),
                "color": color_by_key.get(key, "#666666"),
                "TV_score": float(m["TV_score"]),
                "BSHS": float(m["BSHS"]),
                "Composite": float(m["Composite"]),
                "q_eval": int(q_eval),
                "beta": float(args.beta),
                "holdout_mode": str(args.holdout_mode),
                "train_m": int(args.train_m),
                "sigma": float(args.sigma),
                "K": int(args.K),
                "seed": int(args.seed),
            }
        )

    rows.sort(key=lambda r: float(r["Composite"]), reverse=True)

    csv_path = outdir / "model_comparison_metrics.csv"
    plot_pdf = outdir / "model_comparison_composite_plot.pdf"
    plot_png = outdir / "model_comparison_composite_plot.png"
    json_path = outdir / "model_comparison_summary.json"

    _write_csv(csv_path, rows)
    _plot_comparison(rows, out_pdf=plot_pdf, out_png=plot_png, q_eval=q_eval)

    best = rows[0]
    summary = {
        "config": {
            "beta": float(args.beta),
            "holdout_mode": str(args.holdout_mode),
            "train_m": int(args.train_m),
            "holdout_k": int(args.holdout_k),
            "q_eval": int(q_eval),
            "q_eval_mode": str(args.q_eval_mode),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "seed": int(args.seed),
        },
        "best_model_by_composite": {
            "model_key": str(best["model_key"]),
            "model_label": str(best["model_label"]),
            "Composite": float(best["Composite"]),
            "BSHS": float(best["BSHS"]),
            "TV_score": float(best["TV_score"]),
        },
        "outputs": {
            "metrics_csv": str(csv_path),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[saved] {csv_path}")
    print(f"[saved] {plot_pdf}")
    print(f"[saved] {plot_png}")
    print(f"[saved] {json_path}")
    print(f"[q_eval] {q_eval}")
    print(f"[best] {best['model_key']} | Composite={best['Composite']:.4f} | BSHS={best['BSHS']:.4f} | TV={best['TV_score']:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare IQP parity against 5 baselines with bucket-support metrics.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "45_compare_iqp_vs_5baselines_composite"),
    )

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)

    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)

    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    ap.add_argument("--qmax", type=int, default=10000)

    ap.add_argument("--q-eval-mode", type=str, default="iqp_q80", choices=["iqp_q80", "fixed"])
    ap.add_argument("--q-eval", type=int, default=None)

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
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
