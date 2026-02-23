#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TV-vs-BSHS comparison plot in budgetlaw style over seeds and betas.

Produces a high-quality single-panel scatter (right-plot analogue) with:
  - x-axis: TV_score (lower better)
  - y-axis: BSHS(Q) (higher better)
  - color: beta (continuous colormap)
  - marker/edge style: model identity

Runs IQP parity + 5 baselines for each (beta, seed) combination.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.legacy import exp11_beta_sweep_global_holdout as exp11  # noqa: E402
from experiments.legacy import exp45_compare_iqp_vs_5baselines_composite as exp45  # noqa: E402
from iqp_generative import core as hv  # noqa: E402


MODEL_MARKERS: Dict[str, str] = {
    "iqp_parity_mse": "o",
    "iqp_prob_mse": "o",
    "classical_nnn_fields_parity": "o",
    "classical_dense_fields_xent": "o",
    "classical_transformer_mle": "o",
    "classical_maxent_parity": "o",
}


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_tv_bshs_budgetlaw_style(
    rows: List[Dict[str, object]],
    out_pdf: Path,
    out_png: Path,
    q_eval: int,
    dpi: int,
    title_suffix: str,
) -> None:
    if not rows:
        raise RuntimeError("No rows to plot.")

    df = pd.DataFrame(rows)
    model_specs = {k: (lbl, color, ls, lw) for (k, lbl, color, ls, lw) in exp11.MODEL_SPECS}

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
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.3, 6.1), dpi=dpi)
    ax.set_facecolor("#ececec")

    norm = Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap("Reds")

    for model_key, model_df in df.groupby("model_key", sort=False):
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
            s=130,
            marker=marker,
            edgecolors=edge_color,
            linewidths=1.7,
            alpha=0.90,
            zorder=4,
        )

        # Model centroid overlay for readability across many points.
        cx = float(np.mean(x))
        cy = float(np.mean(y))
        ax.scatter(
            [cx],
            [cy],
            s=250,
            marker=marker,
            facecolors="none",
            edgecolors=edge_color,
            linewidths=2.3,
            zorder=6,
        )

    ax.grid(which="major", linestyle="-", color="#c3c3c3", alpha=0.45, linewidth=0.70)
    ax.grid(which="minor", linestyle="-", color="#d3d3d3", alpha=0.32, linewidth=0.50)
    ax.minorticks_on()

    ax.set_xlabel(r"TV$_{score}$ (lower better)")
    ax.set_ylabel(r"BSHS$(Q)$ (higher better)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)

    if title_suffix:
        ax.set_title(f"IQP parity vs 5 baselines | {title_suffix}")
    else:
        ax.set_title("IQP parity vs 5 baselines")

    legend_handles: List[Line2D] = []
    for key, label, color, _ls, _lw in exp11.MODEL_SPECS:
        if key not in set(df["model_key"].astype(str).tolist()):
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
                markersize=10,
                label=label,
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Model",
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#b8b8b8",
        framealpha=1.0,
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


def run(args: argparse.Namespace) -> None:
    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")
    if not exp11.exp10.HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    hv.set_style(base=8)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    betas = _parse_list_floats(args.betas)
    seeds = _parse_list_ints(args.seeds)

    rows: List[Dict[str, object]] = []

    total_runs = len(betas) * len(seeds)
    run_idx = 0
    for beta in betas:
        for seed in seeds:
            run_idx += 1
            print(f"[run {run_idx}/{total_runs}] beta={beta:g} seed={seed}")

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

            support_mask = p_star > 0.0
            q_eval = int(args.q_eval)

            for mr in model_rows:
                key = str(mr["key"])
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

    points_csv = outdir / "tv_bshs_points_multiseed_beta.csv"
    summary_csv = outdir / "tv_bshs_summary_by_model.csv"
    plot_pdf = outdir / "tv_bshs_scatter_budgetlaw_style_multiseed_beta.pdf"
    plot_png = outdir / "tv_bshs_scatter_budgetlaw_style_multiseed_beta.png"
    summary_json = outdir / "tv_bshs_summary_multiseed_beta.json"

    _write_csv(points_csv, rows)

    df = pd.DataFrame(rows)
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
    summary_df.to_csv(summary_csv, index=False)

    title_suffix = (
        f"global={args.holdout_mode}, m={int(args.train_m)}, Q={int(args.q_eval)}, "
        f"betas={min(betas):g}..{max(betas):g}, seeds={len(seeds)}"
    )
    _plot_tv_bshs_budgetlaw_style(
        rows=rows,
        out_pdf=plot_pdf,
        out_png=plot_png,
        q_eval=int(args.q_eval),
        dpi=int(args.dpi),
        title_suffix=title_suffix,
    )

    best = summary_df.iloc[0].to_dict() if len(summary_df) else {}
    payload = {
        "config": {
            "n": int(args.n),
            "holdout_mode": str(args.holdout_mode),
            "train_m": int(args.train_m),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "betas": [float(b) for b in betas],
            "seeds": [int(s) for s in seeds],
            "q_eval": int(args.q_eval),
            "q80_thr": float(args.q80_thr),
        },
        "best_model_by_composite_mean": best,
        "outputs": {
            "points_csv": str(points_csv),
            "summary_csv": str(summary_csv),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[saved] {points_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {plot_pdf}")
    print(f"[saved] {plot_png}")
    print(f"[saved] {summary_json}")
    if best:
        print(
            "[best-mean]"
            f" {best.get('model_key')} | Composite_mean={float(best.get('Composite_mean', float('nan'))):.4f}"
            f" | BSHS_mean={float(best.get('BSHS_mean', float('nan'))):.4f}"
            f" | TV_mean={float(best.get('TV_score_mean', float('nan'))):.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Budgetlaw-style TV-vs-BSHS plot over seeds and betas.")
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
    ap.add_argument("--dpi", type=int, default=420)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
