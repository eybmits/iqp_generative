#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional restyle for beta-sweep summary plots.

Targets:
  - beta_sweep_Q80_vs_beta.pdf
  - beta_sweep_qH_vs_beta.pdf
  - beta_sweep_fit_tv_vs_beta.pdf

Style:
  - IQP parity: red, visually dominant
  - all other models: gray/black tones
  - no legend (per request)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


MODEL_ORDER = [
    "iqp_parity_mse",
    "iqp_prob_mse",
    "classical_nnn_fields_parity",
    "classical_dense_fields_xent",
    "classical_transformer_mle",
    "classical_maxent_parity",
]

# Same gray-scale language as the professional recovery strip.
STYLE = {
    "iqp_parity_mse": {"color": hv.COLORS["model"], "ls": "-", "lw": 2.6, "ms": 4.0, "z": 10, "alpha": 1.0},
    "iqp_prob_mse": {"color": "#111111", "ls": (0, (3, 2)), "lw": 1.9, "ms": 3.2, "z": 8, "alpha": 0.95},
    "classical_nnn_fields_parity": {"color": "#2B2B2B", "ls": "-", "lw": 1.8, "ms": 3.0, "z": 7, "alpha": 0.9},
    "classical_dense_fields_xent": {"color": "#4A4A4A", "ls": (0, (5, 2)), "lw": 1.8, "ms": 3.0, "z": 6, "alpha": 0.9},
    "classical_transformer_mle": {"color": "#6A6A6A", "ls": "--", "lw": 1.9, "ms": 3.0, "z": 5, "alpha": 0.9},
    "classical_maxent_parity": {"color": "#8A8A8A", "ls": "-.", "lw": 1.9, "ms": 3.0, "z": 4, "alpha": 0.9},
}


def _load_rows(csv_path: Path) -> List[Dict[str, float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append(
                {
                    "beta": float(r["beta"]),
                    "model_key": str(r["model_key"]),
                    "Q80": float(r["Q80"]),
                    "qH": float(r["qH"]),
                    "fit_tv_to_pstar": float(r["fit_tv_to_pstar"]),
                }
            )
    return rows


def _group_by_model(rows: List[Dict[str, float]]) -> Dict[str, List[Dict[str, float]]]:
    by_model: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        by_model.setdefault(r["model_key"], []).append(r)
    for k in by_model:
        by_model[k].sort(key=lambda x: x["beta"])
    return by_model


def _plot_one(
    by_model: Dict[str, List[Dict[str, float]]],
    y_key: str,
    ylabel: str,
    outpath: Path,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.75), constrained_layout=True)

    # Draw non-IQP first; IQP red on top.
    for key in [k for k in MODEL_ORDER if k != "iqp_parity_mse"] + ["iqp_parity_mse"]:
        if key not in by_model:
            continue
        st = STYLE[key]
        xs = [r["beta"] for r in by_model[key]]
        ys = [r[y_key] for r in by_model[key]]
        ax.plot(
            xs,
            ys,
            color=st["color"],
            ls=st["ls"],
            lw=st["lw"],
            marker="o",
            ms=st["ms"],
            alpha=st["alpha"],
            zorder=st["z"],
        )

    if log_y:
        ax.set_yscale("log")

    betas = sorted({r["beta"] for model_rows in by_model.values() for r in model_rows})
    ax.set_xticks(betas)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(ylabel)

    # No legend; small in-panel cue only.
    ax.text(
        0.02,
        0.95,
        "IQP-parity = red",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=hv.COLORS["model"],
    )
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)

    fig.savefig(outpath)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Restyle beta-sweep summary plots without legend.")
    ap.add_argument(
        "--summary-dir",
        type=str,
        default=str(
            Path("outputs")
            / "paper_even_final"
            / "34_claim_beta_sweep_bestparams"
            / "global_m200_sigma1_k512"
            / "summary"
        ),
    )
    args = ap.parse_args()

    hv.set_style(base=8)
    summary_dir = Path(args.summary_dir)
    csv_path = summary_dir / "beta_sweep_metrics_long.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    rows = _load_rows(csv_path)
    by_model = _group_by_model(rows)

    _plot_one(
        by_model=by_model,
        y_key="Q80",
        ylabel=r"$Q_{80}$ (lower is better)",
        outpath=summary_dir / "beta_sweep_Q80_vs_beta.pdf",
        log_y=True,
    )
    _plot_one(
        by_model=by_model,
        y_key="qH",
        ylabel=r"$q(H)$",
        outpath=summary_dir / "beta_sweep_qH_vs_beta.pdf",
        log_y=False,
    )
    _plot_one(
        by_model=by_model,
        y_key="fit_tv_to_pstar",
        ylabel=r"TV($q,p^*$)",
        outpath=summary_dir / "beta_sweep_fit_tv_vs_beta.pdf",
        log_y=False,
    )

    print(f"[saved] {summary_dir / 'beta_sweep_Q80_vs_beta.pdf'}")
    print(f"[saved] {summary_dir / 'beta_sweep_qH_vs_beta.pdf'}")
    print(f"[saved] {summary_dir / 'beta_sweep_fit_tv_vs_beta.pdf'}")


if __name__ == "__main__":
    main()
