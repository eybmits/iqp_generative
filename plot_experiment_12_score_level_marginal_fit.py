#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rerender-only score-level marginal fit panel from saved Experiment 12 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from final_plot_style import save_exact_figure
from experiment_1_kl_diagnostics import (
    BAR_FIGSIZE,
    COLOR_IQP_MSE,
    COLOR_SUBTEXT,
    COLOR_TEXT,
    K_VALUES,
    SIGMA_VALUES,
    ROOT,
    apply_style,
    build_target_distribution_paper,
    get_iqp_pairs_nn_nnn,
    iqp_circuit_zz_only,
    qml,
)


DEFAULT_DATA_NPZ = ROOT / "plots" / "experiment_12_global_best_iqp_vs_mse" / "experiment_12_global_best_iqp_vs_mse_data.npz"
DEFAULT_RUN_CONFIG = ROOT / "plots" / "experiment_12_global_best_iqp_vs_mse" / "RUN_CONFIG.json"
DEFAULT_OUTDIR = ROOT / "plots" / "experiment_12_global_best_iqp_vs_mse"
DEFAULT_SEED = 118
PARITY_COLOR = "#ea8a7d"
TARGET_COLOR = "#4A4A4A"


def _weights_to_probs(*, n: int, layers: int, weights: np.ndarray) -> np.ndarray:
    dev = qml.device("default.qubit", wires=int(n))
    pairs = get_iqp_pairs_nn_nnn(int(n))

    @qml.qnode(dev, interface=None, diff_method=None)
    def circuit(w):
        iqp_circuit_zz_only(w, range(int(n)), pairs, layers=int(layers))
        return qml.probs(wires=range(int(n)))

    q = np.asarray(circuit(np.asarray(weights, dtype=np.float64)), dtype=np.float64)
    q = np.clip(q, 0.0, 1.0)
    q = q / max(1e-15, float(np.sum(q)))
    return q


def _score_masses(dist: np.ndarray, scores: np.ndarray, score_vals: Sequence[int]) -> np.ndarray:
    out = np.zeros(len(score_vals), dtype=np.float64)
    arr = np.asarray(dist, dtype=np.float64)
    for idx, score in enumerate(score_vals):
        out[idx] = float(np.sum(arr[scores == float(score)]))
    return out


def run() -> None:
    ap = argparse.ArgumentParser(description="Rerender score-level marginal fit panel from saved Experiment 12 artifacts.")
    ap.add_argument("--data-npz", type=str, default=str(DEFAULT_DATA_NPZ))
    ap.add_argument("--run-config", type=str, default=str(DEFAULT_RUN_CONFIG))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    data_npz = Path(args.data_npz).expanduser()
    run_config = Path(args.run_config).expanduser()
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(run_config.read_text(encoding="utf-8"))
    z = np.load(data_npz, allow_pickle=True)

    seeds = np.asarray(z["seeds"], dtype=np.int64)
    if int(args.seed) not in set(int(x) for x in seeds.tolist()):
        raise RuntimeError(f"Seed {int(args.seed)} not found in saved Experiment 12 data.")
    seed_idx = int(np.where(seeds == int(args.seed))[0][0])

    beta = float(cfg["beta"])
    n = int(cfg["n"])
    layers = int(cfg["layers"])
    p_star = np.asarray(z["p_star"], dtype=np.float64)
    global_best_sigma = float(np.asarray(z["global_best_sigma"]).reshape(-1)[0])
    global_best_k = int(np.asarray(z["global_best_k"]).reshape(-1)[0])
    parity_weights = np.asarray(z["global_best_parity_weights"], dtype=np.float64)[seed_idx]
    mse_weights = np.asarray(z["mse_weights"], dtype=np.float64)[seed_idx]
    parity_kl = float(np.asarray(z["parity_kl_grid"], dtype=np.float64)[seed_idx, SIGMA_VALUES.index(global_best_sigma), K_VALUES.index(global_best_k)])
    mse_kl = float(np.asarray(z["mse_kl"], dtype=np.float64)[seed_idx])

    _p_check, support, scores = build_target_distribution_paper(n, beta)
    score_vals = np.asarray(sorted(int(s) for s in np.unique(scores[support])), dtype=np.int64)
    q_parity = _weights_to_probs(n=n, layers=layers, weights=parity_weights)
    q_mse = _weights_to_probs(n=n, layers=layers, weights=mse_weights)

    target_mass = _score_masses(p_star, scores, score_vals)
    parity_mass = _score_masses(q_parity, scores, score_vals)
    mse_mass = _score_masses(q_mse, scores, score_vals)

    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE, constrained_layout=False)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.955, bottom=0.235)

    x = np.arange(score_vals.size, dtype=np.float64)
    width = 0.24
    ax.bar(x - width, target_mass, width=width, color=TARGET_COLOR, alpha=0.86, label="Target $p^*$", zorder=3)
    ax.bar(
        x,
        parity_mass,
        width=width,
        color=PARITY_COLOR,
        alpha=0.86,
        label=rf"IQP-parity (KL = {parity_kl:.3f})",
        zorder=3,
    )
    ax.bar(
        x + width,
        mse_mass,
        width=width,
        color=COLOR_IQP_MSE,
        alpha=0.86,
        label=rf"IQP-MSE (KL = {mse_kl:.3f})",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in score_vals.tolist()])
    ax.set_xlabel(r"Score level $s$")
    ax.set_ylabel(r"$p(\ell=s)$")
    ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        fontsize=6.8,
        borderpad=0.18,
        labelspacing=0.18,
        handlelength=1.1,
        handletextpad=0.32,
        facecolor="white",
        edgecolor=COLOR_SUBTEXT,
    )
    ymax = max(float(np.max(target_mass)), float(np.max(parity_mass)), float(np.max(mse_mass)))
    ax.set_ylim(0.0, ymax * 1.16)

    stem = outdir / f"experiment_12_score_level_marginal_fit_seed{int(args.seed)}"
    save_exact_figure(fig, stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=300)
    plt.close(fig)
    print(f"[saved] {stem.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    run()
