#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 4: fixed-beta recovery sigma-K triplet panels.

This script can either rerender from a saved NPZ payload or regenerate the
fixed-beta payload locally from the current Experiment 1 / Experiment 3 helper
functions. The approved selection rule for ``best spectral`` is now explicit:
pick the spectral setting with maximal recovery ``R(Q)`` at a fixed budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from experiment_1_kl_diagnostics import (  # noqa: E402
    HAS_PENNYLANE,
    build_parity_matrix,
    make_bits_table,
    sample_alphas,
    train_iqp_qcbm,
    train_iqp_qcbm_prob_mse,
)
from experiment_3_beta_quality_coverage import (  # noqa: E402
    ELITE_FRAC,
    PARITY_BAND_OFFSET,
    TRAIN_SAMPLE_OFFSET,
    build_target_distribution_paper,
    empirical_dist,
    sample_indices,
    topk_mask_by_scores,
)


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_4_recovery_sigmak_triplet.py"

DEFAULT_DATA_NPZ = ROOT / "plots" / "experiment_4_recovery_sigmak_triplet" / "coverage_sigmak_triplet_data.npz"
DEFAULT_OUTDIR = ROOT / "plots" / "experiment_4_recovery_sigmak_triplet"
DEFAULT_BETA = 0.9
DEFAULT_SEED = 45
DEFAULT_N = 12
DEFAULT_TRAIN_M = 200
DEFAULT_LAYERS = 1
DEFAULT_IQP_STEPS = 600
DEFAULT_IQP_LR = 0.05
DEFAULT_SIGMAS = (0.5, 1.0, 2.0, 3.0)
DEFAULT_KS = (128, 256, 512)
DEFAULT_BEST_BUDGET = 1000
DEFAULT_PARITY_REFERENCE_KEY = "sigma=1, K=512"

FIG_W = 243.12 / 72.0
FIG_H = 185.52 / 72.0

TARGET_COLOR = "#2F2A2B"
UNIFORM_COLOR = "#C6C9CF"
PARITY_BEST_COLOR = "#DC2626"
PARITY_MSE_COLOR = "#1F77B4"
SPECTRAL_BEST_COLOR = "#666666"


def apply_final_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 7.2,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "gray",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.03,
        }
    )


def _try_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _pure_red_shades(n: int) -> List[Tuple[float, float, float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [(0.84, 0.30, 0.34, 1.0)]
    c0 = np.array([0.97, 0.85, 0.86])
    c1 = np.array([0.84, 0.30, 0.34])
    out: List[Tuple[float, float, float, float]] = []
    for i in range(n):
        t = i / float(n - 1)
        c = (1.0 - t) * c0 + t * c1
        out.append((float(c[0]), float(c[1]), float(c[2]), 1.0))
    return out


def _gray_shades(n: int) -> List[Tuple[float, float, float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [(0.45, 0.45, 0.45, 1.0)]
    c0 = np.array([0.88, 0.88, 0.88])
    c1 = np.array([0.40, 0.40, 0.40])
    out: List[Tuple[float, float, float, float]] = []
    for i in range(n):
        t = i / float(n - 1)
        c = (1.0 - t) * c0 + t * c1
        out.append((float(c[0]), float(c[1]), float(c[2]), 1.0))
    return out


def _spectral_colors_by_budget(
    keys: List[str],
    curves_by_key: Dict[str, np.ndarray],
    Q: np.ndarray,
    budget_q: int,
) -> Dict[str, Tuple[float, float, float, float]]:
    if not keys:
        return {}
    values = [(str(key), _curve_value_at_budget(Q, curves_by_key[str(key)], int(budget_q))) for key in keys]
    values.sort(key=lambda item: item[1])
    if len(values) == 1:
        return {values[0][0]: (0.16, 0.16, 0.16, 1.0)}
    c0 = np.array([0.90, 0.90, 0.90])
    c1 = np.array([0.10, 0.10, 0.10])
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for idx, (key, _val) in enumerate(values):
        t = idx / float(len(values) - 1)
        c = (1.0 - t) * c0 + t * c1
        out[str(key)] = (float(c[0]), float(c[1]), float(c[2]), 1.0)
    return out


def _make_ax():
    apply_final_style()
    return plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(float(x.strip())) for x in str(s).split(",") if x.strip()]


def _q_grid(qmax: int = 2000) -> np.ndarray:
    q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.3, 100).astype(int)),
                np.linspace(400, qmax, 80).astype(int),
                np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=int),
            ]
        )
    )
    return q[(q >= 0) & (q <= qmax)]


def _reconstruct_bandlimited(P: np.ndarray, z: np.ndarray, n: int) -> np.ndarray:
    q_lin = (1.0 / (2 ** int(n))) * (1.0 + (P.T @ z))
    q = np.clip(np.asarray(q_lin, dtype=np.float64), 0.0, None)
    s = float(np.sum(q))
    if s <= 0.0:
        q = np.ones_like(q, dtype=np.float64) / float(q.size)
    else:
        q = q / s
    return np.asarray(q, dtype=np.float64)


def _expected_unique_fraction(q: np.ndarray, mask: np.ndarray, Q: np.ndarray) -> np.ndarray:
    qv = np.asarray(q, dtype=np.float64)
    holdout_mask = np.asarray(mask, dtype=bool)
    qgrid = np.asarray(Q, dtype=np.int64)
    H = int(np.sum(holdout_mask))
    if H <= 0:
        return np.zeros_like(qgrid, dtype=np.float64)
    probs = qv[holdout_mask][:, None]
    return np.sum(1.0 - np.power(1.0 - probs, qgrid[None, :]), axis=0) / float(H)


def _curve_value_at_budget(Q: np.ndarray, curve: np.ndarray, budget_q: int) -> float:
    qgrid = np.asarray(Q, dtype=np.int64)
    y = np.asarray(curve, dtype=np.float64)
    matches = np.where(qgrid == int(budget_q))[0]
    if matches.size > 0:
        return float(y[int(matches[0])])
    if qgrid.size == 0:
        return float("nan")
    return float(np.interp(float(budget_q), qgrid.astype(np.float64), y.astype(np.float64)))


def _select_best_key_by_budget(curves_by_key: Dict[str, np.ndarray], Q: np.ndarray, budget_q: int) -> str:
    best_key = ""
    best_val = float("-inf")
    for key, curve in curves_by_key.items():
        val = _curve_value_at_budget(Q, curve, int(budget_q))
        if val > best_val:
            best_val = float(val)
            best_key = str(key)
    if not best_key:
        raise RuntimeError("Failed to select a best key from empty curve family.")
    return best_key


def _resolve_reference_parity_key(payload: Dict[str, object]) -> str:
    parity_by_key = payload["parity_by_key"]
    if DEFAULT_PARITY_REFERENCE_KEY in parity_by_key:
        return DEFAULT_PARITY_REFERENCE_KEY
    return str(payload["best_parity_key"])


def _style_ax(ax: plt.Axes, Q: np.ndarray, ylabel: str = r"$R(Q)$") -> None:
    ax.set_xlim(0.0, float(Q[-1]))
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.16, linestyle="--")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x))}"))
    ax.set_ylim(0.0, 1.0)


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _compute_recovery_payload(
    *,
    beta: float,
    seed: int,
    n: int,
    train_m: int,
    layers: int,
    iqp_steps: int,
    iqp_lr: float,
    sigmas: List[float],
    Ks: List[int],
    qmax: int,
    best_budget_q: int,
) -> Dict[str, object]:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required to recompute Experiment 4 recovery curves.")

    p_star, support, scores = build_target_distribution_paper(int(n), float(beta))
    elite_mask = topk_mask_by_scores(scores, support, frac=float(ELITE_FRAC))
    idxs_train = sample_indices(p_star, int(train_m), seed=int(seed) + int(TRAIN_SAMPLE_OFFSET))
    emp = empirical_dist(idxs_train, p_star.size)

    seen_mask = np.zeros_like(support, dtype=bool)
    seen_mask[np.unique(np.asarray(idxs_train, dtype=np.int64))] = True
    elite_unseen_mask = np.asarray(elite_mask & (~seen_mask), dtype=bool)
    elite_unseen_count = int(np.sum(elite_unseen_mask))
    if elite_unseen_count <= 0:
        raise RuntimeError("Elite unseen set is empty; cannot build recovery triplet.")

    Q = _q_grid(int(qmax))
    bits_table = make_bits_table(int(n))
    parity_by_key: Dict[str, np.ndarray] = {}
    spectral_by_key: Dict[str, np.ndarray] = {}

    for sigma in sigmas:
        for kval in Ks:
            key = f"sigma={float(sigma):g}, K={int(kval)}"
            alphas = sample_alphas(int(n), float(sigma), int(kval), seed=int(seed) + int(PARITY_BAND_OFFSET))
            P = build_parity_matrix(alphas, bits_table)
            z_data = P @ emp

            q_spec = _reconstruct_bandlimited(P, z_data, int(n))
            spectral_by_key[key] = _expected_unique_fraction(q_spec, elite_unseen_mask, Q)

            q_parity = train_iqp_qcbm(
                n=int(n),
                layers=int(layers),
                steps=int(iqp_steps),
                lr=float(iqp_lr),
                P=P,
                z_data=z_data,
                seed_init=int(seed) + 10000 + 7 * int(kval),
            )
            parity_by_key[key] = _expected_unique_fraction(q_parity, elite_unseen_mask, Q)

    q_uniform = np.ones_like(p_star, dtype=np.float64) / float(p_star.size)
    q_mse = train_iqp_qcbm_prob_mse(
        n=int(n),
        layers=int(layers),
        steps=int(iqp_steps),
        lr=float(iqp_lr),
        emp_dist=emp,
        seed_init=int(seed) + 20000 + 7 * 512,
    )

    best_parity_key = DEFAULT_PARITY_REFERENCE_KEY if DEFAULT_PARITY_REFERENCE_KEY in parity_by_key else _select_best_key_by_budget(parity_by_key, Q, int(best_budget_q))
    best_spectral_key = _select_best_key_by_budget(spectral_by_key, Q, int(best_budget_q))

    return {
        "Q": np.asarray(Q, dtype=np.int64),
        "target_curve": _expected_unique_fraction(p_star, elite_unseen_mask, Q),
        "uniform_curve": _expected_unique_fraction(q_uniform, elite_unseen_mask, Q),
        "iqp_mse_curve": _expected_unique_fraction(q_mse, elite_unseen_mask, Q),
        "parity_by_key": parity_by_key,
        "spectral_by_key": spectral_by_key,
        "best_parity_key": best_parity_key,
        "best_spectral_key": best_spectral_key,
        "best_selection_budget_q": int(best_budget_q),
        "beta": float(beta),
        "seed": int(seed),
        "elite_unseen_count": int(elite_unseen_count),
        "curve_storage": "recovery_fraction",
    }


def _save_recovery_payload(payload: Dict[str, object], data_npz: Path) -> None:
    parity_keys = list(payload["parity_by_key"].keys())
    spectral_keys = list(payload["spectral_by_key"].keys())
    np.savez(
        data_npz,
        Q=np.asarray(payload["Q"], dtype=np.int64),
        target_curve=np.asarray(payload["target_curve"], dtype=np.float64),
        uniform_curve=np.asarray(payload["uniform_curve"], dtype=np.float64),
        iqp_mse_curve=np.asarray(payload["iqp_mse_curve"], dtype=np.float64),
        parity_labels=np.asarray(parity_keys, dtype=object),
        parity_curves=np.asarray([payload["parity_by_key"][k] for k in parity_keys], dtype=np.float64),
        spectral_labels=np.asarray(spectral_keys, dtype=object),
        spectral_curves=np.asarray([payload["spectral_by_key"][k] for k in spectral_keys], dtype=np.float64),
        best_parity_key=np.asarray(str(payload["best_parity_key"]), dtype=object),
        best_spectral_key=np.asarray(str(payload["best_spectral_key"]), dtype=object),
        best_selection_budget_q=np.asarray([int(payload["best_selection_budget_q"])], dtype=np.int64),
        beta=np.asarray([float(payload["beta"])], dtype=np.float64),
        seed=np.asarray([int(payload["seed"])], dtype=np.int64),
        elite_unseen_count=np.asarray([int(payload["elite_unseen_count"])], dtype=np.int64),
        curve_storage=np.asarray(["recovery_fraction"], dtype=object),
    )


def _load_recovery_payload(data_npz: Path) -> Dict[str, object]:
    with np.load(data_npz, allow_pickle=True) as z:
        Q = np.asarray(z["Q"], dtype=np.int64)
        target_curve = np.asarray(z["target_curve"], dtype=np.float64)
        uniform_curve = np.asarray(z["uniform_curve"], dtype=np.float64)
        iqp_mse_curve = np.asarray(z["iqp_mse_curve"], dtype=np.float64)
        parity_labels = [str(x) for x in z["parity_labels"].tolist()]
        parity_curves = np.asarray(z["parity_curves"], dtype=np.float64)
        spectral_labels = [str(x) for x in z["spectral_labels"].tolist()]
        spectral_curves = np.asarray(z["spectral_curves"], dtype=np.float64)
        best_parity_key = str(z["best_parity_key"].tolist())
        best_spectral_key = str(z["best_spectral_key"].tolist())
        beta = float(np.asarray(z["beta"]).ravel()[0])
        seed = int(np.asarray(z["seed"]).ravel()[0])
        elite_unseen_count = int(np.asarray(z["elite_unseen_count"]).ravel()[0])
        best_selection_budget_q = int(np.asarray(z["best_selection_budget_q"]).ravel()[0]) if "best_selection_budget_q" in z else DEFAULT_BEST_BUDGET
        curve_storage = str(np.asarray(z["curve_storage"]).ravel()[0]) if "curve_storage" in z else "quality_coverage"

    if curve_storage == "recovery_fraction":
        parity_by_key = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in zip(parity_labels, parity_curves)
        }
        spectral_by_key = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in zip(spectral_labels, spectral_curves)
        }
        target_curve_scaled = np.asarray(target_curve, dtype=np.float64)
        uniform_curve_scaled = np.asarray(uniform_curve, dtype=np.float64)
        iqp_mse_curve_scaled = np.asarray(iqp_mse_curve, dtype=np.float64)
    else:
        H = max(1, int(elite_unseen_count))
        scale = np.asarray(Q, dtype=np.float64) / float(H)
        parity_by_key = {
            k: np.asarray(v, dtype=np.float64) * scale
            for k, v in zip(parity_labels, parity_curves)
        }
        spectral_by_key = {
            k: np.asarray(v, dtype=np.float64) * scale
            for k, v in zip(spectral_labels, spectral_curves)
        }
        target_curve_scaled = np.asarray(target_curve, dtype=np.float64) * scale
        uniform_curve_scaled = np.asarray(uniform_curve, dtype=np.float64) * scale
        iqp_mse_curve_scaled = np.asarray(iqp_mse_curve, dtype=np.float64) * scale

    return {
        "Q": Q,
        "target_curve": target_curve_scaled,
        "uniform_curve": uniform_curve_scaled,
        "iqp_mse_curve": iqp_mse_curve_scaled,
        "parity_by_key": parity_by_key,
        "spectral_by_key": spectral_by_key,
        "best_parity_key": best_parity_key,
        "best_spectral_key": best_spectral_key,
        "best_selection_budget_q": int(best_selection_budget_q),
        "beta": beta,
        "seed": seed,
        "elite_unseen_count": elite_unseen_count,
    }


def render_triplet(
    *,
    payload: Dict[str, object],
    outdir: Path,
    comparison_spectral_key: str,
) -> List[Path]:
    Q = payload["Q"]
    target_curve = payload["target_curve"]
    uniform_curve = payload["uniform_curve"]
    iqp_mse_curve = payload["iqp_mse_curve"]
    parity_by_key = payload["parity_by_key"]
    spectral_by_key = payload["spectral_by_key"]
    best_parity_key = str(payload["best_parity_key"])
    best_spectral_key = str(payload["best_spectral_key"])
    best_selection_budget_q = int(payload["best_selection_budget_q"])

    parity_labels = list(parity_by_key.keys())
    spectral_labels = list(spectral_by_key.keys())
    red_shades = _pure_red_shades(len(parity_labels))
    gray_shades = _gray_shades(len(spectral_labels))
    spectral_colors = _spectral_colors_by_budget(spectral_labels, spectral_by_key, Q, best_selection_budget_q)

    saved: List[Path] = []

    fig1, ax1 = _make_ax()
    ax1.plot(Q, target_curve, color=TARGET_COLOR, lw=2.1, zorder=5)
    ax1.plot(Q, uniform_curve, color=UNIFORM_COLOR, lw=1.7, ls=":", zorder=4)
    ax1.plot(Q, spectral_by_key[comparison_spectral_key], color=SPECTRAL_BEST_COLOR, lw=2.0, ls="-.", zorder=5)
    ax1.plot(Q, parity_by_key[best_parity_key], color=PARITY_BEST_COLOR, lw=2.8, zorder=6)
    _style_ax(ax1, Q)
    ax1.legend(
        handles=[
            Line2D([0], [0], color=TARGET_COLOR, lw=2.1, label=r"Target $p^*$"),
            Line2D([0], [0], color=PARITY_BEST_COLOR, lw=2.8, label=f"IQP parity ({best_parity_key})"),
            Line2D([0], [0], color=SPECTRAL_BEST_COLOR, lw=2.0, ls="-.", label=f"Spectral completion ({comparison_spectral_key})"),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.7, ls=":", label="Uniform"),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#bfbfbf",
        fontsize=6.8,
    )
    path1 = outdir / "experiment_4_recovery_best_iqp_vs_best_spectral.pdf"
    _save_pdf(fig1, path1)
    saved.append(path1)

    fig2, ax2 = _make_ax()
    ax2.plot(Q, target_curve, color=TARGET_COLOR, lw=2.1, zorder=5)
    ax2.plot(Q, uniform_curve, color=UNIFORM_COLOR, lw=1.7, ls=":", zorder=4)
    for (key, y), c in zip(parity_by_key.items(), red_shades):
        if key == best_parity_key:
            continue
        ax2.plot(Q, y, color=c, lw=1.35, alpha=0.98, zorder=2)
    ax2.plot(Q, parity_by_key[best_parity_key], color=PARITY_BEST_COLOR, lw=2.8, zorder=6)
    ax2.plot(Q, iqp_mse_curve, color=PARITY_MSE_COLOR, lw=2.0, zorder=5)
    _style_ax(ax2, Q)
    ax2.legend(
        handles=[
            Line2D([0], [0], color=TARGET_COLOR, lw=2.1, label=r"Target $p^*$"),
            Line2D([0], [0], color=PARITY_BEST_COLOR, lw=2.8, label=f"IQP parity ({best_parity_key})"),
            Line2D([0], [0], color=red_shades[-1], lw=1.4, label="IQP parity (other σ,K)"),
            Line2D([0], [0], color=PARITY_MSE_COLOR, lw=2.0, label="IQP MSE"),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.7, ls=":", label="Uniform"),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#bfbfbf",
        fontsize=6.8,
    )
    path2 = outdir / "experiment_4_recovery_parity_sigmak_vs_iqp_mse.pdf"
    _save_pdf(fig2, path2)
    saved.append(path2)

    fig3, ax3 = _make_ax()
    ax3.plot(Q, target_curve, color=TARGET_COLOR, lw=2.1, zorder=5)
    ax3.plot(Q, uniform_curve, color=UNIFORM_COLOR, lw=1.7, ls=":", zorder=4)
    for key, y in spectral_by_key.items():
        if key == comparison_spectral_key:
            continue
        ax3.plot(Q, y, color=spectral_colors[str(key)], lw=1.35, alpha=0.98, zorder=2)
    ax3.plot(
        Q,
        spectral_by_key[comparison_spectral_key],
        color=spectral_colors[str(comparison_spectral_key)],
        lw=2.2,
        ls="-.",
        zorder=6,
    )
    _style_ax(ax3, Q)
    ax3.legend(
        handles=[
            Line2D([0], [0], color=TARGET_COLOR, lw=2.1, label=r"Target $p^*$"),
            Line2D([0], [0], color=spectral_colors[str(comparison_spectral_key)], lw=2.2, ls="-.", label=f"Spectral completion ({comparison_spectral_key})"),
            Line2D([0], [0], color=(0.45, 0.45, 0.45, 1.0), lw=1.4, label=r"Spectral completion (other $\sigma,K$)"),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.7, ls=":", label="Uniform"),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#bfbfbf",
        fontsize=6.8,
    )
    path3 = outdir / "experiment_4_recovery_spectral_sigmak_only.pdf"
    _save_pdf(fig3, path3)
    saved.append(path3)

    return saved


def run() -> None:
    ap = argparse.ArgumentParser(description="Experiment 4: standalone recovery sigma-K triplet panels.")
    ap.add_argument("--data-npz", type=str, default=str(DEFAULT_DATA_NPZ))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--comparison-spectral-key", type=str, default="")
    ap.add_argument("--recompute-data", type=int, default=0)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--train-m", type=int, default=DEFAULT_TRAIN_M)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--iqp-steps", type=int, default=DEFAULT_IQP_STEPS)
    ap.add_argument("--iqp-lr", type=float, default=DEFAULT_IQP_LR)
    ap.add_argument("--sigmas", type=str, default="0.5,1,2,3")
    ap.add_argument("--Ks", type=str, default="128,256,512")
    ap.add_argument("--qmax", type=int, default=2000)
    ap.add_argument("--best-budget-q", type=int, default=DEFAULT_BEST_BUDGET)
    args = ap.parse_args()

    data_npz = Path(args.data_npz)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if int(args.recompute_data) == 1 or not data_npz.exists():
        payload = _compute_recovery_payload(
            beta=float(args.beta),
            seed=int(args.seed),
            n=int(args.n),
            train_m=int(args.train_m),
            layers=int(args.layers),
            iqp_steps=int(args.iqp_steps),
            iqp_lr=float(args.iqp_lr),
            sigmas=_parse_float_list(args.sigmas),
            Ks=_parse_int_list(args.Ks),
            qmax=int(args.qmax),
            best_budget_q=int(args.best_budget_q),
        )
        _save_recovery_payload(payload, data_npz)
    else:
        payload = _load_recovery_payload(data_npz)

    payload["best_parity_key"] = _resolve_reference_parity_key(payload)
    comparison_spectral_key = str(args.comparison_spectral_key).strip() or str(payload["best_parity_key"])
    if comparison_spectral_key not in payload["spectral_by_key"]:
        raise ValueError(f"Unknown spectral key: {comparison_spectral_key}")

    pdfs = render_triplet(
        payload=payload,
        outdir=outdir,
        comparison_spectral_key=comparison_spectral_key,
    )

    run_config = {
        "script": SCRIPT_REL,
        "data_npz": _try_rel(data_npz),
        "outdir": _try_rel(outdir),
        "beta": float(payload["beta"]),
        "seed": int(payload["seed"]),
        "metric": "recovery",
        "comparison_spectral_key": comparison_spectral_key,
        "best_parity_key": str(payload["best_parity_key"]),
        "best_spectral_key": str(payload["best_spectral_key"]),
        "best_spectral_selection_metric": "max_recovery_at_Q",
        "best_spectral_selection_budget_q": int(payload["best_selection_budget_q"]),
        "elite_unseen_count": int(payload["elite_unseen_count"]),
        "pdf_only": True,
    }
    _write_path = outdir / "RUN_CONFIG.json"
    with _write_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")

    rerender = {
        "script": SCRIPT_REL,
        "data_npz": _try_rel(data_npz),
        "outdir": _try_rel(outdir),
        "rerender_command": (
            f"MPLCONFIGDIR=/tmp/mpl-cache python {SCRIPT_REL} "
            f"--data-npz {_try_rel(data_npz)} --outdir {_try_rel(outdir)} "
            f"--comparison-spectral-key \"{comparison_spectral_key}\" "
            f"--best-budget-q {int(payload['best_selection_budget_q'])}"
        ),
        "pdfs": [_try_rel(p) for p in pdfs],
    }
    with (outdir / "RERENDER_CONFIG.json").open("w", encoding="utf-8") as f:
        json.dump(rerender, f, indent=2)
        f.write("\n")

    for pdf in pdfs:
        print(f"[saved] {pdf}")


if __name__ == "__main__":
    run()
