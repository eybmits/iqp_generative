#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional horizontal recovery strip for beta sweep.

Design goals requested by user:
  - only recovery panel (left-panel content from the collage)
  - IQP (parity) always red
  - all other models in gray/black tones
  - encode fit quality (TV distance to target) directly in the recovery curves
  - horizontal figure for clean paper-style layout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402
from experiments.legacy import exp11_beta_sweep_global_holdout as exp11  # noqa: E402


MODEL_ORDER = [
    "iqp_parity_mse",
    "classical_nnn_fields_parity",
    "classical_dense_fields_xent",
    "classical_transformer_mle",
    "classical_maxent_parity",
]

MODEL_LABELS = {
    "iqp_parity_mse": "IQP (parity)",
    "classical_nnn_fields_parity": "Ising+fields (NN+NNN)",
    "classical_dense_fields_xent": "Dense Ising+fields (xent)",
    "classical_transformer_mle": "AR Transformer (MLE)",
    "classical_maxent_parity": "MaxEnt parity (P,z)",
}

# Default style: IQP parity red, everything else gray/black.
MODEL_STYLE_GRAY = {
    "iqp_parity_mse": {"color": hv.COLORS["model"], "ls": "-", "lw": 2.35, "z": 10},
    "classical_nnn_fields_parity": {"color": "#2B2B2B", "ls": "-", "lw": 1.8, "z": 7},
    "classical_dense_fields_xent": {"color": "#4A4A4A", "ls": (0, (5, 2)), "lw": 1.8, "z": 6},
    "classical_transformer_mle": {"color": "#6A6A6A", "ls": "--", "lw": 1.9, "z": 5},
    "classical_maxent_parity": {"color": "#8A8A8A", "ls": "-.", "lw": 1.9, "z": 4},
}

# Alternative style: colored baselines (keeps IQP red).
MODEL_STYLE_COLOR = {
    "iqp_parity_mse": {"color": hv.COLORS["model"], "ls": "-", "lw": 2.35, "z": 10},
    "classical_nnn_fields_parity": {"color": "#1f77b4", "ls": "-", "lw": 1.85, "z": 7},
    "classical_dense_fields_xent": {"color": "#8c564b", "ls": (0, (5, 2)), "lw": 1.85, "z": 6},
    "classical_transformer_mle": {"color": "#17becf", "ls": "--", "lw": 1.9, "z": 5},
    "classical_maxent_parity": {"color": "#9467bd", "ls": "-.", "lw": 1.9, "z": 4},
}

LEGEND_STYLE = dict(
    loc="lower right",
    bbox_to_anchor=(0.985, 0.03),
    fontsize=6.3,
    frameon=True,
    framealpha=1.0,
    facecolor="#FFFFFF",
    edgecolor="#D8D8D8",
    handlelength=2.7,
    labelspacing=0.24,
    borderpad=0.26,
    handletextpad=0.55,
    borderaxespad=0.0,
)

PANEL_CACHE_VERSION = 1


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _tv_alpha_and_marker(tv_vals: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Map TV distances to visual emphasis:
      lower TV (better fit) -> darker/less transparent + larger endpoint marker.
    """
    tv_min = float(np.min(tv_vals))
    tv_max = float(np.max(tv_vals))
    span = max(1e-12, tv_max - tv_min)
    out: Dict[int, Dict[str, float]] = {}
    for i, tv in enumerate(tv_vals.tolist()):
        quality = 1.0 - ((float(tv) - tv_min) / span)  # 1=best, 0=worst
        alpha = 0.35 + 0.65 * quality
        msize = 2.2 + 3.0 * quality
        out[i] = {"alpha": float(alpha), "msize": float(msize)}
    return out


def _q_grid(Qmax: int = 10000) -> np.ndarray:
    q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 120).astype(int)),
                np.linspace(1000, Qmax, 160).astype(int),
                np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100], dtype=int),
            ]
        )
    )
    return q[(q >= 0) & (q <= Qmax)]


def _legend_handles(model_style: Dict[str, Dict[str, object]]) -> List[Line2D]:
    handles: List[Line2D] = [
        Line2D([0], [0], color=hv.COLORS["target"], lw=2.0, ls="-", label=r"Target $p^*$"),
    ]
    for key in MODEL_ORDER:
        st = model_style[key]
        handles.append(
            Line2D(
                [0],
                [0],
                color=str(st["color"]),
                lw=max(1.8, float(st["lw"]) * 1.06),
                ls=st["ls"],
                label=str(MODEL_LABELS[key]),
            )
        )
    handles.append(Line2D([0], [0], color=hv.COLORS["gray"], lw=1.75, ls="--", label="Uniform"))
    return handles


def _select_model_style(colorful_baselines: bool) -> Dict[str, Dict[str, object]]:
    return MODEL_STYLE_COLOR if colorful_baselines else MODEL_STYLE_GRAY


def _beta_tag(beta: float) -> str:
    return f"{beta:.3f}".replace("-", "m").replace(".", "p")


def _panel_cache_signature(args: argparse.Namespace, run_seeds: List[int], Q: np.ndarray) -> str:
    q_hash = hashlib.sha1(Q.astype(np.int32, copy=False).tobytes()).hexdigest()[:12]
    payload = {
        "version": PANEL_CACHE_VERSION,
        "holdout_mode": args.holdout_mode,
        "n": int(args.n),
        "run_seeds": [int(s) for s in run_seeds],
        "train_m": int(args.train_m),
        "sigma": float(args.sigma),
        "K": int(args.K),
        "layers": int(args.layers),
        "good_frac": float(args.good_frac),
        "holdout_k": int(args.holdout_k),
        "holdout_pool": int(args.holdout_pool),
        "holdout_m_train": int(args.holdout_m_train),
        "q80_thr": float(args.q80_thr),
        "q80_search_max": int(args.q80_search_max),
        "iqp_steps": int(args.iqp_steps),
        "iqp_lr": float(args.iqp_lr),
        "iqp_eval_every": int(args.iqp_eval_every),
        "artr_epochs": int(args.artr_epochs),
        "artr_d_model": int(args.artr_d_model),
        "artr_heads": int(args.artr_heads),
        "artr_layers": int(args.artr_layers),
        "artr_ff": int(args.artr_ff),
        "artr_lr": float(args.artr_lr),
        "artr_batch_size": int(args.artr_batch_size),
        "maxent_steps": int(args.maxent_steps),
        "maxent_lr": float(args.maxent_lr),
        "q_hash": q_hash,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:14]


def _panel_cache_file(cache_dir: Path, beta: float, signature: str) -> Path:
    return cache_dir / f"beta_{_beta_tag(beta)}_{signature}.npz"


def _load_panel_cache(
    cache_file: Path,
    Q: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
    if not cache_file.exists():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as z:
            version = int(z["cache_version"])
            if version != PANEL_CACHE_VERSION:
                return None
            q_cached = np.asarray(z["Q"], dtype=np.int32)
            if not np.array_equal(q_cached, Q.astype(np.int32, copy=False)):
                return None
            model_order = [str(x) for x in z["model_order"].tolist()]
            if model_order != MODEL_ORDER:
                return None
            y_target = np.asarray(z["y_target"], dtype=np.float64)
            y_unif = np.asarray(z["y_unif"], dtype=np.float64)
            y_models = np.asarray(z["y_models"], dtype=np.float64)
            tv_vals = np.asarray(z["tv_vals"], dtype=np.float64)
    except Exception as e:
        print(f"[cache-skip] failed to read {cache_file.name}: {e}")
        return None

    if y_target.shape != (Q.size,) or y_unif.shape != (Q.size,):
        return None
    if y_models.shape != (len(MODEL_ORDER), Q.size):
        return None
    if tv_vals.shape != (len(MODEL_ORDER),):
        return None

    y_by_key = {k: y_models[i] for i, k in enumerate(MODEL_ORDER)}
    return y_target, y_unif, y_by_key, tv_vals


def _save_panel_cache(
    cache_file: Path,
    Q: np.ndarray,
    y_target: np.ndarray,
    y_unif: np.ndarray,
    y_by_key: Dict[str, np.ndarray],
    tv_vals: np.ndarray,
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    y_models = np.stack([np.asarray(y_by_key[k], dtype=np.float64) for k in MODEL_ORDER], axis=0)
    np.savez_compressed(
        cache_file,
        cache_version=np.int64(PANEL_CACHE_VERSION),
        Q=Q.astype(np.int32, copy=False),
        model_order=np.array(MODEL_ORDER),
        y_target=np.asarray(y_target, dtype=np.float64),
        y_unif=np.asarray(y_unif, dtype=np.float64),
        y_models=y_models,
        tv_vals=np.asarray(tv_vals, dtype=np.float64),
    )


def _first_q_crossing(Q: np.ndarray, y: np.ndarray, thr: float) -> float:
    """Return the first Q where y reaches thr via linear interpolation."""
    idx = np.where(y >= thr)[0]
    if idx.size == 0:
        return float("inf")
    i = int(idx[0])
    if i == 0:
        return float(Q[0])
    x0, x1 = float(Q[i - 1]), float(Q[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 <= y0 + 1e-12:
        return x1
    t = (float(thr) - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return x0 + t * (x1 - x0)


def _compute_panel_from_training(
    args: argparse.Namespace,
    beta: float,
    run_seeds: List[int],
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    y_target_runs: List[np.ndarray] = []
    y_unif_runs: List[np.ndarray] = []
    y_runs_by_key: Dict[str, List[np.ndarray]] = {k: [] for k in MODEL_ORDER}
    tv_runs_by_key: Dict[str, List[float]] = {k: [] for k in MODEL_ORDER}

    for seed in run_seeds:
        p_star, _scores, holdout_mask, model_rows, metrics_rows = exp11._train_models_for_beta(
            holdout_mode=args.holdout_mode,
            n=args.n,
            beta=beta,
            seed=int(seed),
            train_m=args.train_m,
            sigma=args.sigma,
            K=args.K,
            layers=args.layers,
            holdout_k=args.holdout_k,
            holdout_pool=args.holdout_pool,
            holdout_m_train=args.holdout_m_train,
            good_frac=args.good_frac,
            iqp_steps=args.iqp_steps,
            iqp_lr=args.iqp_lr,
            iqp_eval_every=args.iqp_eval_every,
            q80_thr=args.q80_thr,
            q80_search_max=args.q80_search_max,
            artr_epochs=args.artr_epochs,
            artr_d_model=args.artr_d_model,
            artr_heads=args.artr_heads,
            artr_layers=args.artr_layers,
            artr_ff=args.artr_ff,
            artr_lr=args.artr_lr,
            artr_batch_size=args.artr_batch_size,
            maxent_steps=args.maxent_steps,
            maxent_lr=args.maxent_lr,
        )

        by_key = {str(r["model_key"]): r for r in metrics_rows}
        model_by_key = {str(r["key"]): r for r in model_rows}

        y_target_runs.append(hv.expected_unique_fraction(p_star, holdout_mask, Q))
        q_unif_seed = np.ones_like(p_star, dtype=np.float64) / p_star.size
        y_unif_runs.append(hv.expected_unique_fraction(q_unif_seed, holdout_mask, Q))

        for key in MODEL_ORDER:
            row = model_by_key[key]
            q = row["q"]
            assert isinstance(q, np.ndarray)
            y_runs_by_key[key].append(hv.expected_unique_fraction(q, holdout_mask, Q))
            if key in by_key:
                tv_runs_by_key[key].append(float(by_key[key]["fit_tv_to_pstar"]))

    y_target = np.mean(np.stack(y_target_runs, axis=0), axis=0)
    y_unif = np.mean(np.stack(y_unif_runs, axis=0), axis=0)
    y_by_key = {key: np.mean(np.stack(y_runs_by_key[key], axis=0), axis=0) for key in MODEL_ORDER}

    tv_vals = np.array(
        [float(np.mean(tv_runs_by_key[key])) if tv_runs_by_key[key] else np.nan for key in MODEL_ORDER],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(tv_vals)):
        finite = tv_vals[np.isfinite(tv_vals)]
        fill = float(np.mean(finite)) if finite.size else 1.0
        tv_vals = np.where(np.isfinite(tv_vals), tv_vals, fill)
    return y_target, y_unif, y_by_key, tv_vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Create professional horizontal recovery strip with TV encoding.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "paper_even_final"
            / "34_claim_beta_sweep_bestparams"
            / "global_m200_sigma1_k512"
        ),
    )
    ap.add_argument("--betas", type=str, default="0.6,0.7,0.8,0.9,1.0,1.1,1.2")
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--seeds", type=str, default="", help="Comma-separated seed list for multi-seed averaging.")
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
    ap.add_argument("--log-x", action="store_true")
    ap.add_argument("--grid-cols", type=int, default=0, help="If >0, arrange panels in a grid with this many columns.")
    ap.add_argument(
        "--legend-in-first-panel",
        type=int,
        default=1,
        choices=[0, 1],
        help="Show a shared legend in the first panel (recommended for 4x2 layouts).",
    )
    ap.add_argument(
        "--colorful-baselines",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use colored classical baselines (legend + curves) instead of gray tones.",
    )
    ap.add_argument(
        "--panel-cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="Reuse/save per-beta panel cache to rerender without retraining.",
    )
    ap.add_argument(
        "--panel-cache-readonly",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, require cache hit for every beta and never retrain.",
    )
    ap.add_argument(
        "--panel-cache-subdir",
        type=str,
        default="panel_cache",
        help="Subdirectory in outdir where per-beta panel cache files are stored.",
    )
    args = ap.parse_args()

    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")

    hv.set_style(base=8)

    outdir = Path(args.outdir)
    out_collage = outdir / "collage"
    out_collage.mkdir(parents=True, exist_ok=True)

    betas = _parse_list_floats(args.betas)
    run_seeds = _parse_list_ints(str(args.seeds)) if str(args.seeds).strip() else [int(args.seed)]
    if not run_seeds:
        run_seeds = [int(args.seed)]
    model_style = _select_model_style(bool(int(args.colorful_baselines)))
    Q = _q_grid(10000)
    q_eval = int(np.max(Q))
    use_panel_cache = bool(int(args.panel_cache))
    cache_readonly = bool(int(args.panel_cache_readonly))
    cache_dir = outdir / str(args.panel_cache_subdir)
    cache_sig = _panel_cache_signature(args, run_seeds, Q)
    if use_panel_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ro_txt = " readonly=1" if cache_readonly else ""
        print(f"[cache] dir={cache_dir} sig={cache_sig}{ro_txt}")

    # Layout: default horizontal strip, optional grid.
    if args.grid_cols and args.grid_cols > 0:
        ncols = int(args.grid_cols)
        nrows = int(np.ceil(len(betas) / ncols))
        fig_w = max(12.5, 2.6 * ncols)
        fig_h = max(4.8, 2.9 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True)
        axes = np.array(axes).reshape(-1)
    else:
        fig_w = max(13.5, 2.55 * len(betas))
        fig, axes = plt.subplots(1, len(betas), figsize=(fig_w, 2.9), sharey=True, constrained_layout=True)
        if len(betas) == 1:
            axes = np.array([axes])

    for i, beta in enumerate(betas):
        ax = axes[i]
        print(f"[pro-fig] beta={beta:g} | seeds={run_seeds}")
        cache_file = _panel_cache_file(cache_dir, beta, cache_sig)
        loaded = _load_panel_cache(cache_file, Q) if use_panel_cache else None
        if loaded is not None:
            y_target, y_unif, y_by_key, tv_vals = loaded
            print(f"[cache-hit] beta={beta:g} -> {cache_file.name}")
        else:
            if cache_readonly and use_panel_cache:
                raise RuntimeError(f"Cache miss for beta={beta:g}: {cache_file}")
            y_target, y_unif, y_by_key, tv_vals = _compute_panel_from_training(args, beta, run_seeds, Q)
            if use_panel_cache:
                _save_panel_cache(cache_file, Q, y_target, y_unif, y_by_key, tv_vals)
                print(f"[cache-save] beta={beta:g} -> {cache_file.name}")
        enc = _tv_alpha_and_marker(tv_vals)

        # Target and uniform references.
        ax.plot(Q, y_target, color=hv.COLORS["target"], linewidth=1.95, zorder=20)
        ax.plot(Q, y_unif, color=hv.COLORS["gray"], linewidth=1.35, linestyle="--", alpha=0.85, zorder=1)

        for j, key in enumerate(MODEL_ORDER):
            y = y_by_key[key]
            st = model_style[key]
            alpha = enc[j]["alpha"]
            msize = enc[j]["msize"]
            ax.plot(
                Q,
                y,
                color=st["color"],
                linestyle=st["ls"],
                linewidth=st["lw"],
                alpha=alpha,
                zorder=st["z"],
            )
            # Endpoint marker encodes TV quality (larger marker => better fit).
            y_end = float(np.interp(float(q_eval), Q.astype(np.float64), y.astype(np.float64)))
            ax.plot(
                [q_eval],
                [y_end],
                marker="o",
                markersize=msize,
                color=st["color"],
                alpha=alpha,
                zorder=st["z"] + 0.2,
            )

        # Highlight Q80 of the baseline that reaches the threshold first on the plotted curves.
        # Include the uniform baseline ("random sampling"), exclude target curve.
        candidate_curves: Dict[str, np.ndarray] = dict(y_by_key)
        candidate_curves["uniform_random"] = y_unif
        candidate_colors: Dict[str, str] = {k: str(model_style[k]["color"]) for k in MODEL_ORDER}
        candidate_colors["uniform_random"] = hv.COLORS["gray"]

        winner_key = None
        winner_q80 = float("inf")
        for key, y_curve in candidate_curves.items():
            q80 = _first_q_crossing(Q.astype(np.float64), y_curve.astype(np.float64), float(args.q80_thr))
            if np.isfinite(q80) and q80 < winner_q80:
                winner_q80 = float(q80)
                winner_key = key

        x_min = 1 if args.log_x else 0
        x_max = 10000
        if winner_key is not None and np.isfinite(winner_q80):
            q80_mark = float(np.clip(winner_q80, x_min, x_max))
            wcolor = candidate_colors[winner_key]
            # Interpolate on the *plotted* curve so the marker sits exactly on the shown baseline.
            y_q80 = float(np.interp(q80_mark, Q.astype(np.float64), candidate_curves[winner_key].astype(np.float64)))
            # Right side "fade-out": white overlay to keep visual focus on the left.
            ax.axvspan(q80_mark, x_max, color="#FFFFFF", alpha=0.42, zorder=25)
            ax.axvline(
                q80_mark,
                color=wcolor,
                linestyle="--",
                linewidth=1.2,
                alpha=0.95,
                zorder=28,
            )
            ax.plot(
                [q80_mark],
                [y_q80],
                marker="o",
                markersize=6.0,
                markerfacecolor=wcolor,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=30,
            )
            # Explicit horizontal Q80 label for the dashed line.
            if args.log_x:
                q80_text_x = float(min(x_max, q80_mark * 1.08))
            else:
                q80_text_x = float(min(x_max, q80_mark + 260.0))
            ax.text(
                q80_text_x,
                0.07,
                "Q80",
                color=wcolor,
                fontsize=7,
                rotation=0,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
                zorder=31,
            )

        if args.log_x:
            ax.set_xscale("log")
            ax.set_xlim(1, 10000)
        else:
            ax.set_xlim(0, 10000)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.6, linewidth=0.9)
        ax.set_title(fr"$\beta={beta:g}$")
        ax.set_xlabel(r"$Q$")
        if args.grid_cols and args.grid_cols > 0:
            if i % int(args.grid_cols) == 0:
                ax.set_ylabel(r"Recovery $R(Q)$")
            else:
                ax.tick_params(labelleft=False)
        elif i == 0:
            ax.set_ylabel(r"Recovery $R(Q)$")
        else:
            ax.tick_params(labelleft=False)

    # Hide unused axes in grid mode.
    if args.grid_cols and args.grid_cols > 0 and len(axes) > len(betas):
        for j in range(len(betas), len(axes)):
            axes[j].axis("off")

    if bool(int(args.legend_in_first_panel)) and len(axes) > 0:
        legend = axes[0].legend(handles=_legend_handles(model_style), **LEGEND_STYLE)
        legend.set_zorder(60)

    style_suffix = "_color_baselines" if bool(int(args.colorful_baselines)) else ""
    suffix = f"_multiseed_n{len(run_seeds)}" if len(run_seeds) > 1 else ""
    out_pdf = out_collage / f"recovery_horizontal_professional_with_tv_encoding{suffix}{style_suffix}.pdf"
    out_png = out_collage / f"recovery_horizontal_professional_with_tv_encoding{suffix}{style_suffix}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=260)
    plt.close(fig)

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
