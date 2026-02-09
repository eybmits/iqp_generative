#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 5: Discovery axis sweep
Systematically compares IQP parity-moment MSE vs prob-MSE on identical holdouts.

Outputs:
  - results_long.csv (one row per loss)
  - results_pair.csv (paired parity vs prob metrics per setting)
  - plots (Q80 vs K, Q80 ratio vs K) per (train_m, layers)
  - recovery curves (optional): per (train_m, layers) all K in one plot
"""

import os
import json
import csv
import argparse
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402

COLOR_PARITY = hv.COLORS["model"]  # red
COLOR_PROB = hv.COLORS["blue"]    # blue


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _nanmean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return float("nan")
    return float(a / b)


def _has_positive_finite(arr: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(arr) & (arr > 0)))


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_q80_vs_k(pair_rows: List[Dict], Ks: List[int], train_m: int, layers: int, outdir: str) -> None:
    data = [r for r in pair_rows if r["train_m"] == train_m and r["layers"] == layers]
    if not data:
        return

    ks_sorted = sorted(Ks)
    mean_parity, std_parity = [], []
    mean_prob, std_prob = [], []
    for k in ks_sorted:
        vals_p = [r["Q80_parity"] for r in data if r["K"] == k]
        vals_q = [r["Q80_prob"] for r in data if r["K"] == k]
        m, s = _nanmean_std(vals_p)
        mean_parity.append(m)
        std_parity.append(s)
        m, s = _nanmean_std(vals_q)
        mean_prob.append(m)
        std_prob.append(s)

    y_parity = np.array(mean_parity, dtype=np.float64)
    y_prob = np.array(mean_prob, dtype=np.float64)
    if not (_has_positive_finite(y_parity) or _has_positive_finite(y_prob)):
        print(f"[Skip] No positive finite Q80 values for m={train_m}, L={layers}.")
        return

    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    ax.plot(ks_sorted, mean_parity, color=COLOR_PARITY, linewidth=2.0, label="parity MSE")
    ax.fill_between(ks_sorted, np.array(mean_parity) - np.array(std_parity),
                    np.array(mean_parity) + np.array(std_parity), color=COLOR_PARITY, alpha=0.15)
    ax.plot(ks_sorted, mean_prob, color=COLOR_PROB, linewidth=2.0, label="prob MSE (fixed baseline)")
    ax.fill_between(ks_sorted, np.array(mean_prob) - np.array(std_prob),
                    np.array(mean_prob) + np.array(std_prob), color=COLOR_PROB, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (parity features; prob fixed over K)")
    ax.set_ylabel("Q80 (lower is better)")
    ax.set_title(f"n={pair_rows[0]['n']} sigma={pair_rows[0]['sigma']}  train_m={train_m}  L={layers}")
    ax.legend(frameon=False, loc="best")
    fig.savefig(os.path.join(outdir, f"plot_Q80_vs_K_m{train_m}_L{layers}.pdf"))
    plt.close(fig)


def _plot_q80_ratio_vs_k(pair_rows: List[Dict], Ks: List[int], train_m: int, layers: int, outdir: str) -> None:
    data = [r for r in pair_rows if r["train_m"] == train_m and r["layers"] == layers]
    if not data:
        return

    ks_sorted = sorted(Ks)
    mean_ratio, std_ratio = [], []
    for k in ks_sorted:
        vals = [r["Q80_ratio_parity_over_prob"] for r in data if r["K"] == k]
        m, s = _nanmean_std(vals)
        mean_ratio.append(m)
        std_ratio.append(s)

    y_ratio = np.array(mean_ratio, dtype=np.float64)
    if not _has_positive_finite(y_ratio):
        print(f"[Skip] No positive finite Q80 ratios for m={train_m}, L={layers}.")
        return

    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    ax.plot(ks_sorted, mean_ratio, color="#444444", linewidth=2.0)
    ax.fill_between(ks_sorted, np.array(mean_ratio) - np.array(std_ratio),
                    np.array(mean_ratio) + np.array(std_ratio), color="#888888", alpha=0.2)
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (parity features; prob fixed over K)")
    ax.set_ylabel("Q80 ratio (parity / prob)")
    ax.set_title(f"n={pair_rows[0]['n']} sigma={pair_rows[0]['sigma']}  train_m={train_m}  L={layers}")
    fig.savefig(os.path.join(outdir, f"plot_Q80_ratio_vs_K_m{train_m}_L{layers}.pdf"))
    plt.close(fig)

def _recovery_Q_grid(Qmax: int) -> np.ndarray:
    Q = np.unique(np.concatenate([
        np.unique(np.logspace(0, 3.5, 120).astype(int)),
        np.linspace(1000, Qmax, 160).astype(int),
    ]))
    return Q[Q <= Qmax]

def _plot_recovery_curves_all(
    recovery_store: Dict[Tuple[int, int, int, str], List[np.ndarray]],
    Q: np.ndarray,
    Ks: List[int],
    train_m: int,
    layers: int,
    outdir: str,
    include_target: bool,
    target_curve: np.ndarray,
    include_uniform: bool,
    uniform_curve: np.ndarray,
    n: int,
    sigma: float,
) -> None:
    ks_sorted = sorted(Ks)
    linestyles = ["-", "--", ":", "-."][:max(1, len(ks_sorted))]

    fig, ax = plt.subplots(figsize=(4.4, 2.9), constrained_layout=True)

    if include_target:
        ax.plot(Q, target_curve, color="#222222", linewidth=1.8, label="target")
    if include_uniform:
        ax.plot(Q, uniform_curve, color=hv.COLORS["gray"], linewidth=1.6, linestyle="--", label="uniform")

    for k, ls in zip(ks_sorted, linestyles):
        key_parity = (train_m, layers, k, "parity_mse")
        key_prob = (train_m, layers, k, "prob_mse")

        if key_parity in recovery_store:
            ys = recovery_store[key_parity]
            mean_y = np.mean(np.stack(ys, axis=0), axis=0)
            ax.plot(Q, mean_y, color=COLOR_PARITY, linewidth=2.0, linestyle=ls)
        if key_prob in recovery_store:
            ys = recovery_store[key_prob]
            mean_y = np.mean(np.stack(ys, axis=0), axis=0)
            ax.plot(Q, mean_y, color=COLOR_PROB, linewidth=2.0, linestyle=ls)

    ax.set_xscale("log")
    ax.set_xlabel("Q samples")
    ax.set_ylabel("Recovery R(Q)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"n={n} sigma={sigma:g}  m={train_m}  L={layers}")

    # Legend: K colors + loss styles
    k_handles = [Line2D([0], [0], color="#666666", lw=2, linestyle=ls, label=f"K={k}")
                 for k, ls in zip(ks_sorted, linestyles)]
    loss_handles = [
        Line2D([0], [0], color=COLOR_PARITY, lw=2, label="parity MSE"),
        Line2D([0], [0], color=COLOR_PROB, lw=2, label="prob MSE (fixed baseline)"),
    ]
    if include_target:
        loss_handles.append(Line2D([0], [0], color="#222222", lw=1.8, label="target"))
    if include_uniform:
        loss_handles.append(Line2D([0], [0], color=hv.COLORS["gray"], lw=1.6, linestyle="--", label="uniform"))
    ax.legend(handles=k_handles, frameon=False, loc="lower right", fontsize=7)
    ax.add_artist(ax.legend(handles=loss_handles, frameon=False, loc="lower left", fontsize=7))

    fig.savefig(os.path.join(outdir, f"plot_recovery_allK_m{train_m}_L{layers}.pdf"))
    plt.close(fig)

def _plot_tradeoff(
    pair_rows: List[Dict],
    Ks: List[int],
    train_m: int,
    layers: int,
    outdir: str,
    x_metric: str = "Q80",
) -> None:
    data = [r for r in pair_rows if r["train_m"] == train_m and r["layers"] == layers]
    if not data:
        return

    ks_sorted = sorted(Ks)
    markers = ["o", "s", "^", "D", "P", "X"][:max(1, len(ks_sorted))]
    k_to_marker = {k: m for k, m in zip(ks_sorted, markers)}

    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)

    x_vals: List[float] = []
    y_vals: List[float] = []

    for r in data:
        k = int(r["K"])
        mk = k_to_marker[k]
        if x_metric == "Q80":
            x_parity = float(r["Q80_parity"])
            x_prob = float(r["Q80_prob"])
            ax.set_xlabel("Q80 (lower is better)")
        else:
            x_parity = float(r["R_Q1000_parity"])
            x_prob = float(r["R_Q1000_prob"])
            ax.set_xlabel("R(Q=1000) (higher is better)")

        y_parity = float(r["fit_prob_mse_parity"])
        y_prob = float(r["fit_prob_mse_prob"])

        x_vals.extend([x_parity, x_prob])
        y_vals.extend([y_parity, y_prob])
        ax.scatter(x_parity, y_parity, color=COLOR_PARITY, marker=mk, s=28, alpha=0.9)
        ax.scatter(x_prob, y_prob, color=COLOR_PROB, marker=mk, s=28, alpha=0.9)

    x_arr = np.array(x_vals, dtype=np.float64)
    y_arr = np.array(y_vals, dtype=np.float64)
    if x_metric == "Q80":
        if not _has_positive_finite(x_arr):
            print(f"[Skip] No positive finite x-values for tradeoff plot (Q80), m={train_m}, L={layers}.")
            plt.close(fig)
            return
        ax.set_xscale("log")

    if not _has_positive_finite(y_arr):
        print(f"[Skip] No positive finite y-values for tradeoff plot, m={train_m}, L={layers}.")
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_ylabel("Prob-MSE to training dist (lower = better fit)")
    ax.set_title(f"n={data[0]['n']} sigma={data[0]['sigma']}  m={train_m}  L={layers}")

    loss_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_PARITY,
               markeredgecolor=COLOR_PARITY, label="parity MSE"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_PROB,
               markeredgecolor=COLOR_PROB, label="prob MSE (data-fit)"),
    ]
    k_handles = [Line2D([0], [0], marker=mk, color="none", markerfacecolor="#666666",
                        markeredgecolor="#666666", label=f"K={k}")
                 for k, mk in zip(ks_sorted, markers)]
    ax.legend(handles=loss_handles, frameon=False, loc="upper right", fontsize=7)
    ax.add_artist(ax.legend(handles=k_handles, frameon=False, loc="lower left", fontsize=7))

    fig.savefig(os.path.join(outdir, f"plot_tradeoff_{x_metric}_m{train_m}_L{layers}.pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "exp05_discovery_axis"))
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=2.0)

    parser.add_argument("--Ks", type=str, default="128,256,512")
    parser.add_argument("--train-ms", type=str, default="1000,5000")
    parser.add_argument("--layers", type=str, default="1,2")
    parser.add_argument("--seeds", type=str, default="42,43,44")

    parser.add_argument("--holdout-k", type=int, default=20)
    parser.add_argument("--holdout-pool", type=int, default=400)
    parser.add_argument("--good-frac", type=float, default=0.05)
    parser.add_argument("--holdout-m-train", type=int, default=None,
                        help="If set, use this m_train for holdout selection (fixed across sweep).")

    parser.add_argument("--Q80-thr", type=float, default=0.8)
    parser.add_argument("--Q80-search-max", type=int, default=200000)

    parser.add_argument("--iqp-steps", type=int, default=600)
    parser.add_argument("--iqp-lr", type=float, default=0.05)
    parser.add_argument("--iqp-eval-every", type=int, default=50)

    parser.add_argument(
        "--target-family",
        type=str,
        default="paper_even",
        choices=["paper_even", "paper"],
    )

    parser.add_argument("--make-plots", type=int, default=1)
    parser.add_argument("--plot-recovery", type=int, default=1)
    parser.add_argument("--recovery-Qmax", type=int, default=10000)
    parser.add_argument("--recovery-include-target", type=int, default=1)
    parser.add_argument("--recovery-include-uniform", type=int, default=1)
    parser.add_argument("--plot-tradeoff", type=int, default=1)
    parser.add_argument("--tradeoff-x", type=str, default="Q80", choices=["Q80", "R_Q1000"])

    args = parser.parse_args()

    target_family = str(args.target_family).strip().lower()
    if target_family == "paper":
        target_family = "paper_even"
    if target_family != "paper_even":
        raise ValueError("exp05 supports only target-family=paper_even.")
    args.target_family = target_family

    outdir = _ensure_outdir(args.outdir)
    hv.set_style(base=8)

    Ks = _parse_list_ints(args.Ks)
    train_ms = _parse_list_ints(args.train_ms)
    layers_list = _parse_list_ints(args.layers)
    seeds = _parse_list_ints(args.seeds)

    holdout_m_train = args.holdout_m_train if args.holdout_m_train is not None else max(train_ms)

    # Save config
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    bits_table = hv.make_bits_table(args.n)

    # Target distribution (shared)
    p_star, support, scores = hv.build_target_distribution_paper(args.n, args.beta)

    good_mask = hv.topk_mask_by_scores(scores, support, frac=args.good_frac)

    # Holdout masks per seed (fixed across K/train_m/layers)
    holdout_masks: Dict[int, np.ndarray] = {}
    for seed in seeds:
        hmask = hv.select_holdout_smart(
            p_star=p_star,
            good_mask=good_mask,
            bits_table=bits_table,
            m_train=holdout_m_train,
            holdout_k=args.holdout_k,
            pool_size=args.holdout_pool,
            seed=seed + 111,
        )
        holdout_masks[seed] = hmask

        # Save holdout list for each seed
        hv.save_holdout_list(
            hmask,
            bits_table,
            p_star,
            scores,
            outdir,
            name=f"holdout_strings_seed{seed}.txt",
        )

    Q_grid = _recovery_Q_grid(args.recovery_Qmax)

    # Target recovery curve averaged across seeds (for plotting)
    target_curves = [hv.expected_unique_fraction(p_star, holdout_masks[s], Q_grid) for s in seeds]
    target_curve_mean = np.mean(np.stack(target_curves, axis=0), axis=0)
    # Uniform recovery curve averaged across seeds
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    uniform_curves = [hv.expected_unique_fraction(q_unif, holdout_masks[s], Q_grid) for s in seeds]
    uniform_curve_mean = np.mean(np.stack(uniform_curves, axis=0), axis=0)

    results_long: List[Dict] = []
    results_pair: List[Dict] = []
    recovery_store: Dict[Tuple[int, int, int, str], List[np.ndarray]] = {}

    for seed in seeds:
        holdout_mask = holdout_masks[seed]
        H_size = int(np.sum(holdout_mask))
        N = p_star.size
        q_unif = np.ones(N, dtype=np.float64) / N
        qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0

        for train_m in train_ms:
            # Fixed empirical training distribution (same for all K)
            p_train = p_star.copy()
            if H_size > 0:
                p_train[holdout_mask] = 0.0
                p_train = p_train / p_train.sum()
            idxs_train = hv.sample_indices(p_train, train_m, seed=seed + 7)
            emp = hv.empirical_dist(idxs_train, p_star.size)

            for layers in layers_list:
                K_prob_ref = int(min(Ks))
                cfg_prob_common = dict(
                    n=args.n,
                    beta=args.beta,
                    train_m=train_m,
                    holdout_k=args.holdout_k,
                    holdout_pool=args.holdout_pool,
                    seed=seed,
                    good_frac=args.good_frac,
                    sigmas=[float(args.sigma)],
                    Ks=[K_prob_ref],
                    Qmax=10000,
                    Q80_thr=args.Q80_thr,
                    Q80_search_max=args.Q80_search_max,
                    target_family=target_family,
                    adversarial=False,
                    use_iqp=True,
                    use_classical=False,
                    iqp_steps=args.iqp_steps,
                    iqp_lr=args.iqp_lr,
                    iqp_eval_every=args.iqp_eval_every,
                    iqp_layers=layers,
                    outdir=outdir,
                )

                cfg_prob = hv.Config(iqp_loss="prob_mse", **cfg_prob_common)
                art_prob = hv.rerun_single_setting(
                    cfg_prob,
                    p_star,
                    holdout_mask,
                    bits_table,
                    sigma=float(args.sigma),
                    K=K_prob_ref,
                    return_hist=False,
                    iqp_loss="prob_mse",
                )
                q_prob = art_prob["q_iqp"]
                assert q_prob is not None
                metrics_prob = hv.compute_metrics_for_q(
                    q_prob,
                    holdout_mask,
                    qH_unif,
                    H_size,
                    args.Q80_thr,
                    args.Q80_search_max,
                )
                fit_prob_mse_prob = float(np.mean((q_prob - emp) ** 2))
                if bool(args.plot_recovery):
                    y_prob = hv.expected_unique_fraction(q_prob, holdout_mask, Q_grid)

                for K in Ks:
                    cfg_common = dict(
                        n=args.n,
                        beta=args.beta,
                        train_m=train_m,
                        holdout_k=args.holdout_k,
                        holdout_pool=args.holdout_pool,
                        seed=seed,
                        good_frac=args.good_frac,
                        sigmas=[float(args.sigma)],
                        Ks=[int(K)],
                        Qmax=10000,
                        Q80_thr=args.Q80_thr,
                        Q80_search_max=args.Q80_search_max,
                        target_family=target_family,
                        adversarial=False,
                        use_iqp=True,
                        use_classical=False,
                        iqp_steps=args.iqp_steps,
                        iqp_lr=args.iqp_lr,
                        iqp_eval_every=args.iqp_eval_every,
                        iqp_layers=layers,
                        outdir=outdir,
                    )

                    cfg_parity = hv.Config(iqp_loss="parity_mse", **cfg_common)
                    art_parity = hv.rerun_single_setting(
                        cfg_parity,
                        p_star,
                        holdout_mask,
                        bits_table,
                        sigma=float(args.sigma),
                        K=int(K),
                        return_hist=False,
                        iqp_loss="parity_mse",
                    )
                    q_parity = art_parity["q_iqp"]
                    assert q_parity is not None
                    metrics_parity = hv.compute_metrics_for_q(
                        q_parity,
                        holdout_mask,
                        qH_unif,
                        H_size,
                        args.Q80_thr,
                        args.Q80_search_max,
                    )
                    fit_prob_mse_parity = float(np.mean((q_parity - emp) ** 2))

                    for loss_name, metrics in [("parity_mse", metrics_parity), ("prob_mse", metrics_prob)]:
                        results_long.append(
                            {
                                "seed": seed,
                                "n": args.n,
                                "sigma": float(args.sigma),
                                "K": int(K),
                                "train_m": int(train_m),
                                "layers": int(layers),
                                "loss": loss_name,
                                "qH": float(metrics["qH"]),
                                "qH_ratio": float(metrics["qH_ratio"]),
                                "R_Q1000": float(metrics["R_Q1000"]),
                                "R_Q10000": float(metrics["R_Q10000"]),
                                "Q80": float(metrics["Q80"]),
                                "Q80_pred": float(metrics["Q80_pred"]),
                                "Q80_lb": float(metrics["Q80_lb"]),
                                "fit_prob_mse": fit_prob_mse_parity if loss_name == "parity_mse" else fit_prob_mse_prob,
                            }
                        )

                    q80_parity = float(metrics_parity["Q80"])
                    q80_prob = float(metrics_prob["Q80"])
                    results_pair.append(
                        {
                            "seed": seed,
                            "n": args.n,
                            "sigma": float(args.sigma),
                            "K": int(K),
                            "train_m": int(train_m),
                            "layers": int(layers),
                            "qH_parity": float(metrics_parity["qH"]),
                            "qH_prob": float(metrics_prob["qH"]),
                            "qH_ratio_parity": float(metrics_parity["qH_ratio"]),
                            "qH_ratio_prob": float(metrics_prob["qH_ratio"]),
                            "Q80_parity": q80_parity,
                            "Q80_prob": q80_prob,
                            "Q80_ratio_parity_over_prob": _safe_ratio(q80_parity, q80_prob),
                            "R_Q1000_parity": float(metrics_parity["R_Q1000"]),
                            "R_Q1000_prob": float(metrics_prob["R_Q1000"]),
                            "R_Q10000_parity": float(metrics_parity["R_Q10000"]),
                            "R_Q10000_prob": float(metrics_prob["R_Q10000"]),
                            "Q80_pred_parity": float(metrics_parity["Q80_pred"]),
                            "Q80_pred_prob": float(metrics_prob["Q80_pred"]),
                            "fit_prob_mse_parity": fit_prob_mse_parity,
                            "fit_prob_mse_prob": fit_prob_mse_prob,
                        }
                    )

                    if bool(args.plot_recovery):
                        y_parity = hv.expected_unique_fraction(q_parity, holdout_mask, Q_grid)
                        recovery_store.setdefault((train_m, layers, int(K), "parity_mse"), []).append(y_parity)
                        recovery_store.setdefault((train_m, layers, int(K), "prob_mse"), []).append(y_prob)

    # Save CSVs
    _write_csv(os.path.join(outdir, "results_long.csv"), results_long)
    _write_csv(os.path.join(outdir, "results_pair.csv"), results_pair)

    # Plots
    if bool(args.make_plots):
        for train_m in train_ms:
            for layers in layers_list:
                _plot_q80_vs_k(results_pair, Ks, train_m, layers, outdir)
                _plot_q80_ratio_vs_k(results_pair, Ks, train_m, layers, outdir)
                if bool(args.plot_recovery):
                    _plot_recovery_curves_all(
                        recovery_store,
                        Q_grid,
                        Ks,
                        train_m,
                        layers,
                        outdir,
                        include_target=bool(args.recovery_include_target),
                        target_curve=target_curve_mean,
                        include_uniform=bool(args.recovery_include_uniform),
                        uniform_curve=uniform_curve_mean,
                        n=args.n,
                        sigma=args.sigma,
                    )
                if bool(args.plot_tradeoff):
                    _plot_tradeoff(
                        results_pair,
                        Ks,
                        train_m,
                        layers,
                        outdir,
                        x_metric=args.tradeoff_x,
                    )

    print(f"Done. Results in ./{outdir}/")


if __name__ == "__main__":
    main()
