#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 9: Fair classical baseline vs IQP-QCBM under two holdout protocols.

Goals
-----
1) Fair baseline claim:
   Compare IQP-QCBM (parity_mse) vs classical Ising control under matched
   supervision, training data, optimizer type, steps, and learning rate.
2) Global holdout claim:
   Extend holdout from high-value states only to the full target support.

Outputs
-------
- raw_rows.csv: one row per (mode, seed, train_m, layers, sigma, K)
- pair_rows.csv: paired IQP-vs-class rows
- holdout_meta.csv: holdout construction diagnostics
- summary.json: aggregated claim metrics
- plots:
  - fig_q80_vs_k_<mode>_m*_L*.pdf
  - fig_qh_vs_k_<mode>_m*_L*.pdf
  - fig_q80ratio_vs_k_<mode>_m*_L*.pdf
  - fig_mode_comparison_boxplots.pdf
  - fig_holdout_protocol_comparison.pdf
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


COLOR_IQP = hv.COLORS["model"]  # red
COLOR_CLASS = hv.COLORS["blue"]  # blue
COLOR_MODE = {
    "high_value": "#D62728",
    "global": "#1F77B4",
}


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_modes(s: str) -> List[str]:
    out: List[str] = []
    for x in s.split(","):
        m = x.strip().lower()
        if not m:
            continue
        if m not in ("high_value", "global"):
            raise ValueError(f"Unsupported holdout mode: {m}")
        out.append(m)
    if not out:
        raise ValueError("No valid holdout modes provided.")
    return out


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return float("nan")
    return float(a / b)


def _nanmean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _holdout_mask_for_mode(
    mode: str,
    p_star: np.ndarray,
    support: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    m_train_for_holdout: int,
    holdout_k: int,
    holdout_pool: int,
    seed: int,
) -> np.ndarray:
    if mode == "high_value":
        candidate_mask = good_mask
    elif mode == "global":
        candidate_mask = support.astype(bool)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate_mask,
        bits_table=bits_table,
        m_train=m_train_for_holdout,
        holdout_k=holdout_k,
        pool_size=holdout_pool,
        seed=seed + 111,
    )


def _q80_win_iqp_over_class(q80_iqp: float, q80_class: float) -> float:
    fi = np.isfinite(q80_iqp)
    fc = np.isfinite(q80_class)
    if fi and fc:
        if q80_iqp < q80_class:
            return 1.0
        if q80_iqp > q80_class:
            return 0.0
        return float("nan")
    if fi and (not fc):
        return 1.0
    if (not fi) and fc:
        return 0.0
    return float("nan")


def _aggregate_summary(pair_rows: List[Dict]) -> Dict[str, object]:
    if not pair_rows:
        return {}

    out: Dict[str, object] = {}
    modes = sorted(set(str(r["holdout_mode"]) for r in pair_rows))
    out["n_pairs_total"] = int(len(pair_rows))

    for mode in modes:
        rows = [r for r in pair_rows if str(r["holdout_mode"]) == mode]
        deltas = np.array([float(r["delta_qH_iqp_minus_class"]) for r in rows], dtype=np.float64)
        ratios = np.array([float(r["Q80_ratio_iqp_over_class"]) for r in rows], dtype=np.float64)
        wins = np.array(
            [_q80_win_iqp_over_class(float(r["Q80_iqp"]), float(r["Q80_class"])) for r in rows],
            dtype=np.float64,
        )
        pH = np.array([float(r["p_star_holdout"]) for r in rows], dtype=np.float64)
        out[mode] = {
            "n_pairs": int(len(rows)),
            "qH_win_frac_iqp_over_class": float(np.mean(deltas > 0)),
            "qH_delta_median": float(np.median(deltas)),
            "qH_delta_mean": float(np.mean(deltas)),
            "Q80_win_frac_iqp_over_class": float(np.nanmean(wins)),
            "Q80_ratio_median_finite": float(np.nanmedian(ratios)),
            "finite_Q80_ratio_n": int(np.sum(np.isfinite(ratios))),
            "median_p_star_holdout": float(np.median(pH)),
        }

    if "high_value" in out and "global" in out:
        out["mode_delta"] = {
            "delta_qH_win_frac_global_minus_high_value": float(
                out["global"]["qH_win_frac_iqp_over_class"] - out["high_value"]["qH_win_frac_iqp_over_class"]
            ),
            "delta_Q80_win_frac_global_minus_high_value": float(
                out["global"]["Q80_win_frac_iqp_over_class"] - out["high_value"]["Q80_win_frac_iqp_over_class"]
            ),
            "delta_median_p_star_holdout_global_minus_high_value": float(
                out["global"]["median_p_star_holdout"] - out["high_value"]["median_p_star_holdout"]
            ),
        }
    return out


def _plot_q80_vs_k(pair_rows: List[Dict], Ks: List[int], train_m: int, layers: int, mode: str, outdir: str) -> None:
    rows = [r for r in pair_rows if r["train_m"] == train_m and r["layers"] == layers and r["holdout_mode"] == mode]
    if not rows:
        return
    ks = sorted(Ks)
    m_iqp, s_iqp, m_cls, s_cls = [], [], [], []
    for k in ks:
        vals_iqp = [float(r["Q80_iqp"]) for r in rows if int(r["K"]) == int(k)]
        vals_cls = [float(r["Q80_class"]) for r in rows if int(r["K"]) == int(k)]
        a, b = _nanmean_std(vals_iqp)
        m_iqp.append(a); s_iqp.append(b)
        a, b = _nanmean_std(vals_cls)
        m_cls.append(a); s_cls.append(b)

    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    ax.plot(ks, m_iqp, color=COLOR_IQP, linewidth=2.0, label="IQP-QCBM (parity MSE)")
    ax.fill_between(ks, np.array(m_iqp) - np.array(s_iqp), np.array(m_iqp) + np.array(s_iqp), color=COLOR_IQP, alpha=0.15)
    ax.plot(ks, m_cls, color=COLOR_CLASS, linewidth=2.0, label="Classical control")
    ax.fill_between(ks, np.array(m_cls) - np.array(s_cls), np.array(m_cls) + np.array(s_cls), color=COLOR_CLASS, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (parity features)")
    ax.set_ylabel("Q80 (lower is better)")
    ax.set_title(f"{mode} holdout | m={train_m}, L={layers}")
    ax.legend(frameon=False, loc="best", fontsize=7)
    fig.savefig(os.path.join(outdir, f"fig_q80_vs_k_{mode}_m{train_m}_L{layers}.pdf"))
    plt.close(fig)


def _plot_qh_vs_k(pair_rows: List[Dict], Ks: List[int], train_m: int, layers: int, mode: str, outdir: str) -> None:
    rows = [r for r in pair_rows if r["train_m"] == train_m and r["layers"] == layers and r["holdout_mode"] == mode]
    if not rows:
        return
    ks = sorted(Ks)
    m_iqp, s_iqp, m_cls, s_cls = [], [], [], []
    for k in ks:
        vals_iqp = [float(r["qH_iqp"]) for r in rows if int(r["K"]) == int(k)]
        vals_cls = [float(r["qH_class"]) for r in rows if int(r["K"]) == int(k)]
        a, b = _nanmean_std(vals_iqp)
        m_iqp.append(a); s_iqp.append(b)
        a, b = _nanmean_std(vals_cls)
        m_cls.append(a); s_cls.append(b)

    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    ax.plot(ks, m_iqp, color=COLOR_IQP, linewidth=2.0, label="IQP-QCBM (parity MSE)")
    ax.fill_between(ks, np.array(m_iqp) - np.array(s_iqp), np.array(m_iqp) + np.array(s_iqp), color=COLOR_IQP, alpha=0.15)
    ax.plot(ks, m_cls, color=COLOR_CLASS, linewidth=2.0, label="Classical control")
    ax.fill_between(ks, np.array(m_cls) - np.array(s_cls), np.array(m_cls) + np.array(s_cls), color=COLOR_CLASS, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (parity features)")
    ax.set_ylabel("Holdout mass q(H)")
    ax.set_title(f"{mode} holdout | m={train_m}, L={layers}")
    ax.legend(frameon=False, loc="best", fontsize=7)
    fig.savefig(os.path.join(outdir, f"fig_qh_vs_k_{mode}_m{train_m}_L{layers}.pdf"))
    plt.close(fig)


def _plot_q80_ratio_vs_k(pair_rows: List[Dict], Ks: List[int], train_m: int, layers: int, mode: str, outdir: str) -> None:
    rows = [r for r in pair_rows if r["train_m"] == train_m and r["layers"] == layers and r["holdout_mode"] == mode]
    if not rows:
        return
    ks = sorted(Ks)
    m_ratio, s_ratio = [], []
    for k in ks:
        vals = [float(r["Q80_ratio_iqp_over_class"]) for r in rows if int(r["K"]) == int(k)]
        a, b = _nanmean_std(vals)
        m_ratio.append(a); s_ratio.append(b)

    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    ax.plot(ks, m_ratio, color="#333333", linewidth=2.0)
    ax.fill_between(ks, np.array(m_ratio) - np.array(s_ratio), np.array(m_ratio) + np.array(s_ratio), color="#777777", alpha=0.2)
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (parity features)")
    ax.set_ylabel("Q80 ratio (IQP / Classical)")
    ax.set_title(f"{mode} holdout | m={train_m}, L={layers}")
    fig.savefig(os.path.join(outdir, f"fig_q80ratio_vs_k_{mode}_m{train_m}_L{layers}.pdf"))
    plt.close(fig)


def _plot_mode_comparison_boxplots(pair_rows: List[Dict], outdir: str) -> None:
    modes = ["high_value", "global"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)

    data_a = []
    data_b = []
    labels = []
    for m in modes:
        rows = [r for r in pair_rows if str(r["holdout_mode"]) == m]
        if not rows:
            continue
        data_a.append([float(r["delta_qH_iqp_minus_class"]) for r in rows])
        data_b.append([float(r["Q80_ratio_iqp_over_class"]) for r in rows if np.isfinite(float(r["Q80_ratio_iqp_over_class"]))])
        labels.append(m)

    if data_a:
        axes[0].boxplot(data_a, tick_labels=labels, showfliers=False)
    axes[0].axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Delta q(H): IQP - Classical")
    axes[0].set_title("Mass advantage by protocol")

    if data_b:
        axes[1].boxplot(data_b, tick_labels=labels, showfliers=False)
    axes[1].axhline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Q80 ratio: IQP / Classical")
    axes[1].set_title("Discovery advantage by protocol")

    fig.savefig(os.path.join(outdir, "fig_mode_comparison_boxplots.pdf"))
    plt.close(fig)


def _plot_holdout_protocol_comparison(holdout_meta: List[Dict], outdir: str) -> None:
    if not holdout_meta:
        return
    rows = holdout_meta
    modes = ["high_value", "global"]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
    pH_vals = []
    score_vals = []
    labels = []
    for m in modes:
        sub = [r for r in rows if str(r["holdout_mode"]) == m]
        if not sub:
            continue
        pH_vals.append([float(r["p_star_holdout"]) for r in sub])
        score_vals.append([float(r["holdout_score_mean"]) for r in sub])
        labels.append(m)

    if pH_vals:
        axes[0].boxplot(pH_vals, tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("p*(H)")
    axes[0].set_title("Holdout mass under target")

    if score_vals:
        axes[1].boxplot(score_vals, tick_labels=labels, showfliers=False)
    axes[1].set_ylabel("Mean score on holdout")
    axes[1].set_title("Holdout score profile")

    fig.savefig(os.path.join(outdir, "fig_holdout_protocol_comparison.pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "paper_even_final" / "06_claim_fair_baseline_global_holdout"))
    ap.add_argument("--target-family", type=str, default="paper_even", choices=["paper_even", "paper_nonparity", "paper"])
    ap.add_argument("--holdout-modes", type=str, default="high_value,global")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--sigmas", type=str, default="2.0")
    ap.add_argument("--Ks", type=str, default="128,256,512")
    ap.add_argument("--train-ms", type=str, default="1000,5000")
    ap.add_argument("--layers", type=str, default="1,2")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-m-train", type=int, default=None)
    ap.add_argument("--Q80-thr", type=float, default=0.8)
    ap.add_argument("--Q80-search-max", type=int, default=200000)
    ap.add_argument("--iqp-steps", type=int, default=400)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--mode", type=str, default="run+analyze", choices=["run", "analyze", "run+analyze"])
    args = ap.parse_args()

    target_family = str(args.target_family).strip().lower()
    if target_family == "paper":
        target_family = "paper_even"
    args.target_family = target_family

    outdir = _ensure_outdir(args.outdir)
    hv.set_style(base=8)

    holdout_modes = _parse_modes(args.holdout_modes)
    sigmas = _parse_list_floats(args.sigmas)
    Ks = _parse_list_ints(args.Ks)
    train_ms = _parse_list_ints(args.train_ms)
    layers_list = _parse_list_ints(args.layers)
    seeds = _parse_list_ints(args.seeds)
    holdout_m_train = args.holdout_m_train if args.holdout_m_train is not None else max(train_ms)

    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    raw_rows_path = os.path.join(outdir, "raw_rows.csv")
    pair_rows_path = os.path.join(outdir, "pair_rows.csv")
    holdout_meta_path = os.path.join(outdir, "holdout_meta.csv")

    if args.mode in ("run", "run+analyze"):
        bits_table = hv.make_bits_table(args.n)

        if target_family == "paper_even":
            p_star, support, scores = hv.build_target_distribution_paper(args.n, args.beta)
        elif target_family == "paper_nonparity":
            p_star, support, scores = hv.build_target_distribution_paper_nonparity(args.n, args.beta)
        else:
            raise ValueError(f"Unsupported target family: {target_family}")

        good_mask = hv.topk_mask_by_scores(scores, support, frac=args.good_frac)

        holdout_masks: Dict[Tuple[str, int], np.ndarray] = {}
        holdout_meta: List[Dict] = []
        for mode in holdout_modes:
            for seed in seeds:
                h = _holdout_mask_for_mode(
                    mode=mode,
                    p_star=p_star,
                    support=support,
                    good_mask=good_mask,
                    bits_table=bits_table,
                    m_train_for_holdout=holdout_m_train,
                    holdout_k=args.holdout_k,
                    holdout_pool=args.holdout_pool,
                    seed=seed,
                )
                holdout_masks[(mode, seed)] = h
                h_idx = np.where(h)[0]
                holdout_meta.append(
                    {
                        "holdout_mode": mode,
                        "seed": int(seed),
                        "holdout_size": int(np.sum(h)),
                        "p_star_holdout": float(p_star[h].sum()),
                        "holdout_score_mean": float(np.mean(scores[h_idx])) if h_idx.size else float("nan"),
                        "holdout_score_min": float(np.min(scores[h_idx])) if h_idx.size else float("nan"),
                        "holdout_score_max": float(np.max(scores[h_idx])) if h_idx.size else float("nan"),
                    }
                )
                hv.save_holdout_list(
                    h,
                    bits_table,
                    p_star,
                    scores,
                    outdir,
                    name=f"holdout_{mode}_seed{seed}.txt",
                )

        raw_rows: List[Dict] = []
        pair_rows: List[Dict] = []

        for mode in holdout_modes:
            for seed in seeds:
                holdout_mask = holdout_masks[(mode, seed)]
                N = p_star.size
                H_size = int(np.sum(holdout_mask))
                q_unif = np.ones(N, dtype=np.float64) / N
                qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0

                for train_m in train_ms:
                    p_train = p_star.copy()
                    if H_size > 0:
                        p_train[holdout_mask] = 0.0
                        p_train /= p_train.sum()
                    idxs_train = hv.sample_indices(p_train, train_m, seed=seed + 7)
                    emp = hv.empirical_dist(idxs_train, N)

                    for layers in layers_list:
                        for sigma in sigmas:
                            for K in Ks:
                                cfg = hv.Config(
                                    n=args.n,
                                    beta=args.beta,
                                    train_m=train_m,
                                    holdout_k=args.holdout_k,
                                    holdout_pool=args.holdout_pool,
                                    seed=seed,
                                    good_frac=args.good_frac,
                                    sigmas=[float(sigma)],
                                    Ks=[int(K)],
                                    Qmax=10000,
                                    Q80_thr=args.Q80_thr,
                                    Q80_search_max=args.Q80_search_max,
                                    target_family=target_family,
                                    adversarial=False,
                                    use_iqp=True,
                                    use_classical=True,
                                    iqp_steps=args.iqp_steps,
                                    iqp_lr=args.iqp_lr,
                                    iqp_eval_every=args.iqp_eval_every,
                                    iqp_layers=layers,
                                    iqp_loss="parity_mse",
                                    outdir=outdir,
                                )
                                art = hv.rerun_single_setting(
                                    cfg=cfg,
                                    p_star=p_star,
                                    holdout_mask=holdout_mask,
                                    bits_table=bits_table,
                                    sigma=float(sigma),
                                    K=int(K),
                                    return_hist=False,
                                    iqp_loss="parity_mse",
                                )

                                q_iqp = art["q_iqp"]
                                q_class = art["q_class"]
                                assert q_iqp is not None
                                assert q_class is not None

                                met_iqp = hv.compute_metrics_for_q(
                                    q_iqp, holdout_mask, qH_unif, H_size, args.Q80_thr, args.Q80_search_max
                                )
                                met_class = hv.compute_metrics_for_q(
                                    q_class, holdout_mask, qH_unif, H_size, args.Q80_thr, args.Q80_search_max
                                )
                                fit_prob_iqp = float(np.mean((q_iqp - emp) ** 2))
                                fit_prob_class = float(np.mean((q_class - emp) ** 2))

                                row = {
                                    "holdout_mode": mode,
                                    "seed": int(seed),
                                    "n": int(args.n),
                                    "beta": float(args.beta),
                                    "sigma": float(sigma),
                                    "K": int(K),
                                    "train_m": int(train_m),
                                    "layers": int(layers),
                                    "target_family": target_family,
                                    "holdout_size": int(H_size),
                                    "p_star_holdout": float(p_star[holdout_mask].sum()),
                                    "qH_iqp": float(met_iqp["qH"]),
                                    "qH_ratio_iqp": float(met_iqp["qH_ratio"]),
                                    "Q80_iqp": float(met_iqp["Q80"]),
                                    "Q80_pred_iqp": float(met_iqp["Q80_pred"]),
                                    "R_Q1000_iqp": float(met_iqp["R_Q1000"]),
                                    "R_Q10000_iqp": float(met_iqp["R_Q10000"]),
                                    "fit_prob_mse_iqp": fit_prob_iqp,
                                    "qH_class": float(met_class["qH"]),
                                    "qH_ratio_class": float(met_class["qH_ratio"]),
                                    "Q80_class": float(met_class["Q80"]),
                                    "Q80_pred_class": float(met_class["Q80_pred"]),
                                    "R_Q1000_class": float(met_class["R_Q1000"]),
                                    "R_Q10000_class": float(met_class["R_Q10000"]),
                                    "fit_prob_mse_class": fit_prob_class,
                                }
                                raw_rows.append(row)

                                pair_rows.append(
                                    {
                                        **row,
                                        "delta_qH_iqp_minus_class": float(met_iqp["qH"] - met_class["qH"]),
                                        "delta_qH_ratio_iqp_minus_class": float(met_iqp["qH_ratio"] - met_class["qH_ratio"]),
                                        "Q80_ratio_iqp_over_class": _safe_ratio(float(met_iqp["Q80"]), float(met_class["Q80"])),
                                        "delta_logQ80_iqp_minus_class": (
                                            float(np.log10(met_iqp["Q80"]) - np.log10(met_class["Q80"]))
                                            if np.isfinite(float(met_iqp["Q80"])) and np.isfinite(float(met_class["Q80"]))
                                            and float(met_iqp["Q80"]) > 0 and float(met_class["Q80"]) > 0
                                            else float("nan")
                                        ),
                                    }
                                )
                                print(
                                    f"[Run] mode={mode} seed={seed} m={train_m} L={layers} sigma={sigma:g} K={K} | "
                                    f"Q80 IQP={met_iqp['Q80']:.0f} CL={met_class['Q80']:.0f}"
                                )

        _write_csv(raw_rows_path, raw_rows)
        _write_csv(pair_rows_path, pair_rows)
        _write_csv(holdout_meta_path, holdout_meta)

    # Analyze (from CSVs)
    if not os.path.exists(pair_rows_path):
        raise RuntimeError(f"Missing {pair_rows_path}. Run with --mode run or run+analyze first.")
    pair_rows = []
    with open(pair_rows_path, "r", encoding="utf-8") as f:
        pair_rows = list(csv.DictReader(f))
    pair_rows_num: List[Dict] = []
    for r in pair_rows:
        rr = dict(r)
        for k in (
            "seed", "n", "K", "train_m", "layers", "holdout_size",
            "beta", "sigma", "p_star_holdout",
            "qH_iqp", "qH_ratio_iqp", "Q80_iqp", "Q80_pred_iqp", "R_Q1000_iqp", "R_Q10000_iqp", "fit_prob_mse_iqp",
            "qH_class", "qH_ratio_class", "Q80_class", "Q80_pred_class", "R_Q1000_class", "R_Q10000_class", "fit_prob_mse_class",
            "delta_qH_iqp_minus_class", "delta_qH_ratio_iqp_minus_class", "Q80_ratio_iqp_over_class", "delta_logQ80_iqp_minus_class",
        ):
            if k in rr:
                rr[k] = float(rr[k])
        pair_rows_num.append(rr)

    summary = _aggregate_summary(pair_rows_num)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    for mode in sorted(set(str(r["holdout_mode"]) for r in pair_rows_num)):
        for train_m in sorted(set(int(r["train_m"]) for r in pair_rows_num)):
            for layers in sorted(set(int(r["layers"]) for r in pair_rows_num)):
                _plot_q80_vs_k(pair_rows_num, Ks, train_m, layers, mode, outdir)
                _plot_qh_vs_k(pair_rows_num, Ks, train_m, layers, mode, outdir)
                _plot_q80_ratio_vs_k(pair_rows_num, Ks, train_m, layers, mode, outdir)

    _plot_mode_comparison_boxplots(pair_rows_num, outdir)
    if os.path.exists(holdout_meta_path):
        holdout_meta = []
        with open(holdout_meta_path, "r", encoding="utf-8") as f:
            holdout_meta = list(csv.DictReader(f))
        _plot_holdout_protocol_comparison(holdout_meta, outdir)

    print(f"Done. Results in {outdir}")


if __name__ == "__main__":
    main()

