#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WP3 mechanistic intervention:
  - Construct low/mid/high visibility holdouts with matched holdout mass and size.
  - Measure discovery metrics per visibility quantile.
  - Quantify relation visibility -> Q80 and simplex fill-in -> q(H) gain.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _auc_q(q: np.ndarray, y: np.ndarray, qmax: int = 10000) -> float:
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.trapz(y, q) / float(qmax))


def _rankdata(x: np.ndarray) -> np.ndarray:
    idx = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(idx, dtype=np.float64)
    ranks[idx] = np.arange(1, x.size + 1, dtype=np.float64)
    return ranks


def _spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan"), float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    c = 0
    for _ in range(int(n_perm)):
        rp = rng.permutation(ry)
        rr = float(np.corrcoef(rx, rp)[0, 1])
        if abs(rr) >= abs(r) - 1e-15:
            c += 1
    p = float((c + 1) / (n_perm + 1))
    return r, p


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _best_mass_match_subset(
    rng: np.random.Generator,
    cand_idx: np.ndarray,
    p_star: np.ndarray,
    holdout_k: int,
    target_mass: float,
    n_trials: int = 2000,
) -> np.ndarray:
    best = None
    best_err = float("inf")
    if cand_idx.size <= holdout_k:
        return cand_idx.copy()
    for _ in range(int(n_trials)):
        pick = rng.choice(cand_idx, size=int(holdout_k), replace=False)
        m = float(np.sum(p_star[pick]))
        err = abs(m - target_mass)
        if err < best_err:
            best_err = err
            best = pick.copy()
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Visibility-causal intervention analysis.")
    ap.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "paper_even_final" / "40_claim_visibility_causal"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")

    seeds = _parse_ints(args.seeds)
    if args.smoke:
        seeds = seeds[:1]

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for seed in seeds:
        bits_table = hv.make_bits_table(int(args.n))
        p_star, support, scores = hv.build_target_distribution_paper(int(args.n), float(args.beta))
        good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))
        candidate = support if args.holdout_mode == "global" else good_mask
        cand_idx = np.where(candidate)[0]

        # Build visibility proxy from reference full moments.
        alphas = hv.sample_alphas(int(args.n), float(args.sigma), int(args.K), seed=int(seed) + 222)
        P = hv.build_parity_matrix(alphas, bits_table)
        z_ref = P @ p_star
        q_lin_ref = hv.linear_band_reconstruction(P, z_ref, int(args.n))
        vis = q_lin_ref[cand_idx]

        q1, q2 = np.quantile(vis, [1 / 3, 2 / 3])
        low_idx = cand_idx[vis <= q1]
        mid_idx = cand_idx[(vis > q1) & (vis <= q2)]
        high_idx = cand_idx[vis > q2]

        # common target holdout mass via full-candidate median probability
        target_mass = float(np.median(p_star[cand_idx]) * int(args.holdout_k))
        rng = np.random.default_rng(int(seed) + 901)
        picks = {
            "low": _best_mass_match_subset(rng, low_idx, p_star, int(args.holdout_k), target_mass),
            "mid": _best_mass_match_subset(rng, mid_idx, p_star, int(args.holdout_k), target_mass),
            "high": _best_mass_match_subset(rng, high_idx, p_star, int(args.holdout_k), target_mass),
        }

        for qname, pick in picks.items():
            holdout_mask = np.zeros(p_star.size, dtype=bool)
            holdout_mask[pick] = True

            cfg = hv.Config(
                n=int(args.n),
                beta=float(args.beta),
                train_m=int(args.train_m),
                holdout_k=int(args.holdout_k),
                holdout_pool=400,
                seed=int(seed),
                good_frac=float(args.good_frac),
                sigmas=[float(args.sigma)],
                Ks=[int(args.K)],
                Qmax=10000,
                Q80_thr=float(args.q80_thr),
                Q80_search_max=int(args.q80_search_max),
                target_family="paper_even",
                adversarial=False,
                use_iqp=True,
                use_classical=False,
                iqp_steps=int(args.steps),
                iqp_lr=float(args.lr),
                iqp_eval_every=int(args.eval_every),
                iqp_layers=int(args.layers),
                iqp_loss="parity_mse",
                outdir=str(outdir),
            )
            art = hv.rerun_single_setting(
                cfg=cfg,
                p_star=p_star,
                holdout_mask=holdout_mask,
                bits_table=bits_table,
                sigma=float(args.sigma),
                K=int(args.K),
                return_hist=False,
                iqp_loss="parity_mse",
            )
            q_iqp = art["q_iqp"]
            q_spec = art["q_spec"]
            q_lin = art["q_lin"]
            assert isinstance(q_iqp, np.ndarray)
            assert isinstance(q_spec, np.ndarray)
            assert isinstance(q_lin, np.ndarray)

            H_size = int(np.sum(holdout_mask))
            q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
            qH_unif = float(np.sum(q_unif[holdout_mask]))
            met = hv.compute_metrics_for_q(
                q=q_iqp,
                holdout_mask=holdout_mask,
                qH_unif=qH_unif,
                H_size=H_size,
                Q80_thr=float(args.q80_thr),
                Q80_search_max=int(args.q80_search_max),
            )
            qgrid = np.unique(np.concatenate([np.arange(0, 1001, 20), np.linspace(1000, 10000, 181).astype(int)]))
            y_iqp = hv.expected_unique_fraction(q_iqp, holdout_mask, qgrid)
            auc = _auc_q(qgrid, y_iqp, qmax=10000)
            q_lin_pos = np.clip(q_lin, 0.0, None)
            fillin = float(np.sum(q_spec[holdout_mask]) - np.sum(q_lin_pos[holdout_mask]))
            qh_gain = float(np.sum(q_iqp[holdout_mask]) - np.sum(q_unif[holdout_mask]))

            rows.append(
                {
                    "seed": int(seed),
                    "holdout_mode": str(args.holdout_mode),
                    "beta": float(args.beta),
                    "m": int(args.train_m),
                    "sigma": float(args.sigma),
                    "K": int(args.K),
                    "model": "iqp_parity",
                    "loss": "parity_mse",
                    "holdout_quantile": qname,
                    "visibility_mean": float(np.mean(q_lin_ref[pick])),
                    "visibility_median": float(np.median(q_lin_ref[pick])),
                    "p_star_holdout": float(np.sum(p_star[holdout_mask])),
                    "Q80": float(met["Q80"]),
                    "AUC_R_0_10000": float(auc),
                    "qH": float(met["qH"]),
                    "qH_ratio": float(met["qH_ratio"]),
                    "R_Q1000": float(met["R_Q1000"]),
                    "R_Q10000": float(met["R_Q10000"]),
                    "fit_tv": float(0.5 * np.sum(np.abs(np.asarray(p_star, dtype=np.float64) - np.asarray(q_iqp, dtype=np.float64)))),
                    "simplex_fillin_holdout": float(fillin),
                    "qh_gain_over_uniform": float(qh_gain),
                }
            )
            print(
                f"[done] seed={seed} q={qname} vis_med={np.median(q_lin_ref[pick]):.3e} "
                f"Q80={met['Q80']:.0f} qH_ratio={met['qH_ratio']:.2f}"
            )

    _write_csv(outdir / "visibility_causal_metrics.csv", rows)
    df = pd.DataFrame(rows)
    if df.empty:
        print("[warn] no rows generated")
        return

    # Trend tests
    q_to_num = {"low": 0.0, "mid": 1.0, "high": 2.0}
    qnum = np.array([q_to_num.get(str(v), np.nan) for v in df["holdout_quantile"].tolist()], dtype=np.float64)
    vis = df["visibility_median"].to_numpy(np.float64)
    q80 = df["Q80"].to_numpy(np.float64)
    fill = df["simplex_fillin_holdout"].to_numpy(np.float64)
    qgain = df["qh_gain_over_uniform"].to_numpy(np.float64)

    rho_vis_q80, p_vis_q80 = _spearman_perm(vis, -q80, int(args.n_perm), seed=101)
    rho_quant_q80, p_quant_q80 = _spearman_perm(qnum, -q80, int(args.n_perm), seed=102)
    rho_fill_qgain, p_fill_qgain = _spearman_perm(fill, qgain, int(args.n_perm), seed=103)

    (outdir / "visibility_causal_stats.txt").write_text(
        "\n".join(
            [
                f"spearman(visibility,-Q80)={rho_vis_q80:.6f}, p={p_vis_q80:.6g}",
                f"spearman(quantile,-Q80)={rho_quant_q80:.6f}, p={p_quant_q80:.6g}",
                f"spearman(fillin,qh_gain)={rho_fill_qgain:.6f}, p={p_fill_qgain:.6g}",
            ]
        ),
        encoding="utf-8",
    )

    # Plot: quantile vs Q80 and qH_ratio
    g = df.groupby("holdout_quantile", as_index=False).agg(
        Q80_mean=("Q80", "mean"),
        Q80_std=("Q80", "std"),
        qH_ratio_mean=("qH_ratio", "mean"),
        qH_ratio_std=("qH_ratio", "std"),
    )
    order = ["low", "mid", "high"]
    g["rank"] = g["holdout_quantile"].map({k: i for i, k in enumerate(order)})
    g = g.sort_values("rank")
    x = np.arange(g.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    axes[0].errorbar(x, g["Q80_mean"], yerr=g["Q80_std"].fillna(0.0), fmt="o-", color="#C40000", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(g["holdout_quantile"].tolist())
    axes[0].set_ylabel("Q80")
    axes[0].set_title("Visibility quantile vs Q80")
    axes[0].grid(True, alpha=0.14, linestyle="--")

    axes[1].errorbar(x, g["qH_ratio_mean"], yerr=g["qH_ratio_std"].fillna(0.0), fmt="o-", color="#1F77B4", capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(g["holdout_quantile"].tolist())
    axes[1].set_ylabel("q(H) / q_unif(H)")
    axes[1].set_title("Visibility quantile vs qH ratio")
    axes[1].grid(True, alpha=0.14, linestyle="--")

    fig.savefig(outdir / "visibility_quantile_effects.pdf")
    fig.savefig(outdir / "visibility_quantile_effects.png", dpi=280)
    plt.close(fig)

    # Plot: fill-in vs qh_gain
    fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)
    ax.scatter(fill, qgain, color="#222222", s=18, alpha=0.9)
    ax.set_xlabel("Simplex fill-in on holdout")
    ax.set_ylabel("q(H) gain over uniform")
    ax.grid(True, alpha=0.14, linestyle="--")
    fig.savefig(outdir / "simplex_fillin_vs_qh_gain.pdf")
    fig.savefig(outdir / "simplex_fillin_vs_qh_gain.png", dpi=280)
    plt.close(fig)

    print(f"[saved] {outdir / 'visibility_causal_metrics.csv'}")
    print(f"[saved] {outdir / 'visibility_causal_stats.txt'}")


if __name__ == "__main__":
    main()
