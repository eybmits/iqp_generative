#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WP2 core loss ablation for Nature-impact package.

Compares IQP losses on identical circuits/budgets:
  - parity_mse
  - mmd (tau selected by inner validation from candidate set)
  - xent (NLL)

Outputs:
  - loss_ablation_metrics_long.csv
  - loss_ablation_mmd_tau_candidates.csv
  - recovery_overlay_<mode>_m<m>.pdf
  - forest_q80_ratio_parity_vs_refs.pdf
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
from experiments.legacy import exp11_beta_sweep_global_holdout as exp11  # noqa: E402


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_modes(s: str) -> List[str]:
    out = []
    for x in s.split(","):
        m = x.strip().lower()
        if not m:
            continue
        if m not in ("global", "high_value"):
            raise ValueError(f"Unsupported mode: {m}")
        out.append(m)
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


def _auc_q(q: np.ndarray, y: np.ndarray, qmax: int = 10000) -> float:
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.trapz(y, q) / float(qmax))


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))))


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _bootstrap_ratio_ci(x: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, x.size, size=x.size)
        means[i] = float(np.mean(x[idx]))
    return float(np.mean(x)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description="Loss ablation: parity vs mmd vs xent")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "39_claim_loss_ablation_nature"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--holdout-modes", type=str, default="global,high_value")
    ap.add_argument("--train-ms", type=str, default="200,1000,5000")
    ap.add_argument("--betas", type=str, default="0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--mmd-taus", type=str, default="1,2,4")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--global-sigma", type=float, default=1.0)
    ap.add_argument("--global-K", type=int, default=512)
    ap.add_argument("--high-sigma", type=float, default=2.0)
    ap.add_argument("--high-K", type=int, default=256)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    modes = _parse_modes(args.holdout_modes)
    train_ms = _parse_ints(args.train_ms)
    betas = _parse_floats(args.betas)
    seeds = _parse_ints(args.seeds)
    mmd_taus = _parse_floats(args.mmd_taus)
    if args.smoke:
        betas = betas[:2]
        seeds = seeds[:1]
        train_ms = train_ms[:1]

    Q = _q_grid(10000)
    long_rows: List[Dict] = []
    tau_rows: List[Dict] = []
    curve_rows: List[Dict] = []

    for holdout_mode in modes:
        for train_m in train_ms:
            sigma = float(args.global_sigma) if holdout_mode == "global" else float(args.high_sigma)
            K = int(args.global_K) if holdout_mode == "global" else int(args.high_K)
            for beta in betas:
                for seed in seeds:
                    bits_table = hv.make_bits_table(int(args.n))
                    p_star, support, scores = hv.build_target_distribution_paper(int(args.n), float(beta))
                    good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))
                    holdout_mask = exp11._build_holdout(
                        holdout_mode=holdout_mode,
                        p_star=p_star,
                        support=support,
                        good_mask=good_mask,
                        bits_table=bits_table,
                        m_train_for_holdout=int(args.holdout_m_train),
                        holdout_k=int(args.holdout_k),
                        holdout_pool=int(args.holdout_pool),
                        seed=int(seed),
                    )

                    N = p_star.size
                    H_size = int(np.sum(holdout_mask))
                    p_train = p_star.copy()
                    p_train[holdout_mask] = 0.0
                    p_train /= max(1e-15, float(np.sum(p_train)))
                    idxs = hv.sample_indices(p_train, int(train_m), seed=int(seed) + 7)

                    rng = np.random.default_rng(int(seed) + 999)
                    perm = rng.permutation(idxs.size)
                    cut = int(np.clip(round((1.0 - float(args.val_frac)) * idxs.size), 1, idxs.size - 1))
                    idx_train = idxs[perm[:cut]]
                    idx_val = idxs[perm[cut:]]
                    emp_train = hv.empirical_dist(idx_train, N)
                    emp_val = hv.empirical_dist(idx_val, N)

                    alphas = hv.sample_alphas(int(args.n), sigma, K, seed=int(seed) + 222)
                    P = hv.build_parity_matrix(alphas, bits_table)
                    z_train = P @ emp_train

                    q_unif = np.ones(N, dtype=np.float64) / N
                    qH_unif = float(np.sum(q_unif[holdout_mask])) if H_size > 0 else 1.0

                    base_seed_init = int(seed) + 10000 + 97 * int(K)
                    common = dict(
                        n=int(args.n),
                        layers=int(args.layers),
                        steps=int(args.steps),
                        lr=float(args.lr),
                        P=P,
                        z_data=z_train,
                        eval_every=int(args.eval_every),
                        return_hist=False,
                        xent_emp=emp_train,
                    )

                    q_parity, _, _ = hv.train_iqp_qcbm(seed_init=base_seed_init, loss_mode="parity_mse", **common)
                    q_xent, _, _ = hv.train_iqp_qcbm(seed_init=base_seed_init, loss_mode="xent", **common)

                    best_tau = None
                    best_tau_val = float("inf")
                    best_q_mmd = None
                    for tau in mmd_taus:
                        q_mmd, _, _ = hv.train_iqp_qcbm(
                            seed_init=base_seed_init,
                            loss_mode="mmd",
                            mmd_tau=float(tau),
                            mmd_kernel="hamming_rbf",
                            **common,
                        )
                        val_score = _tv(emp_val, q_mmd)
                        tau_rows.append(
                            {
                                "holdout_mode": holdout_mode,
                                "m": int(train_m),
                                "beta": float(beta),
                                "seed": int(seed),
                                "sigma": float(sigma),
                                "K": int(K),
                                "tau": float(tau),
                                "val_tv": float(val_score),
                            }
                        )
                        if val_score < best_tau_val:
                            best_tau_val = float(val_score)
                            best_tau = float(tau)
                            best_q_mmd = q_mmd

                    assert best_q_mmd is not None and best_tau is not None

                    model_pack = [
                        ("iqp_parity", "parity_mse", q_parity, np.nan),
                        ("iqp_mmd", "mmd", best_q_mmd, best_tau),
                        ("iqp_xent", "xent", q_xent, np.nan),
                    ]

                    for model, loss, q, tau in model_pack:
                        met = hv.compute_metrics_for_q(
                            q=q,
                            holdout_mask=holdout_mask,
                            qH_unif=qH_unif,
                            H_size=H_size,
                            Q80_thr=float(args.q80_thr),
                            Q80_search_max=int(args.q80_search_max),
                        )
                        y = hv.expected_unique_fraction(q, holdout_mask, Q)
                        auc = _auc_q(Q, y, qmax=10000)
                        long_rows.append(
                            {
                                "seed": int(seed),
                                "holdout_mode": holdout_mode,
                                "beta": float(beta),
                                "m": int(train_m),
                                "sigma": float(sigma),
                                "K": int(K),
                                "model": model,
                                "loss": loss,
                                "Q80": float(met["Q80"]),
                                "AUC_R_0_10000": float(auc),
                                "qH": float(met["qH"]),
                                "qH_ratio": float(met["qH_ratio"]),
                                "fit_tv": float(_tv(p_star, q)),
                                "mmd_tau_selected": float(tau) if np.isfinite(tau) else np.nan,
                            }
                        )
                        for qq, rr in zip(Q.tolist(), y.tolist()):
                            curve_rows.append(
                                {
                                    "holdout_mode": holdout_mode,
                                    "m": int(train_m),
                                    "beta": float(beta),
                                    "seed": int(seed),
                                    "model": model,
                                    "Q": int(qq),
                                    "R": float(rr),
                                }
                            )

                    # target/uniform for overlay means
                    y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
                    y_uni = hv.expected_unique_fraction(q_unif, holdout_mask, Q)
                    for qq, rr in zip(Q.tolist(), y_star.tolist()):
                        curve_rows.append(
                            {
                                "holdout_mode": holdout_mode,
                                "m": int(train_m),
                                "beta": float(beta),
                                "seed": int(seed),
                                "model": "target",
                                "Q": int(qq),
                                "R": float(rr),
                            }
                        )
                    for qq, rr in zip(Q.tolist(), y_uni.tolist()):
                        curve_rows.append(
                            {
                                "holdout_mode": holdout_mode,
                                "m": int(train_m),
                                "beta": float(beta),
                                "seed": int(seed),
                                "model": "uniform",
                                "Q": int(qq),
                                "R": float(rr),
                            }
                        )

                    print(
                        f"[done] mode={holdout_mode} m={train_m} beta={beta:g} seed={seed} "
                        f"Q80(par/mmd/xent)="
                        f"{long_rows[-3]['Q80']:.0f}/{long_rows[-2]['Q80']:.0f}/{long_rows[-1]['Q80']:.0f}"
                    )

    _write_csv(outdir / "loss_ablation_metrics_long.csv", long_rows)
    _write_csv(outdir / "loss_ablation_mmd_tau_candidates.csv", tau_rows)
    _write_csv(outdir / "loss_ablation_recovery_curves_long.csv", curve_rows)

    dfm = pd.DataFrame(long_rows)
    dfc = pd.DataFrame(curve_rows)

    # Recovery overlays per regime
    for holdout_mode in modes:
        for train_m in train_ms:
            sub = dfc[(dfc["holdout_mode"] == holdout_mode) & (dfc["m"] == int(train_m))]
            if sub.empty:
                continue
            mean = sub.groupby(["model", "Q"], as_index=False)["R"].mean()
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for model, color, lw, ls in [
                ("target", "#111111", 2.2, "-"),
                ("iqp_parity", "#C40000", 2.8, "-"),
                ("iqp_mmd", "#1F77B4", 2.1, "-"),
                ("iqp_xent", "#4C78A8", 2.1, "--"),
                ("uniform", "#6E6E6E", 1.7, "--"),
            ]:
                cur = mean[mean["model"] == model].sort_values("Q")
                if cur.empty:
                    continue
                ax.plot(cur["Q"].to_numpy(), cur["R"].to_numpy(), color=color, linewidth=lw, linestyle=ls, label=model)
            ax.set_xlim(0, 10000)
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel(r"$Q$ samples from model")
            ax.set_ylabel(r"Recovery $R(Q)$")
            ax.grid(True, alpha=0.14, linestyle="--")
            ax.legend(
                handles=[
                    plt.Line2D([0], [0], color="#111111", lw=2.2, label="Target $p^*$"),
                    plt.Line2D([0], [0], color="#C40000", lw=2.8, label="IQP Parity"),
                    plt.Line2D([0], [0], color="#1F77B4", lw=2.1, label="IQP MMD"),
                    plt.Line2D([0], [0], color="#4C78A8", lw=2.1, ls="--", label="IQP NLL"),
                    plt.Line2D([0], [0], color="#6E6E6E", lw=1.7, ls="--", label="Uniform"),
                ],
                loc="lower right",
                frameon=False,
                fontsize=8,
            )
            fig.savefig(outdir / f"recovery_overlay_{holdout_mode}_m{train_m}.pdf")
            fig.savefig(outdir / f"recovery_overlay_{holdout_mode}_m{train_m}.png", dpi=280)
            plt.close(fig)

    # Forest plot: Q80 ratios parity/reference with bootstrap CI
    rows_forest = []
    for holdout_mode in modes:
        for train_m in train_ms:
            for ref in ("mmd", "xent"):
                a = dfm[
                    (dfm["holdout_mode"] == holdout_mode)
                    & (dfm["m"] == int(train_m))
                    & (dfm["loss"] == "parity_mse")
                ][["seed", "beta", "Q80"]].rename(columns={"Q80": "Q80_parity"})
                b = dfm[
                    (dfm["holdout_mode"] == holdout_mode)
                    & (dfm["m"] == int(train_m))
                    & (dfm["loss"] == ref)
                ][["seed", "beta", "Q80"]].rename(columns={"Q80": "Q80_ref"})
                merged = pd.merge(a, b, on=["seed", "beta"], how="inner")
                if merged.empty:
                    continue
                ratio = merged["Q80_parity"].to_numpy(np.float64) / np.maximum(merged["Q80_ref"].to_numpy(np.float64), 1e-12)
                mean, lo, hi = _bootstrap_ratio_ci(ratio, n_boot=int(args.n_boot), seed=123 + int(train_m))
                rows_forest.append(
                    {
                        "holdout_mode": holdout_mode,
                        "m": int(train_m),
                        "comparison": f"parity/{ref}",
                        "n_pairs": int(merged.shape[0]),
                        "ratio_mean": float(mean),
                        "ratio_ci_lo": float(lo),
                        "ratio_ci_hi": float(hi),
                    }
                )
    _write_csv(outdir / "forest_q80_ratio_rows.csv", rows_forest)
    dff = pd.DataFrame(rows_forest)
    if not dff.empty:
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        dff = dff.sort_values(["holdout_mode", "m", "comparison"]).reset_index(drop=True)
        y = np.arange(dff.shape[0])
        x = dff["ratio_mean"].to_numpy(np.float64)
        lo = x - dff["ratio_ci_lo"].to_numpy(np.float64)
        hi = dff["ratio_ci_hi"].to_numpy(np.float64) - x
        colors = ["#C40000" if "mmd" in c else "#4C78A8" for c in dff["comparison"].tolist()]
        ax.errorbar(x, y, xerr=np.vstack([lo, hi]), fmt="o", color="#222222", ecolor="#222222", capsize=3, lw=1.2)
        ax.scatter(x, y, c=colors, s=28, zorder=3)
        labels = [f"{r.holdout_mode}, m={int(r.m)}, {r.comparison}" for _, r in dff.iterrows()]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(1.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Q80 ratio (Parity / Reference)  [<1 is better for Parity]")
        ax.grid(True, axis="x", alpha=0.14, linestyle="--")
        fig.savefig(outdir / "forest_q80_ratio_parity_vs_refs.pdf")
        fig.savefig(outdir / "forest_q80_ratio_parity_vs_refs.png", dpi=280)
        plt.close(fig)

    print(f"[saved] {outdir / 'loss_ablation_metrics_long.csv'}")
    print(f"[saved] {outdir / 'loss_ablation_mmd_tau_candidates.csv'}")
    print(f"[saved] {outdir / 'forest_q80_ratio_parity_vs_refs.pdf'}")


if __name__ == "__main__":
    main()
