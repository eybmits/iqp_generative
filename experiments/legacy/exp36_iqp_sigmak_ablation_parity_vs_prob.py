#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 36: IQP sigma-K ablation (parity-MSE vs reference loss) for paper_even.

Primary target use-case from current paper thread:
  - n=12, beta=0.8, train_m=200, global holdout
  - compare IQP training losses over sigma x K:
      * parity_mse  (red shades)
      * reference loss (xent/mmd/prob_mse) in blue

Outputs:
  <outdir>/sigmak_ablation_metrics.csv
  <outdir>/sigmak_ablation_recovery_parity_vs_<reference>.pdf
  <outdir>/sigmak_ablation_recovery_parity_vs_<reference>.png
  (compatibility copy also saved to ...parity_vs_prob.*)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402
from experiments.legacy import exp11_beta_sweep_global_holdout as exp11  # noqa: E402


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fit_tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def _q_grid(Qmax: int = 10000) -> np.ndarray:
    Q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 120).astype(int)),
                np.linspace(1000, Qmax, 160).astype(int),
                np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100], dtype=int),
            ]
        )
    )
    return Q[(Q >= 0) & (Q <= Qmax)]


def _pure_red_shades(n: int) -> List[tuple]:
    """Light-to-mid red shades without orange tint."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.92, 0.62, 0.64, 1.0)]
    c0 = np.array([0.97, 0.85, 0.86])  # very light red
    c1 = np.array([0.84, 0.30, 0.34])  # medium red
    out: List[tuple] = []
    for i in range(n):
        t = i / float(n - 1)
        c = (1.0 - t) * c0 + t * c1
        out.append((float(c[0]), float(c[1]), float(c[2]), 1.0))
    return out


def _ref_label(loss_name: str) -> str:
    name = str(loss_name).lower()
    if name == "xent":
        return "IQP NLL/Cross-Entropy"
    if name == "mmd":
        return "IQP MMD"
    if name == "prob_mse":
        return "IQP MSE"
    return f"IQP {name}"


def run() -> None:
    ap = argparse.ArgumentParser(description="IQP sigma-K ablation (parity vs reference loss).")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "paper_even_final"
            / "36_claim_iqp_sigmak_ablation_parity_vs_prob_m200_beta08"
        ),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--sigmas", type=str, default="0.5,1.0,2.0,3.0")
    ap.add_argument("--Ks", type=str, default="128,256,512")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--iqp-steps", type=int, default=300)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--reference-loss", type=str, default="mmd", choices=["xent", "mmd", "prob_mse"])
    ap.add_argument("--reference-mmd-tau", type=float, default=2.0)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    args = ap.parse_args()

    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")

    hv.set_style(base=8)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sigmas = _parse_list_floats(args.sigmas)
    Ks = _parse_list_ints(args.Ks)

    bits_table = hv.make_bits_table(args.n)
    p_star, support, scores = hv.build_target_distribution_paper(args.n, args.beta)
    good_mask = hv.topk_mask_by_scores(scores, support, frac=args.good_frac)
    holdout_mask = exp11._build_holdout(
        holdout_mode=args.holdout_mode,
        p_star=p_star,
        support=support,
        good_mask=good_mask,
        bits_table=bits_table,
        m_train_for_holdout=args.holdout_m_train,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        seed=args.seed,
    )

    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    qH_unif = float(np.sum(q_unif[holdout_mask]))
    H_size = int(np.sum(holdout_mask))

    rows: List[Dict] = []
    parity_runs: List[Tuple[float, int, np.ndarray, float]] = []  # sigma, K, q, Q80
    q_xent_ref: np.ndarray | None = None
    for sigma in sigmas:
        for K in Ks:
            base_cfg = hv.Config(
                n=args.n,
                beta=args.beta,
                train_m=args.train_m,
                holdout_k=args.holdout_k,
                holdout_pool=args.holdout_pool,
                seed=args.seed,
                good_frac=args.good_frac,
                sigmas=[float(sigma)],
                Ks=[int(K)],
                Qmax=10000,
                Q80_thr=args.q80_thr,
                Q80_search_max=args.q80_search_max,
                target_family="paper_even",
                adversarial=False,
                use_iqp=True,
                use_classical=False,
                iqp_steps=args.iqp_steps,
                iqp_lr=args.iqp_lr,
                iqp_eval_every=args.iqp_eval_every,
                iqp_layers=args.layers,
                iqp_loss="parity_mse",
                iqp_mmd_tau=args.reference_mmd_tau,
                outdir=str(outdir),
            )

            loss_mode = "parity_mse"
            print(f"[run] beta={args.beta:g} m={args.train_m} sigma={sigma:g} K={K} loss={loss_mode}")
            art = hv.rerun_single_setting(
                cfg=base_cfg,
                p_star=p_star,
                holdout_mask=holdout_mask,
                bits_table=bits_table,
                sigma=float(sigma),
                K=int(K),
                return_hist=False,
                iqp_loss=loss_mode,
            )
            q = art["q_iqp"]
            assert isinstance(q, np.ndarray)
            m = hv.compute_metrics_for_q(
                q=q,
                holdout_mask=holdout_mask,
                qH_unif=qH_unif,
                H_size=H_size,
                Q80_thr=args.q80_thr,
                Q80_search_max=args.q80_search_max,
            )
            rows.append(
                dict(
                    holdout_mode=args.holdout_mode,
                    n=args.n,
                    beta=float(args.beta),
                    train_m=int(args.train_m),
                    seed=int(args.seed),
                    sigma=float(sigma),
                    K=int(K),
                    loss=loss_mode,
                    qH=float(m["qH"]),
                    qH_ratio=float(m["qH_ratio"]),
                    R_Q1000=float(m["R_Q1000"]),
                    R_Q10000=float(m["R_Q10000"]),
                    Q80=float(m["Q80"]),
                    Q80_pred=float(m["Q80_pred"]),
                    fit_tv_to_pstar=_fit_tv(p_star, q),
                )
            )
            parity_runs.append((float(sigma), int(K), q.copy(), float(m["Q80"])))

    # Single reference-loss run (objective does not use parity feature family sigma/K).
    sigma_ref = float(sigmas[0])
    K_ref = int(Ks[0])
    ref_cfg = hv.Config(
        n=args.n,
        beta=args.beta,
        train_m=args.train_m,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        seed=args.seed,
        good_frac=args.good_frac,
        sigmas=[sigma_ref],
        Ks=[K_ref],
        Qmax=10000,
        Q80_thr=args.q80_thr,
        Q80_search_max=args.q80_search_max,
        target_family="paper_even",
        adversarial=False,
        use_iqp=True,
        use_classical=False,
        iqp_steps=args.iqp_steps,
        iqp_lr=args.iqp_lr,
        iqp_eval_every=args.iqp_eval_every,
        iqp_layers=args.layers,
        iqp_loss=str(args.reference_loss),
        iqp_mmd_tau=args.reference_mmd_tau,
        outdir=str(outdir),
    )
    print(f"[run] beta={args.beta:g} m={args.train_m} {args.reference_loss} reference")
    art_xent = hv.rerun_single_setting(
        cfg=ref_cfg,
        p_star=p_star,
        holdout_mask=holdout_mask,
        bits_table=bits_table,
        sigma=sigma_ref,
        K=K_ref,
        return_hist=False,
        iqp_loss=str(args.reference_loss),
    )
    q_xent = art_xent["q_iqp"]
    assert isinstance(q_xent, np.ndarray)
    q_xent_ref = q_xent.copy()
    m_xent = hv.compute_metrics_for_q(
        q=q_xent,
        holdout_mask=holdout_mask,
        qH_unif=qH_unif,
        H_size=H_size,
        Q80_thr=args.q80_thr,
        Q80_search_max=args.q80_search_max,
    )
    rows.append(
        dict(
            holdout_mode=args.holdout_mode,
            n=args.n,
            beta=float(args.beta),
            train_m=int(args.train_m),
            seed=int(args.seed),
            sigma=float(sigma_ref),
            K=int(K_ref),
            loss=str(args.reference_loss),
            qH=float(m_xent["qH"]),
            qH_ratio=float(m_xent["qH_ratio"]),
            R_Q1000=float(m_xent["R_Q1000"]),
            R_Q10000=float(m_xent["R_Q10000"]),
            Q80=float(m_xent["Q80"]),
            Q80_pred=float(m_xent["Q80_pred"]),
            fit_tv_to_pstar=_fit_tv(p_star, q_xent),
        )
    )

    csv_path = outdir / "sigmak_ablation_metrics.csv"
    _write_csv(csv_path, rows)
    print(f"[saved] {csv_path}")

    if not parity_runs:
        raise RuntimeError("No parity runs available.")
    if q_xent_ref is None:
        raise RuntimeError("No reference run available.")

    # --- Recovery plot in requested style ---
    Q = _q_grid(10000)
    y_target = hv.expected_unique_fraction(p_star, holdout_mask, Q)
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, Q)
    y_xent = hv.expected_unique_fraction(q_xent_ref, holdout_mask, Q)

    # Best parity = smallest finite Q80.
    finite = [r for r in parity_runs if np.isfinite(r[3])]
    if finite:
        best_sigma, best_K, q_best, best_q80 = sorted(finite, key=lambda t: t[3])[0]
    else:
        best_sigma, best_K, q_best, best_q80 = parity_runs[0]
    y_best = hv.expected_unique_fraction(q_best, holdout_mask, Q)

    other = [(s, k, q, q80) for (s, k, q, q80) in parity_runs if not (float(s) == float(best_sigma) and int(k) == int(best_K))]
    other_sorted = sorted(other, key=lambda t: (np.inf if not np.isfinite(t[3]) else t[3], t[0], t[1]))

    # Nature-like single-panel geometry.
    fig, ax = plt.subplots(figsize=(5.1, 3.35))

    # Target + uniform
    ax.plot(Q, y_target, color="#111111", linewidth=2.35, zorder=30)
    ax.plot(Q, y_unif, color="#6E6E6E", linewidth=1.8, linestyle="--", zorder=5)

    # Other parity settings in light red shades.
    n_other = max(1, len(other_sorted))
    reds = _pure_red_shades(n_other)
    for i, (_s, _k, q, _q80) in enumerate(other_sorted):
        y = hv.expected_unique_fraction(q, holdout_mask, Q)
        ax.plot(Q, y, color=reds[i], linewidth=1.5, alpha=0.98, zorder=10)

    # Best parity in strong red.
    ax.plot(Q, y_best, color="#C40000", linewidth=3.0, zorder=40)

    # NLL/Cross-Entropy in blue.
    ax.plot(Q, y_xent, color="#1F77B4", linewidth=2.2, zorder=20)

    ax.set_xlim(0, 10000)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.grid(True, alpha=0.16, linestyle="--")

    handles = [
        Line2D([0], [0], color="#111111", lw=2.35, label=r"Target $p^*$"),
        Line2D([0], [0], color="#C40000", lw=3.0, label=fr"IQP Parity (best), $\sigma={best_sigma:g}, K={best_K}$"),
        Line2D([0], [0], color=_pure_red_shades(1)[0], lw=1.7, label=fr"IQP Parity (other settings, n={len(other_sorted)})"),
        Line2D([0], [0], color="#1F77B4", lw=2.2, label=_ref_label(args.reference_loss)),
        Line2D([0], [0], color="#6E6E6E", lw=1.8, ls="--", label="Uniform"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.1)

    ref_tag = str(args.reference_loss).lower()
    pdf_path = outdir / f"sigmak_ablation_recovery_parity_vs_{ref_tag}.pdf"
    png_path = outdir / f"sigmak_ablation_recovery_parity_vs_{ref_tag}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=280)
    # Compatibility: keep old filename updated, since it's referenced in the thread.
    fig.savefig(outdir / "sigmak_ablation_recovery_parity_vs_prob.pdf")
    fig.savefig(outdir / "sigmak_ablation_recovery_parity_vs_prob.png", dpi=280)
    plt.close(fig)
    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    run()
