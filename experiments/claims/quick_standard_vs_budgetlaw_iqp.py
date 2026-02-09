#!/usr/bin/env python3
"""Compare standard coverage vs budget-law proxy on one IQP example."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


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


def _budgetlaw_curve_from_qH(qH: float, H_size: int, Q_vals: np.ndarray) -> np.ndarray:
    if H_size <= 0:
        return np.zeros_like(Q_vals, dtype=np.float64)
    mu = float(qH) / float(H_size)
    mu = float(max(0.0, min(1.0 - 1e-15, mu)))
    return 1.0 - np.power(1.0 - mu, Q_vals.astype(np.float64))


def _select_holdout(
    args: argparse.Namespace,
    p_star: np.ndarray,
    support: np.ndarray,
    scores: np.ndarray,
    bits_table: np.ndarray,
    outdir: Path,
) -> np.ndarray:
    good_mask = hv.topk_mask_by_scores(scores, support, frac=args.good_frac)
    candidate_mask = support.astype(bool) if args.holdout_mode == "global" else good_mask
    holdout_mask = hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate_mask,
        bits_table=bits_table,
        m_train=args.train_m,
        holdout_k=args.holdout_k,
        pool_size=args.holdout_pool,
        seed=args.seed + 111,
    )
    holdout_name = "holdout_strings_global.txt" if args.holdout_mode == "global" else "holdout_strings_high_value.txt"
    hv.save_holdout_list(holdout_mask, bits_table, p_star, scores, str(outdir), name=holdout_name)
    return holdout_mask


def _plot(
    out_pdf: Path,
    out_png: Path,
    p_star: np.ndarray,
    q_iqp: np.ndarray,
    holdout_mask: np.ndarray,
    cfg: Dict[str, float],
) -> Dict[str, float]:
    q_vals = _q_grid(int(cfg["Qmax"]))
    y_target = hv.expected_unique_fraction(p_star, holdout_mask, q_vals)
    y_std = hv.expected_unique_fraction(q_iqp, holdout_mask, q_vals)
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, q_vals)

    H_size = int(np.sum(holdout_mask))
    qH = float(q_iqp[holdout_mask].sum()) if H_size > 0 else 0.0
    y_budget = _budgetlaw_curve_from_qH(qH, H_size, q_vals)
    q80_meas = hv.find_Q_threshold(q_iqp, holdout_mask, thr=0.8, Qmax=int(cfg["Q80_search_max"]))
    q80_pred = hv.Q80_prediction_from_qH(qH, H_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=hv.fig_size("full", 2.95))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.84, wspace=0.30)

    # Left panel: standard coverage tool.
    ax1.plot(q_vals, y_target, color=hv.COLORS["target"], lw=2.0, label=r"Target $p^*$")
    ax1.plot(q_vals, y_std, color=hv.COLORS["model"], lw=2.2, label="IQP coverage $R(Q)$")
    ax1.plot(q_vals, y_unif, color=hv.COLORS["gray"], lw=1.5, ls="--", label="Uniform")
    ax1.axhline(1.0, color=hv.COLORS["gray"], ls=":", alpha=0.7)
    ax1.set_xlim(0, int(cfg["Qmax"]))
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlabel(r"$Q$ samples")
    ax1.set_ylabel(r"Recovery / coverage $R(Q)$")
    ax1.set_title("Standard coverage metric", pad=4)
    ax1.legend(loc="lower right", frameon=False)

    # Right panel: our budget-law proxy tool.
    ax2.plot(q_vals, y_std, color=hv.COLORS["target"], lw=2.0, label="Measured coverage $R(Q)$")
    ax2.plot(q_vals, y_budget, color=hv.COLORS["model"], lw=2.1, ls="--", label=r"Budget-law proxy from $q(H)$")
    if np.isfinite(q80_meas):
        ax2.axvline(float(q80_meas), color=hv.COLORS["target"], lw=1.2, ls=":")
    if np.isfinite(q80_pred):
        ax2.axvline(float(q80_pred), color=hv.COLORS["model"], lw=1.2, ls=":")
    ax2.set_xlim(0, int(cfg["Qmax"]))
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel(r"$Q$ samples")
    ax2.set_ylabel(r"Recovery / coverage")
    ax2.set_title("Our budget-law view", pad=4)
    ax2.text(
        0.02,
        0.06,
        f"q(H)={qH:.4f}\nQ80(meas)={q80_meas:.0f}\nQ80(pred)={q80_pred:.0f}",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#dddddd", alpha=0.92),
    )
    ax2.legend(loc="lower right", frameon=False)

    fig.suptitle(
        (
            f"IQP example | holdout={cfg['holdout_mode']} | beta={cfg['beta']} | m={int(cfg['train_m'])} | "
            f"sigma={cfg['sigma']} | K={int(cfg['K'])}"
        ),
        fontsize=9,
        y=0.93,
    )
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    return {
        "qH": qH,
        "H_size": float(H_size),
        "Q80_meas": float(q80_meas),
        "Q80_pred": float(q80_pred),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "quick_standard_vs_budgetlaw_iqp"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--holdout-mode", type=str, default="high_value", choices=["high_value", "global"])
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument(
        "--sigmas-context",
        type=str,
        default="",
        help="Optional comma-list used as cfg.sigmas context (for consistent sigma indexing).",
    )
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-loss", type=str, default="parity_mse", choices=["parity_mse", "prob_mse", "xent"])
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--Qmax", type=int, default=10000)
    ap.add_argument("--Q80-search-max", type=int, default=200000)
    args = ap.parse_args()

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bits_table = hv.make_bits_table(args.n)
    p_star, support, scores = hv.build_target_distribution_paper(args.n, args.beta)
    holdout_mask = _select_holdout(args, p_star, support, scores, bits_table, outdir)

    sigmas_context = _parse_floats(args.sigmas_context) if str(args.sigmas_context).strip() else [float(args.sigma)]
    if float(args.sigma) not in sigmas_context:
        sigmas_context.append(float(args.sigma))

    cfg = hv.Config(
        n=args.n,
        beta=args.beta,
        train_m=args.train_m,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        seed=args.seed,
        good_frac=args.good_frac,
        sigmas=sigmas_context,
        Ks=[args.K],
        Qmax=args.Qmax,
        Q80_thr=0.8,
        Q80_search_max=args.Q80_search_max,
        target_family="paper_even",
        adversarial=False,
        use_iqp=True,
        use_classical=False,
        iqp_steps=args.iqp_steps,
        iqp_lr=args.iqp_lr,
        iqp_eval_every=args.iqp_eval_every,
        iqp_layers=args.layers,
        iqp_loss=args.iqp_loss,
        outdir=str(outdir),
    )

    art = hv.rerun_single_setting(
        cfg=cfg,
        p_star=p_star,
        holdout_mask=holdout_mask,
        bits_table=bits_table,
        sigma=args.sigma,
        K=args.K,
        return_hist=False,
        iqp_loss=args.iqp_loss,
    )
    q_iqp = np.array(art["q_iqp"], dtype=np.float64)

    cfg_dump = {
        "n": float(args.n),
        "beta": float(args.beta),
        "train_m": float(args.train_m),
        "holdout_mode": args.holdout_mode,
        "sigma": float(args.sigma),
        "sigmas_context": ",".join(str(x) for x in sigmas_context),
        "K": float(args.K),
        "Qmax": float(args.Qmax),
        "Q80_search_max": float(args.Q80_search_max),
    }

    plot_pdf = outdir / f"standard_vs_budgetlaw_iqp_{args.holdout_mode}_beta{args.beta:g}.pdf"
    plot_png = outdir / f"standard_vs_budgetlaw_iqp_{args.holdout_mode}_beta{args.beta:g}.png"
    stats = _plot(plot_pdf, plot_png, p_star, q_iqp, holdout_mask, cfg_dump)

    summary = {
        "config": {
            "n": args.n,
            "beta": args.beta,
            "train_m": args.train_m,
            "holdout_mode": args.holdout_mode,
            "holdout_k": args.holdout_k,
            "sigma": args.sigma,
            "sigmas_context": sigmas_context,
            "K": args.K,
            "iqp_loss": args.iqp_loss,
            "iqp_steps": args.iqp_steps,
        },
        "holdout": {
            "size": int(np.sum(holdout_mask)),
            "p_star_holdout": float(p_star[holdout_mask].sum()),
        },
        "metrics": stats,
        "files": {
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {plot_pdf}")
    print(f"[Saved] {plot_png}")
    print(f"[Saved] {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
