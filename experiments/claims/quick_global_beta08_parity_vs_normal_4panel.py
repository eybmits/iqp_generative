#!/usr/bin/env python3
"""Short experiment: global holdout, beta=0.8, IQP parity vs normal loss."""

from __future__ import annotations

import argparse
import json
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


def _q_grid(Qmax: int = 10000) -> np.ndarray:
    q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 140).astype(int)),
                np.linspace(1000, Qmax, 180).astype(int),
                np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100], dtype=int),
            ]
        )
    )
    return q[(q >= 0) & (q <= Qmax)]


def _score_profile(q: np.ndarray, scores: np.ndarray, support: np.ndarray, levels: List[int]) -> Tuple[np.ndarray, float]:
    out = []
    s_int = scores.astype(int)
    for lv in levels:
        mask = support & (s_int == int(lv))
        out.append(float(q[mask].sum()))
    off_support = float(q[~support].sum())
    return np.array(out, dtype=np.float64), off_support


def _lorenz_curve(shares: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (population_fraction, cumulative_mass) for sorted shares."""
    v = np.array(shares, dtype=np.float64)
    if v.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    v = np.sort(np.clip(v, 0.0, None))
    s = float(v.sum())
    if s <= 0.0:
        cum = np.linspace(0.0, 1.0, v.size + 1)
        pop = np.linspace(0.0, 1.0, v.size + 1)
        return pop, cum
    v = v / s
    cum = np.concatenate(([0.0], np.cumsum(v)))
    pop = np.linspace(0.0, 1.0, v.size + 1)
    return pop, cum


def _gini_from_shares(shares: np.ndarray) -> float:
    pop, cum = _lorenz_curve(shares)
    area = float(np.trapezoid(cum, pop))
    return float(max(0.0, min(1.0, 1.0 - 2.0 * area)))


def _plot(
    outpath_pdf: Path,
    outpath_png: Path,
    p_star: np.ndarray,
    support: np.ndarray,
    scores: np.ndarray,
    holdout_mask: np.ndarray,
    q_parity: np.ndarray,
    q_normal: np.ndarray,
    metrics_parity: Dict[str, float],
    metrics_normal: Dict[str, float],
    normal_loss: str,
    cfg: Dict[str, float],
) -> None:
    q_vals = _q_grid(int(cfg["Qmax"]))
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size

    y_star = hv.expected_unique_fraction(p_star, holdout_mask, q_vals)
    y_parity = hv.expected_unique_fraction(q_parity, holdout_mask, q_vals)
    y_normal = hv.expected_unique_fraction(q_normal, holdout_mask, q_vals)
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, q_vals)

    levels = sorted({int(v) for v in scores[support]})
    p_score, p_off = _score_profile(p_star, scores, support, levels)
    parity_score, parity_off = _score_profile(q_parity, scores, support, levels)
    normal_score, normal_off = _score_profile(q_normal, scores, support, levels)
    tv_parity = 0.5 * float(np.sum(np.abs(parity_score - p_score)))
    tv_normal = 0.5 * float(np.sum(np.abs(normal_score - p_score)))

    hold_idx = np.where(holdout_mask)[0]
    order = hold_idx[np.argsort(-p_star[hold_idx])]
    target_share = p_star[order] / max(1e-15, float(p_star[order].sum()))
    parity_share = q_parity[order] / max(1e-15, float(q_parity[order].sum()))
    normal_share = q_normal[order] / max(1e-15, float(q_normal[order].sum()))

    fig, axes = plt.subplots(2, 2, figsize=hv.fig_size("full", 4.9))
    ax1, ax2, ax3, ax4 = axes.ravel()
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.085, top=0.84, wspace=0.26, hspace=0.45)

    # 1) Holdout recovery
    ax1.plot(q_vals, y_star, color=hv.COLORS["target"], lw=2.0, label=r"Target $p^*$")
    ax1.plot(q_vals, y_parity, color=hv.COLORS["model"], lw=2.2, label="IQP (parity)")
    ax1.plot(q_vals, y_normal, color=hv.COLORS["blue"], lw=2.1, ls="--", label=f"IQP ({normal_loss})")
    ax1.plot(q_vals, y_unif, color=hv.COLORS["gray"], lw=1.6, ls="--", label="Uniform")
    ax1.set_xlim(0, int(cfg["Qmax"]))
    ax1.set_ylim(-0.02, 1.03)
    ax1.set_xlabel(r"$Q$ samples")
    ax1.set_ylabel(r"Recovery $R(Q)$")
    ax1.set_title("1) Holdout recovery", pad=4)

    # 2) Beta score profile and match quality
    x = np.arange(len(levels))
    w = 0.24
    ax2.bar(x - w, p_score, width=w, color="#222222", alpha=0.85, label="Target p*")
    ax2.bar(x, parity_score, width=w, color=hv.COLORS["model"], alpha=0.85, label="IQP parity")
    ax2.bar(x + w, normal_score, width=w, color=hv.COLORS["blue"], alpha=0.85, label=f"IQP {normal_loss}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in levels])
    ax2.set_xlim(-0.7, len(levels) - 0.3)
    ax2.set_xlabel("Score level")
    ax2.set_ylabel("Probability mass")
    ax2.set_title(f"2) Score profile match | TV: parity={tv_parity:.3f}, {normal_loss}={tv_normal:.3f}", pad=4)

    # 3) Holdout recovery mass
    names = ["IQP parity", f"IQP {normal_loss}"]
    vals_qh = [metrics_parity["qH"], metrics_normal["qH"]]
    colors = [hv.COLORS["model"], hv.COLORS["blue"]]
    bars = ax3.bar(names, vals_qh, color=colors, alpha=0.9)
    ax3.axhline(float(q_unif[holdout_mask].sum()), color=hv.COLORS["gray"], ls="--", lw=1.6, label=r"Uniform $q(H)$")
    ax3.set_ylabel(r"Holdout mass $q(H)$")
    ax3.set_title("3) Holdout recovery mass", pad=4)

    # 4) Holdout mass equality (Lorenz-style)
    pop_t, cum_t = _lorenz_curve(target_share)
    pop_p, cum_p = _lorenz_curve(parity_share)
    pop_n, cum_n = _lorenz_curve(normal_share)
    g_t = _gini_from_shares(target_share)
    g_p = _gini_from_shares(parity_share)
    g_n = _gini_from_shares(normal_share)

    ax4.plot(pop_t, cum_t, color="#222222", lw=2.0, label=fr"Target (Gini={g_t:.3f})")
    ax4.plot(pop_p, cum_p, color=hv.COLORS["model"], lw=2.0, label=fr"Parity (Gini={g_p:.3f})")
    ax4.plot(pop_n, cum_n, color=hv.COLORS["blue"], lw=2.0, ls="--", label=fr"{normal_loss} (Gini={g_n:.3f})")
    ax4.plot([0.0, 1.0], [0.0, 1.0], color=hv.COLORS["gray"], lw=1.4, ls=":", label="Perfectly equal")
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)
    ax4.set_xlabel("Fraction of holdout states")
    ax4.set_ylabel("Cumulative holdout mass")
    ax4.set_title("4) Holdout mass equality", pad=4)
    ax4.legend(loc="lower right", frameon=False)

    legend_handles = [
        Line2D([0], [0], color=hv.COLORS["target"], lw=2.0, label=r"Target $p^*$"),
        Line2D([0], [0], color=hv.COLORS["model"], lw=2.2, label="IQP (parity)"),
        Line2D([0], [0], color=hv.COLORS["blue"], lw=2.1, ls="--", label=f"IQP ({normal_loss})"),
        Line2D([0], [0], color=hv.COLORS["gray"], lw=1.6, ls="--", label="Uniform"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        columnspacing=1.3,
        handlelength=1.9,
    )

    fig.suptitle(
        (
            f"Global holdout | beta={cfg['beta']} | n={int(cfg['n'])} | m={int(cfg['train_m'])} | "
            f"sigma={cfg['sigma']} | K={int(cfg['K'])} | steps={int(cfg['iqp_steps'])}"
        ),
        fontsize=9,
        y=0.90,
    )
    fig.savefig(outpath_pdf)
    fig.savefig(outpath_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "quick_global_beta08_parity_vs_normal_4panel"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--holdout-m-train", type=int, default=None)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=400)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--normal-loss", type=str, default="prob_mse", choices=["prob_mse", "xent"])
    ap.add_argument("--Qmax", type=int, default=10000)
    ap.add_argument("--Q80-thr", type=float, default=0.8)
    ap.add_argument("--Q80-search-max", type=int, default=200000)
    args = ap.parse_args()

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bits_table = hv.make_bits_table(args.n)
    p_star, support, scores = hv.build_target_distribution_paper(args.n, args.beta)
    good_mask = hv.topk_mask_by_scores(scores, support, frac=args.good_frac)
    holdout_m_train = args.holdout_m_train if args.holdout_m_train is not None else args.train_m

    # Global holdout: choose from full support, not only high-value subset.
    holdout_mask = hv.select_holdout_smart(
        p_star=p_star,
        good_mask=support.astype(bool),
        bits_table=bits_table,
        m_train=holdout_m_train,
        holdout_k=args.holdout_k,
        pool_size=args.holdout_pool,
        seed=args.seed + 111,
    )
    hv.save_holdout_list(holdout_mask, bits_table, p_star, scores, str(outdir), name="holdout_strings_global.txt")

    cfg = hv.Config(
        n=args.n,
        beta=args.beta,
        train_m=args.train_m,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        seed=args.seed,
        good_frac=args.good_frac,
        sigmas=[args.sigma],
        Ks=[args.K],
        Qmax=args.Qmax,
        Q80_thr=args.Q80_thr,
        Q80_search_max=args.Q80_search_max,
        target_family="paper_even",
        adversarial=False,
        use_iqp=True,
        use_classical=False,
        iqp_steps=args.iqp_steps,
        iqp_lr=args.iqp_lr,
        iqp_eval_every=args.iqp_eval_every,
        iqp_layers=args.layers,
        iqp_loss="parity_mse",
        outdir=str(outdir),
    )

    art_parity = hv.rerun_single_setting(
        cfg=cfg,
        p_star=p_star,
        holdout_mask=holdout_mask,
        bits_table=bits_table,
        sigma=args.sigma,
        K=args.K,
        return_hist=False,
        iqp_loss="parity_mse",
    )
    art_normal = hv.rerun_single_setting(
        cfg=cfg,
        p_star=p_star,
        holdout_mask=holdout_mask,
        bits_table=bits_table,
        sigma=args.sigma,
        K=args.K,
        return_hist=False,
        iqp_loss=args.normal_loss,
    )

    q_parity = np.array(art_parity["q_iqp"], dtype=np.float64)
    q_normal = np.array(art_normal["q_iqp"], dtype=np.float64)

    n_states = p_star.size
    h_size = int(np.sum(holdout_mask))
    q_unif = np.ones(n_states, dtype=np.float64) / n_states
    qH_unif = float(q_unif[holdout_mask].sum()) if h_size > 0 else 1.0
    metrics_parity = hv.compute_metrics_for_q(q_parity, holdout_mask, qH_unif, h_size, args.Q80_thr, args.Q80_search_max)
    metrics_normal = hv.compute_metrics_for_q(q_normal, holdout_mask, qH_unif, h_size, args.Q80_thr, args.Q80_search_max)

    cfg_dump: Dict[str, float] = {
        "n": float(args.n),
        "beta": float(args.beta),
        "train_m": float(args.train_m),
        "holdout_m_train": float(holdout_m_train),
        "holdout_k": float(args.holdout_k),
        "holdout_pool": float(args.holdout_pool),
        "seed": float(args.seed),
        "sigma": float(args.sigma),
        "K": float(args.K),
        "layers": float(args.layers),
        "iqp_steps": float(args.iqp_steps),
        "iqp_lr": float(args.iqp_lr),
        "iqp_eval_every": float(args.iqp_eval_every),
        "Qmax": float(args.Qmax),
    }

    plot_pdf = outdir / "global_holdout_beta08_parity_vs_normal_4panel.pdf"
    plot_png = outdir / "global_holdout_beta08_parity_vs_normal_4panel.png"
    _plot(
        outpath_pdf=plot_pdf,
        outpath_png=plot_png,
        p_star=p_star,
        support=support,
        scores=scores,
        holdout_mask=holdout_mask,
        q_parity=q_parity,
        q_normal=q_normal,
        metrics_parity=metrics_parity,
        metrics_normal=metrics_normal,
        normal_loss=args.normal_loss,
        cfg=cfg_dump,
    )

    summary = {
        "config": {
            "n": args.n,
            "beta": args.beta,
            "train_m": args.train_m,
            "holdout_m_train": holdout_m_train,
            "holdout_k": args.holdout_k,
            "holdout_pool": args.holdout_pool,
            "seed": args.seed,
            "sigma": args.sigma,
            "K": args.K,
            "layers": args.layers,
            "iqp_steps": args.iqp_steps,
            "iqp_lr": args.iqp_lr,
            "iqp_eval_every": args.iqp_eval_every,
            "normal_loss": args.normal_loss,
            "Qmax": args.Qmax,
            "Q80_thr": args.Q80_thr,
            "Q80_search_max": args.Q80_search_max,
        },
        "holdout": {
            "mode": "global",
            "size": h_size,
            "p_star_holdout": float(p_star[holdout_mask].sum()),
            "q_unif_holdout": qH_unif,
        },
        "metrics": {
            "parity": metrics_parity,
            "normal": metrics_normal,
        },
        "files": {
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
            "holdout_strings": str(outdir / "holdout_strings_global.txt"),
        },
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {plot_pdf}")
    print(f"[Saved] {plot_png}")
    print(f"[Saved] {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
