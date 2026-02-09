#!/usr/bin/env python3
"""Quick experiment: target vs spectral reconstruction (only)."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _q_grid(Qmax: int = 10000) -> np.ndarray:
    q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 120).astype(int)),
                np.linspace(1000, Qmax, 140).astype(int),
                np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100], dtype=int),
            ]
        )
    )
    return q[(q >= 0) & (q <= Qmax)]


def _score_profile(q: np.ndarray, scores: np.ndarray, support: np.ndarray, levels: List[int]) -> np.ndarray:
    s_int = scores.astype(int)
    vals = []
    for lv in levels:
        mask = support & (s_int == int(lv))
        vals.append(float(q[mask].sum()))
    return np.array(vals, dtype=np.float64)


def _build_holdout(
    args: argparse.Namespace,
    p_star: np.ndarray,
    support: np.ndarray,
    scores: np.ndarray,
    bits_table: np.ndarray,
    outdir: Path,
) -> np.ndarray:
    good_mask = hv.topk_mask_by_scores(scores, support, frac=args.good_frac)
    holdout_m_train = args.holdout_m_train if args.holdout_m_train is not None else args.train_m
    candidate_mask = support.astype(bool) if args.holdout_mode == "global" else good_mask
    holdout_mask = hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate_mask,
        bits_table=bits_table,
        m_train=holdout_m_train,
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
    q_spec: np.ndarray,
    scores: np.ndarray,
    support: np.ndarray,
    holdout_mask: np.ndarray,
    cfg: Dict[str, float],
    left_only: bool = False,
    no_title: bool = False,
) -> Dict[str, float]:
    q_vals = _q_grid(int(cfg["Qmax"]))
    y_star = hv.expected_unique_fraction(p_star, holdout_mask, q_vals)
    y_spec = hv.expected_unique_fraction(q_spec, holdout_mask, q_vals)
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_u = hv.expected_unique_fraction(q_unif, holdout_mask, q_vals)

    levels = sorted({int(v) for v in scores[support]})
    p_score = _score_profile(p_star, scores, support, levels)
    spec_score = _score_profile(q_spec, scores, support, levels)
    tv_score = 0.5 * float(np.sum(np.abs(spec_score - p_score)))

    if left_only:
        # Match the compact style of the visibility/invisibility single-panel plot.
        fig, ax1 = plt.subplots(1, 1, figsize=(3.18, 2.38), constrained_layout=True)
        ax2 = None
    else:
        fig, axes = plt.subplots(1, 2, figsize=hv.fig_size("full", 2.95))
        ax1, ax2 = axes.ravel()
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.83, wspace=0.30)

    # Left: holdout recovery kinetics
    ax1.plot(q_vals, y_star, color=hv.COLORS["target"], lw=1.9, label=r"Target $p^*$", zorder=6)
    ax1.plot(q_vals, y_spec, color="#555555", lw=1.9, ls="-.", label=r"Spectral $\tilde{q}$", zorder=5)
    ax1.plot(q_vals, y_u, color=hv.COLORS["gray"], lw=1.6, ls="--", alpha=0.9, label="Uniform", zorder=4)
    ax1.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7, zorder=3)
    ax1.set_xlim(0, int(cfg["Qmax"]))
    ax1.set_ylim(-0.02, 1.03)
    ax1.set_xlabel(r"$Q$")
    ax1.set_ylabel(r"Recovery $R(Q)$")
    if not no_title:
        ax1.set_title("1) Holdout recovery", pad=4)
    if left_only:
        ax1.legend(
            loc="center right",
            bbox_to_anchor=(0.985, 0.16),
            frameon=False,
            borderaxespad=0.10,
            handlelength=1.8,
            fontsize=6.8,
        )
    else:
        ax1.legend(loc="lower right", frameon=False)

    # Right: score-level target distribution vs spectral
    if ax2 is not None:
        x = np.arange(len(levels))
        w = 0.34
        ax2.bar(x - 0.5 * w, p_score, width=w, color="#222222", alpha=0.9, label=r"Target $p^*$")
        ax2.bar(x + 0.5 * w, spec_score, width=w, color="#777777", alpha=0.9, label=r"Spectral $\tilde{q}$")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(v) for v in levels])
        ax2.set_xlabel("Score level")
        ax2.set_ylabel("Probability mass")
        ax2.set_title(f"2) Score profile (TV={tv_score:.3f})", pad=4)
        ax2.legend(loc="upper right", frameon=False)

    if not no_title:
        fig.suptitle(
            (
                f"Target vs Spectral only | holdout={cfg['holdout_mode']} | beta={cfg['beta']} | "
                f"n={int(cfg['n'])} | m={int(cfg['train_m'])} | sigma={cfg['sigma']} | K={int(cfg['K'])}"
            ),
            fontsize=9,
            y=0.93 if ax2 is not None else 0.995,
        )
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    return {
        "tv_score": tv_score,
        "R1000_target": float(hv.expected_unique_fraction(p_star, holdout_mask, np.array([1000]))[0]),
        "R1000_spectral": float(hv.expected_unique_fraction(q_spec, holdout_mask, np.array([1000]))[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "quick_target_vs_spectral_only"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--holdout-m-train", type=int, default=None)
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--Qmax", type=int, default=10000)
    ap.add_argument("--left-only", action="store_true")
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bits_table = hv.make_bits_table(args.n)
    p_star, support, scores = hv.build_target_distribution_paper(args.n, args.beta)
    holdout_mask = _build_holdout(args, p_star, support, scores, bits_table, outdir)

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
        target_family="paper_even",
        adversarial=False,
        use_iqp=False,
        use_classical=False,
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
    )
    q_spec = np.array(art["q_spec"], dtype=np.float64)

    cfg_dump: Dict[str, float] = {
        "n": float(args.n),
        "beta": float(args.beta),
        "train_m": float(args.train_m),
        "holdout_k": float(args.holdout_k),
        "holdout_pool": float(args.holdout_pool),
        "seed": float(args.seed),
        "sigma": float(args.sigma),
        "K": float(args.K),
        "Qmax": float(args.Qmax),
        "holdout_mode": args.holdout_mode,
    }

    suffix = "_recovery_only_notitle" if (args.left_only and args.no_title) else (
        "_recovery_only" if args.left_only else ""
    )
    plot_pdf = outdir / f"target_vs_spectral_only_{args.holdout_mode}_beta{args.beta:g}{suffix}.pdf"
    plot_png = outdir / f"target_vs_spectral_only_{args.holdout_mode}_beta{args.beta:g}{suffix}.png"
    stats = _plot(
        out_pdf=plot_pdf,
        out_png=plot_png,
        p_star=p_star,
        q_spec=q_spec,
        scores=scores,
        support=support,
        holdout_mask=holdout_mask,
        cfg=cfg_dump,
        left_only=args.left_only,
        no_title=args.no_title,
    )

    summary = {
        "config": {
            "n": args.n,
            "beta": args.beta,
            "train_m": args.train_m,
            "holdout_mode": args.holdout_mode,
            "holdout_k": args.holdout_k,
            "holdout_pool": args.holdout_pool,
            "seed": args.seed,
            "sigma": args.sigma,
            "K": args.K,
            "Qmax": args.Qmax,
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
