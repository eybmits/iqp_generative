#!/usr/bin/env python3
"""IQP-only panel: red-shaded recovery curves + IQP Q80 heatmap."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _q_grid(Qmax: int) -> np.ndarray:
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


def _blend(c1: Tuple[float, float, float], c2: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    t = float(max(0.0, min(1.0, t)))
    return (
        (1.0 - t) * c1[0] + t * c2[0],
        (1.0 - t) * c1[1] + t * c2[1],
        (1.0 - t) * c1[2] + t * c2[2],
    )


def _rank_key(row: Dict[str, object]) -> Tuple[float, float]:
    q80 = float(row["Q80"])
    qhr = float(row["qH_ratio"])
    if np.isfinite(q80):
        return (q80, -qhr)
    return (1e99, -qhr)


def _select_holdout(
    p_star: np.ndarray,
    support: np.ndarray,
    scores: np.ndarray,
    bits_table: np.ndarray,
    holdout_mode: str,
    good_frac: float,
    holdout_k: int,
    holdout_pool: int,
    train_m: int,
    seed: int,
    outdir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    good_mask = hv.topk_mask_by_scores(scores, support, frac=good_frac)
    candidate_mask = support.astype(bool) if holdout_mode == "global" else good_mask
    holdout_mask = hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate_mask,
        bits_table=bits_table,
        m_train=train_m,
        holdout_k=holdout_k,
        pool_size=holdout_pool,
        seed=seed + 111,
    )
    holdout_name = "holdout_strings_global.txt" if holdout_mode == "global" else "holdout_strings_high_value.txt"
    hv.save_holdout_list(holdout_mask, bits_table, p_star, scores, str(outdir), name=holdout_name)
    return good_mask, holdout_mask


def _run_iqp_grid(
    cfg_base: hv.Config,
    p_star: np.ndarray,
    holdout_mask: np.ndarray,
    bits_table: np.ndarray,
    sigmas: List[float],
    Ks: List[int],
    q80_thr: float,
    q80_search_max: int,
) -> List[Dict[str, object]]:
    n_states = p_star.size
    h_size = int(np.sum(holdout_mask))
    q_unif = np.ones(n_states, dtype=np.float64) / n_states
    qH_unif = float(q_unif[holdout_mask].sum()) if h_size > 0 else 1.0

    rows: List[Dict[str, object]] = []
    for sigma in sigmas:
        for K in Ks:
            art = hv.rerun_single_setting(
                cfg=cfg_base,
                p_star=p_star,
                holdout_mask=holdout_mask,
                bits_table=bits_table,
                sigma=float(sigma),
                K=int(K),
                return_hist=False,
                iqp_loss=str(cfg_base.iqp_loss),
            )
            q_iqp = np.array(art["q_iqp"], dtype=np.float64)
            met = hv.compute_metrics_for_q(
                q_iqp,
                holdout_mask,
                qH_unif,
                h_size,
                q80_thr,
                q80_search_max,
            )
            rows.append(
                {
                    "sigma": float(sigma),
                    "K": int(K),
                    "q": q_iqp,
                    "qH_ratio": float(met["qH_ratio"]),
                    "Q80": float(met["Q80"]),
                    "R_Q1000": float(met["R_Q1000"]),
                }
            )
    return rows


def _plot(
    out_pdf: Path,
    out_png: Path,
    out_left_pdf: Path,
    out_left_png: Path,
    out_right_pdf: Path,
    out_right_png: Path,
    p_star: np.ndarray,
    holdout_mask: np.ndarray,
    rows: List[Dict[str, object]],
    sigmas: List[float],
    Ks: List[int],
    Qmax: int,
    title: str,
) -> Dict[str, object]:
    q_vals = _q_grid(Qmax)
    y_star = hv.expected_unique_fraction(p_star, holdout_mask, q_vals)
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, q_vals)

    ranked = sorted(rows, key=_rank_key)
    n = len(ranked)
    dark_red = (0.73, 0.00, 0.00)
    light_red = (0.98, 0.78, 0.78)

    cmap_redblack = LinearSegmentedColormap.from_list("RedBlack", ["#FF1A1A", "#AA0000", "#000000"])
    mat_q80 = np.full((len(sigmas), len(Ks)), np.nan, dtype=np.float64)
    for r in rows:
        i = sigmas.index(float(r["sigma"]))
        j = Ks.index(int(r["K"]))
        mat_q80[i, j] = float(r["Q80"])
    mat_log_q80 = hv._safe_log10_for_heatmap(mat_q80)

    best = ranked[0]
    best_label = fr"Best IQP-QCBM, $\sigma$={best['sigma']:g}, $K$={int(best['K'])}"

    legend_handles_left = [
        Line2D([0], [0], color=hv.COLORS["target"], lw=2.0, label=r"Target $p^*$"),
        Line2D([0], [0], color=dark_red, lw=2.4, label=best_label),
        Line2D([0], [0], color=light_red, lw=1.5, label=f"Other IQP settings (n={n-1})"),
        Line2D([0], [0], color=hv.COLORS["gray"], lw=1.5, ls="--", label="Uniform"),
    ]

    max_pd = float(np.max(mat_log_q80[np.isfinite(mat_log_q80)])) if np.isfinite(mat_log_q80).any() else 1.0

    def _draw_left(ax: plt.Axes) -> None:
        # Left: IQP baselines in red shades.
        for idx, r in enumerate(ranked[::-1]):
            rank = n - 1 - idx
            t = (rank / max(1, n - 1))
            color = _blend(dark_red, light_red, t)
            q = np.array(r["q"], dtype=np.float64)
            y = hv.expected_unique_fraction(q, holdout_mask, q_vals)
            lw = 1.2 if rank > 0 else 2.4
            alpha = 0.65 if rank > 0 else 1.0
            ax.plot(q_vals, y, color=color, lw=lw, alpha=alpha, zorder=3 + rank)

        ax.plot(q_vals, y_star, color=hv.COLORS["target"], lw=2.0, zorder=20)
        ax.plot(q_vals, y_unif, color=hv.COLORS["gray"], lw=1.5, ls="--", zorder=2)
        ax.axhline(1.0, color=hv.COLORS["gray"], ls=":", alpha=0.7)
        ax.set_xlim(0, Qmax)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$Q$ samples from model")
        ax.set_ylabel(r"Recovery $R(Q)$")
        ax.legend(handles=legend_handles_left, loc="lower right", frameon=False)

    def _draw_right(ax: plt.Axes, fig_for_cbar: plt.Figure) -> None:
        # Right: IQP Q80 heatmap.
        im = ax.imshow(mat_log_q80, aspect="auto", cmap=cmap_redblack)
        ax.set_xticks(np.arange(len(Ks)))
        ax.set_yticks(np.arange(len(sigmas)))
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_yticklabels([str(s) for s in sigmas])
        ax.set_xlabel("K (number of parity features)")
        ax.set_ylabel(r"$\sigma$ (feature sparsity)")

        ax.set_xticks(np.arange(-0.5, len(Ks), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(sigmas), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.28)
        ax.tick_params(which="minor", bottom=False, left=False)

        cbar = fig_for_cbar.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(r"$\log_{10} Q_{80}$")

        for i in range(mat_q80.shape[0]):
            for j in range(mat_q80.shape[1]):
                val = mat_q80[i, j]
                txt = r"$\infty$" if not np.isfinite(val) else f"{val:.0f}"
                txt_color = "white"
                if mat_log_q80[i, j] < max_pd * 0.45:
                    txt_color = "black"
                ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=8, fontweight="bold")

    # Combined panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=hv.fig_size("full", 2.95))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.84, wspace=0.32)
    _draw_left(ax1)
    _draw_right(ax2, fig)

    if title:
        fig.suptitle(title, fontsize=9, y=0.93)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    # Left-only panel (no title)
    fig_l, ax_l = plt.subplots(1, 1, figsize=hv.fig_size("col", 2.95))
    fig_l.subplots_adjust(left=0.13, right=0.99, bottom=0.18, top=0.98)
    _draw_left(ax_l)
    fig_l.savefig(out_left_pdf)
    fig_l.savefig(out_left_png, dpi=220)
    plt.close(fig_l)

    # Right-only panel (no title)
    fig_r, ax_r = plt.subplots(1, 1, figsize=hv.fig_size("col", 2.95))
    fig_r.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.98)
    _draw_right(ax_r, fig_r)
    fig_r.savefig(out_right_pdf)
    fig_r.savefig(out_right_png, dpi=220)
    plt.close(fig_r)

    return {
        "best_sigma": float(best["sigma"]),
        "best_K": int(best["K"]),
        "best_Q80": float(best["Q80"]),
        "best_R_Q1000": float(best["R_Q1000"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "quick_iqp_red_baselines_panel"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--holdout-mode", type=str, default="high_value", choices=["high_value", "global"])
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigmas", type=str, default="0.5,1.0,2.0,3.0")
    ap.add_argument("--Ks", type=str, default="128,256,512")
    ap.add_argument("--iqp-loss", type=str, default="parity_mse", choices=["parity_mse", "prob_mse", "xent"])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--Qmax", type=int, default=10000)
    ap.add_argument("--Q80-thr", type=float, default=0.8)
    ap.add_argument("--Q80-search-max", type=int, default=200000)
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sigmas = _parse_floats(args.sigmas)
    Ks = _parse_ints(args.Ks)
    if not sigmas or not Ks:
        raise ValueError("sigmas and Ks must be non-empty.")

    bits_table = hv.make_bits_table(args.n)
    p_star, support, scores = hv.build_target_distribution_paper(args.n, args.beta)
    _, holdout_mask = _select_holdout(
        p_star=p_star,
        support=support,
        scores=scores,
        bits_table=bits_table,
        holdout_mode=args.holdout_mode,
        good_frac=args.good_frac,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        train_m=args.train_m,
        seed=args.seed,
        outdir=outdir,
    )

    cfg = hv.Config(
        n=args.n,
        beta=args.beta,
        train_m=args.train_m,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        seed=args.seed,
        good_frac=args.good_frac,
        sigmas=sigmas,
        Ks=Ks,
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
        iqp_loss=args.iqp_loss,
        outdir=str(outdir),
    )

    rows = _run_iqp_grid(
        cfg_base=cfg,
        p_star=p_star,
        holdout_mask=holdout_mask,
        bits_table=bits_table,
        sigmas=sigmas,
        Ks=Ks,
        q80_thr=args.Q80_thr,
        q80_search_max=args.Q80_search_max,
    )
    rows_sorted = sorted(rows, key=_rank_key)

    title = ""
    if not args.no_title:
        title = (
            f"IQP-QCBM baselines only | holdout={args.holdout_mode} | beta={args.beta} | "
            f"m={args.train_m} | loss={args.iqp_loss}"
        )

    stem = f"iqp_red_baselines_panel_{args.holdout_mode}_beta{args.beta:g}"
    plot_pdf = outdir / f"{stem}.pdf"
    plot_png = outdir / f"{stem}.png"
    left_pdf = outdir / f"{stem}_left_only_notitle.pdf"
    left_png = outdir / f"{stem}_left_only_notitle.png"
    right_pdf = outdir / f"{stem}_right_only_notitle.pdf"
    right_png = outdir / f"{stem}_right_only_notitle.png"
    best_stats = _plot(
        out_pdf=plot_pdf,
        out_png=plot_png,
        out_left_pdf=left_pdf,
        out_left_png=left_png,
        out_right_pdf=right_pdf,
        out_right_png=right_png,
        p_star=p_star,
        holdout_mask=holdout_mask,
        rows=rows,
        sigmas=sigmas,
        Ks=Ks,
        Qmax=args.Qmax,
        title=title,
    )

    summary = {
        "config": {
            "n": args.n,
            "beta": args.beta,
            "train_m": args.train_m,
            "holdout_mode": args.holdout_mode,
            "holdout_k": args.holdout_k,
            "holdout_pool": args.holdout_pool,
            "good_frac": args.good_frac,
            "seed": args.seed,
            "sigmas": sigmas,
            "Ks": Ks,
            "iqp_loss": args.iqp_loss,
            "layers": args.layers,
            "iqp_steps": args.iqp_steps,
            "iqp_lr": args.iqp_lr,
            "iqp_eval_every": args.iqp_eval_every,
            "Qmax": args.Qmax,
            "Q80_thr": args.Q80_thr,
            "Q80_search_max": args.Q80_search_max,
        },
        "holdout": {
            "size": int(np.sum(holdout_mask)),
            "p_star_holdout": float(p_star[holdout_mask].sum()),
        },
        "best": best_stats,
        "rows_ranked": [
            {
                "sigma": float(r["sigma"]),
                "K": int(r["K"]),
                "Q80": float(r["Q80"]),
                "qH_ratio": float(r["qH_ratio"]),
                "R_Q1000": float(r["R_Q1000"]),
            }
            for r in rows_sorted
        ],
        "files": {
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
            "left_only_pdf": str(left_pdf),
            "left_only_png": str(left_png),
            "right_only_pdf": str(right_pdf),
            "right_only_png": str(right_png),
        },
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {plot_pdf}")
    print(f"[Saved] {plot_png}")
    print(f"[Saved] {left_pdf}")
    print(f"[Saved] {left_png}")
    print(f"[Saved] {right_pdf}")
    print(f"[Saved] {right_png}")
    print(f"[Saved] {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
