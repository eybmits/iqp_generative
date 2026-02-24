#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 49: Test hypothesis that spectral-completion Q80 is a useful proxy for IQP settings.

Protocol:
1) Sample N settings (sigma, K).
2) For each setting, compute spectral completion q_spec and Q80_spec.
3) Pick best and worst settings by Q80_spec.
4) Train IQP parity only on these two settings.
5) Plot:
   - spectral ranking over N settings
   - recovery curves for target/uniform + (spec,iqp) for best and worst
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402
from experiments.legacy.exp44_beta_holdout_score_state_diagnostic import _build_holdout  # noqa: E402


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _beta_tag(beta: float) -> str:
    return f"{beta:.2f}".replace("-", "m").replace(".", "p")


def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _q_grid(qmax: int = 10000) -> np.ndarray:
    q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 120).astype(int)),
                np.linspace(1000, qmax, 160).astype(int),
                np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100], dtype=int),
            ]
        )
    )
    return q[(q >= 0) & (q <= qmax)]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _select_pairs(sigmas: Sequence[float], Ks: Sequence[int], num_settings: int, seed: int) -> List[Tuple[float, int]]:
    all_pairs: List[Tuple[float, int]] = []
    for s in sigmas:
        for k in Ks:
            all_pairs.append((float(s), int(k)))
    if num_settings >= len(all_pairs):
        return all_pairs
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(all_pairs), size=int(num_settings), replace=False)
    picked = [all_pairs[int(i)] for i in idx]
    picked.sort(key=lambda t: (t[0], t[1]))
    return picked


def _build_cfg(
    args: argparse.Namespace,
    sigma: float,
    K: int,
    use_iqp: bool,
) -> hv.Config:
    return hv.Config(
        n=int(args.n),
        beta=float(args.beta),
        train_m=int(args.train_m),
        holdout_k=int(args.holdout_k),
        holdout_pool=int(args.holdout_pool),
        seed=int(args.seed),
        good_frac=float(args.good_frac),
        sigmas=[float(sigma)],
        Ks=[int(K)],
        Qmax=int(args.qmax),
        Q80_thr=float(args.q80_thr),
        Q80_search_max=int(args.q80_search_max),
        target_family="paper_even",
        adversarial=False,
        use_iqp=bool(use_iqp),
        use_classical=False,
        iqp_steps=int(args.iqp_steps),
        iqp_lr=float(args.iqp_lr),
        iqp_eval_every=int(args.iqp_eval_every),
        iqp_layers=int(args.layers),
        iqp_loss="parity_mse",
        outdir=str(args.outdir),
    )


def _pick_best_worst(rows: Sequence[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, object]]:
    if len(rows) == 0:
        raise RuntimeError("No spectral rows available.")
    finite = [r for r in rows if np.isfinite(float(r["Q80_spec"]))]
    if len(finite) >= 2:
        best = min(finite, key=lambda r: float(r["Q80_spec"]))
        worst = max(finite, key=lambda r: float(r["Q80_spec"]))
        return best, worst
    if len(finite) == 1:
        best = finite[0]
        # If only one finite value exists, use largest (possibly inf) as worst.
        worst = max(rows, key=lambda r: float(r["Q80_spec"]) if np.isfinite(float(r["Q80_spec"])) else float("inf"))
        return best, worst
    # No finite values; fallback to max/min qH_ratio as proxy.
    best = max(rows, key=lambda r: float(r["qH_ratio_spec"]))
    worst = min(rows, key=lambda r: float(r["qH_ratio_spec"]))
    return best, worst


def run(args: argparse.Namespace) -> None:
    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP training. Install with `pip install pennylane`.")

    hv.set_style(base=8)
    outdir = _ensure_outdir(Path(args.outdir))

    bits_table = hv.make_bits_table(int(args.n))
    p_star, support, scores = hv.build_target_distribution_paper(int(args.n), float(args.beta))
    good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))
    holdout_mask = _build_holdout(
        holdout_mode=str(args.holdout_mode),
        holdout_selection=str(args.holdout_selection),
        p_star=p_star,
        support=support,
        scores=scores,
        good_mask=good_mask,
        bits_table=bits_table,
        m_train_for_holdout=int(args.holdout_m_train),
        holdout_k=int(args.holdout_k),
        holdout_pool=int(args.holdout_pool),
        seed=int(args.seed),
        protect_max_score=bool(int(args.protect_max_score)),
        dense_global_levels_only=bool(int(args.dense_global_levels_only)),
        dense_global_min_states_per_level=int(args.dense_global_min_states_per_level),
    )

    holdout_size = int(np.sum(holdout_mask))
    if holdout_size <= 0:
        raise RuntimeError("Holdout is empty.")
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    qH_unif = float(np.sum(q_unif[holdout_mask]))

    sigmas = _parse_list_floats(str(args.sigmas))
    Ks = _parse_list_ints(str(args.Ks))
    pairs = _select_pairs(sigmas, Ks, int(args.num_settings), int(args.selection_seed))
    if len(pairs) < 2:
        raise RuntimeError("Need at least 2 settings for best-vs-worst comparison.")

    spectral_rows: List[Dict[str, object]] = []
    spectral_q: Dict[Tuple[float, int], np.ndarray] = {}
    for i, (sigma, K) in enumerate(pairs, start=1):
        print(f"[spectral {i}/{len(pairs)}] sigma={sigma:g}, K={K}")
        cfg_spec = _build_cfg(args=args, sigma=float(sigma), K=int(K), use_iqp=False)
        art = hv.rerun_single_setting(
            cfg=cfg_spec,
            p_star=p_star,
            holdout_mask=holdout_mask,
            bits_table=bits_table,
            sigma=float(sigma),
            K=int(K),
            return_hist=False,
            iqp_loss="parity_mse",
        )
        q_spec = np.asarray(art["q_spec"], dtype=np.float64)
        spectral_q[(float(sigma), int(K))] = q_spec.copy()
        m = hv.compute_metrics_for_q(
            q=q_spec,
            holdout_mask=holdout_mask,
            qH_unif=qH_unif,
            H_size=holdout_size,
            Q80_thr=float(args.q80_thr),
            Q80_search_max=int(args.q80_search_max),
        )
        spectral_rows.append(
            {
                "sigma": float(sigma),
                "K": int(K),
                "Q80_spec": float(m["Q80"]),
                "qH_spec": float(m["qH"]),
                "qH_ratio_spec": float(m["qH_ratio"]),
                "R_spec_Q1000": float(m["R_Q1000"]),
                "R_spec_Q10000": float(m["R_Q10000"]),
            }
        )

    best_spec, worst_spec = _pick_best_worst(spectral_rows)
    best_key = (float(best_spec["sigma"]), int(best_spec["K"]))
    worst_key = (float(worst_spec["sigma"]), int(worst_spec["K"]))
    print(
        "[selected] best spec:",
        f"sigma={best_key[0]:g},K={best_key[1]},Q80={float(best_spec['Q80_spec']):.0f}",
    )
    print(
        "[selected] worst spec:",
        f"sigma={worst_key[0]:g},K={worst_key[1]},Q80={float(worst_spec['Q80_spec']):.0f}",
    )

    selected_rows: List[Dict[str, object]] = []
    q_iqp_by_tag: Dict[str, np.ndarray] = {}
    for tag, key in (("best", best_key), ("worst", worst_key)):
        sigma, K = key
        print(f"[iqp/{tag}] sigma={sigma:g}, K={K}")
        cfg_iqp = _build_cfg(args=args, sigma=float(sigma), K=int(K), use_iqp=True)
        art = hv.rerun_single_setting(
            cfg=cfg_iqp,
            p_star=p_star,
            holdout_mask=holdout_mask,
            bits_table=bits_table,
            sigma=float(sigma),
            K=int(K),
            return_hist=False,
            iqp_loss="parity_mse",
        )
        q_iqp = art["q_iqp"]
        if not isinstance(q_iqp, np.ndarray):
            raise RuntimeError(f"IQP distribution missing for {tag} setting.")
        q_iqp_by_tag[tag] = q_iqp.copy()
        met = hv.compute_metrics_for_q(
            q=q_iqp,
            holdout_mask=holdout_mask,
            qH_unif=qH_unif,
            H_size=holdout_size,
            Q80_thr=float(args.q80_thr),
            Q80_search_max=int(args.q80_search_max),
        )
        spec_row = best_spec if tag == "best" else worst_spec
        selected_rows.append(
            {
                "tag": tag,
                "sigma": float(sigma),
                "K": int(K),
                "Q80_spec": float(spec_row["Q80_spec"]),
                "Q80_iqp": float(met["Q80"]),
                "qH_ratio_spec": float(spec_row["qH_ratio_spec"]),
                "qH_ratio_iqp": float(met["qH_ratio"]),
                "R_iqp_Q1000": float(met["R_Q1000"]),
                "R_iqp_Q10000": float(met["R_Q10000"]),
            }
        )

    ranked = sorted(
        spectral_rows,
        key=lambda r: float(r["Q80_spec"]) if np.isfinite(float(r["Q80_spec"])) else float("inf"),
    )
    rank_lookup: Dict[Tuple[float, int], int] = {
        (float(r["sigma"]), int(r["K"])): i for i, r in enumerate(ranked)
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2), constrained_layout=False)

    ax = axes[0]
    x = np.arange(len(ranked))
    y = np.array(
        [
            float(r["Q80_spec"]) if np.isfinite(float(r["Q80_spec"])) else float(args.q80_search_max) * 1.05
            for r in ranked
        ],
        dtype=np.float64,
    )
    bars = ax.bar(x, y, color="#9A9A9A", alpha=0.85)

    best_idx = rank_lookup[best_key]
    worst_idx = rank_lookup[worst_key]
    bars[best_idx].set_color("#2CA02C")
    bars[worst_idx].set_color("#D62728")

    labels = [fr"$\sigma$={float(r['sigma']):g},K={int(r['K'])}" for r in ranked]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"Spectral $Q_{80}$ (lower better)")
    ax.set_title(f"{len(ranked)} spectral completions ranked by holdout discovery")

    finite_vals = [float(r["Q80_spec"]) for r in ranked if np.isfinite(float(r["Q80_spec"])) and float(r["Q80_spec"]) > 0]
    if finite_vals:
        ratio = max(finite_vals) / max(min(finite_vals), 1e-9)
        if ratio >= 6.0:
            ax.set_yscale("log")

    handles = [
        Patch(facecolor="#2CA02C", alpha=0.9, label="Best spectral setting"),
        Patch(facecolor="#D62728", alpha=0.9, label="Worst spectral setting"),
        Patch(facecolor="#9A9A9A", alpha=0.9, label="Other sampled settings"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.8, frameon=True)

    ax = axes[1]
    Q = _q_grid(int(args.qmax))
    y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, Q)
    ax.plot(Q, y_star, color=hv.COLORS["target"], linewidth=2.0, label=r"Target $p^*$")
    ax.plot(Q, y_unif, color=hv.COLORS["gray"], linestyle="--", linewidth=1.6, label="Uniform")

    for tag, color in (("best", "#2CA02C"), ("worst", "#D62728")):
        spec = best_spec if tag == "best" else worst_spec
        key = (float(spec["sigma"]), int(spec["K"]))
        q_spec = spectral_q[key]
        q_iqp = q_iqp_by_tag[tag]
        y_spec = hv.expected_unique_fraction(q_spec, holdout_mask, Q)
        y_iqp = hv.expected_unique_fraction(q_iqp, holdout_mask, Q)

        ax.plot(
            Q,
            y_spec,
            color=color,
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            label=(
                f"Spectral {tag}: "
                + fr"$\sigma$={float(spec['sigma']):g}, K={int(spec['K'])}, Q80={float(spec['Q80_spec']):.0f}"
            ),
        )
        q80_iqp = float([r for r in selected_rows if str(r["tag"]) == tag][0]["Q80_iqp"])
        ax.plot(
            Q,
            y_iqp,
            color=color,
            linewidth=2.3,
            linestyle="-",
            label=(
                f"IQP on {tag}: "
                + fr"$\sigma$={float(spec['sigma']):g}, K={int(spec['K'])}, Q80={q80_iqp:.0f}"
            ),
        )

    ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlim(0, int(args.qmax))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_title("IQP recovery after selecting best vs worst spectral proxy")
    ax.legend(loc="lower right", fontsize=6.2, frameon=True)

    fig.suptitle(
        fr"$\beta$={float(args.beta):g}, holdout={args.holdout_mode}/{args.holdout_selection}, "
        + fr"$m$={int(args.train_m)}, |H|={int(args.holdout_k)}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.18, wspace=0.25)

    btag = _beta_tag(float(args.beta))
    prefix = (
        f"beta_{btag}_{args.holdout_mode}_{args.holdout_selection}_m{int(args.train_m)}"
        f"_holdout{int(args.holdout_k)}_spectral{len(ranked)}_bestworst_iqp"
    )
    plot_pdf = outdir / f"{prefix}.pdf"
    plot_png = outdir / f"{prefix}.png"
    spectral_csv = outdir / f"{prefix}_spectral_rankings.csv"
    selected_csv = outdir / f"{prefix}_selected_iqp_metrics.csv"
    summary_json = outdir / f"{prefix}_summary.json"

    fig.savefig(plot_pdf)
    fig.savefig(plot_png, dpi=300)
    plt.close(fig)

    ranked_rows: List[Dict[str, object]] = []
    for i, r in enumerate(ranked, start=1):
        rr = dict(r)
        rr["rank"] = int(i)
        rr["is_best"] = bool((float(r["sigma"]), int(r["K"])) == best_key)
        rr["is_worst"] = bool((float(r["sigma"]), int(r["K"])) == worst_key)
        ranked_rows.append(rr)

    _write_csv(spectral_csv, ranked_rows)
    _write_csv(selected_csv, selected_rows)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "n": int(args.n),
                    "beta": float(args.beta),
                    "seed": int(args.seed),
                    "train_m": int(args.train_m),
                    "holdout_mode": str(args.holdout_mode),
                    "holdout_selection": str(args.holdout_selection),
                    "holdout_k": int(args.holdout_k),
                    "iqp_steps": int(args.iqp_steps),
                    "iqp_lr": float(args.iqp_lr),
                    "layers": int(args.layers),
                    "q80_thr": float(args.q80_thr),
                    "q80_search_max": int(args.q80_search_max),
                    "num_settings": int(args.num_settings),
                    "selection_seed": int(args.selection_seed),
                },
                "best_setting": {
                    "sigma": float(best_key[0]),
                    "K": int(best_key[1]),
                    "Q80_spec": float(best_spec["Q80_spec"]),
                    "Q80_iqp": float([r for r in selected_rows if str(r["tag"]) == "best"][0]["Q80_iqp"]),
                },
                "worst_setting": {
                    "sigma": float(worst_key[0]),
                    "K": int(worst_key[1]),
                    "Q80_spec": float(worst_spec["Q80_spec"]),
                    "Q80_iqp": float([r for r in selected_rows if str(r["tag"]) == "worst"][0]["Q80_iqp"]),
                },
                "outputs": {
                    "plot_pdf": str(plot_pdf),
                    "plot_png": str(plot_png),
                    "spectral_rankings_csv": str(spectral_csv),
                    "selected_iqp_metrics_csv": str(selected_csv),
                },
            },
            f,
            indent=2,
        )

    print(f"[saved] {plot_pdf}")
    print(f"[saved] {plot_png}")
    print(f"[saved] {spectral_csv}")
    print(f"[saved] {selected_csv}")
    print(f"[saved] {summary_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="10x spectral proxy test: best-vs-worst IQP transfer.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "49_spectral_proxy_bestworst_iqp"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--good-frac", type=float, default=0.05)

    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--holdout-selection", type=str, default="smart", choices=["smart", "random"])
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--protect-max-score", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dense-global-levels-only", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dense-global-min-states-per-level", type=int, default=64)

    ap.add_argument("--sigmas", type=str, default="0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4")
    ap.add_argument("--Ks", type=str, default="64,96,128,192,256")
    ap.add_argument("--num-settings", type=int, default=10)
    ap.add_argument("--selection-seed", type=int, default=123)

    ap.add_argument("--iqp-steps", type=int, default=300)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--layers", type=int, default=1)

    ap.add_argument("--qmax", type=int, default=10000)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

