#!/usr/bin/env python3
"""Visibility plot with additional standard spectral reconstruction baseline."""

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

from experiments.legacy import exp03_visibility_minvis as vis  # noqa: E402
from iqp_generative import core as hv  # noqa: E402


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _build_visible_invisible(
    n: int,
    beta: float,
    good_frac: float,
    score_level: int,
    holdout_k: int,
    train_m: int,
    K: int,
    seed: int,
    improve_steps: int,
    sigma_grid: List[float],
) -> Dict[str, object]:
    N = 2 ** n
    bits_table = vis.make_bits_table(n)
    p_star, support, scores = vis.build_target_distribution(n, beta)
    good_mask = vis.topk_mask(scores, support, frac=good_frac)
    s_int = scores.astype(int)

    def pool_for_level(level: int) -> np.ndarray:
        return np.where(good_mask & (s_int == int(level)))[0]

    cand = pool_for_level(score_level)
    if cand.size < 2 * holdout_k:
        levels = sorted(set(s_int[good_mask].tolist()))
        viable = [lvl for lvl in levels if pool_for_level(lvl).size >= 2 * holdout_k]
        if not viable:
            raise RuntimeError("No score level has >= 2*holdout_k candidates.")
        chosen_level = min(viable, key=lambda lv: abs(int(lv) - int(score_level)))
        cand = pool_for_level(chosen_level)
        score_level = int(chosen_level)

    best = None
    for gi, sigma in enumerate(sigma_grid):
        alphas = vis.sample_alphas(n, float(sigma), int(K), seed=int(seed) + 999 + 17 * gi)
        P = vis.build_parity_matrix(alphas, bits_table)

        # Reference spectral signal used only for holdout construction.
        z_ref = P @ p_star
        r_ref = P.T @ z_ref
        cand_vals = r_ref[cand]

        H_inv, _ = vis.greedy_min_abs_sum(
            cand_idxs=cand,
            cand_vals=cand_vals,
            k=int(holdout_k),
            improve_steps=int(improve_steps),
            seed=int(seed) + 12345 + 31 * gi,
        )
        inv_set = set(int(i) for i in H_inv.tolist())
        remaining = np.array([int(i) for i in cand.tolist() if int(i) not in inv_set], dtype=int)
        if remaining.size >= int(holdout_k):
            rem_vals = r_ref[remaining]
            order_vis = np.argsort(-rem_vals)
            H_vis = remaining[order_vis[: int(holdout_k)]]
        else:
            order_vis = np.argsort(-cand_vals)
            H_vis = cand[order_vis[: int(holdout_k)]]

        vis_vis_ref = float(np.sum(r_ref[H_vis]) / N)
        vis_inv_ref = float(np.sum(r_ref[H_inv]) / N)
        obj = abs(vis_inv_ref)
        tie = abs(vis_vis_ref)
        if (best is None) or (obj < best["obj"] - 1e-18) or (abs(obj - best["obj"]) <= 1e-18 and tie > best["tie"]):
            best = {
                "sigma": float(sigma),
                "P": P,
                "H_vis": np.array(H_vis, dtype=int),
                "H_inv": np.array(H_inv, dtype=int),
                "Vis_vis_ref": vis_vis_ref,
                "Vis_inv_ref": vis_inv_ref,
                "obj": obj,
                "tie": tie,
            }

    assert best is not None
    P = np.array(best["P"], dtype=np.float64)
    H_vis = np.array(best["H_vis"], dtype=int)
    H_inv = np.array(best["H_inv"], dtype=int)

    def completion_for_holdout_idxs(holdout_idxs: np.ndarray, seed_offset: int) -> Tuple[np.ndarray, np.ndarray, float]:
        mask = np.zeros(N, dtype=bool)
        mask[holdout_idxs] = True
        p_train = p_star.copy()
        p_train[mask] = 0.0
        p_train /= float(np.sum(p_train))
        idxs_train = vis.sample_indices(p_train, int(train_m), seed=int(seed) + 7 + seed_offset)
        emp = vis.empirical_dist(idxs_train, N)
        z_train = P @ emp
        r_train = P.T @ z_train
        vis_train = float(np.sum(r_train[holdout_idxs]) / N)
        q_lin = vis.linear_band_reconstruction(P, z_train, n)
        q_tilde = vis.completion_by_axioms(q_lin)
        return q_tilde, mask, vis_train

    q_vis, mask_vis, vis_vis_train = completion_for_holdout_idxs(H_vis, seed_offset=0)
    q_inv, mask_inv, vis_inv_train = completion_for_holdout_idxs(H_inv, seed_offset=123)

    return {
        "bits_table": bits_table,
        "p_star": p_star,
        "support": support,
        "scores": scores,
        "good_mask": good_mask,
        "score_level": int(score_level),
        "P": P,
        "sigma_used": float(best["sigma"]),
        "H_vis": H_vis,
        "H_inv": H_inv,
        "mask_vis": mask_vis,
        "mask_inv": mask_inv,
        "q_vis": q_vis,
        "q_inv": q_inv,
        "Vis_vis_ref": float(best["Vis_vis_ref"]),
        "Vis_inv_ref": float(best["Vis_inv_ref"]),
        "Vis_vis_train": float(vis_vis_train),
        "Vis_inv_train": float(vis_inv_train),
        "completion_for_holdout_idxs": completion_for_holdout_idxs,
    }


def _plot_curves(
    out_pdf: Path,
    out_png: Path,
    p_star: np.ndarray,
    mask_vis: np.ndarray,
    mask_inv: np.ndarray,
    mask_std: np.ndarray,
    q_vis: np.ndarray,
    q_inv: np.ndarray,
    q_std: np.ndarray,
    qmax: int,
    std_holdout_mode: str,
) -> None:
    Q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 110).astype(int)),
                np.linspace(1000, qmax, 120).astype(int),
            ]
        )
    )
    Q = Q[Q <= qmax]

    y_star = vis.expected_unique_fraction(p_star, mask_vis, Q)
    y_vis = vis.expected_unique_fraction(q_vis, mask_vis, Q)
    y_inv = vis.expected_unique_fraction(q_inv, mask_inv, Q)
    y_std = vis.expected_unique_fraction(q_std, mask_std, Q)
    y_u = vis.expected_unique_fraction(np.ones_like(p_star) / p_star.size, mask_vis, Q)

    fig, ax = plt.subplots(figsize=(3.18, 2.38), constrained_layout=True)
    ax.plot(Q, y_star, color=vis.COLORS["target"], linestyle=vis.LS["target"], linewidth=1.9, label=r"Target $p^*$", zorder=6)
    ax.plot(Q, y_vis, color=vis.COLORS["vis"], linestyle=vis.LS["vis"], linewidth=2.2, label=r"Visible $H_{\mathrm{vis}}$", zorder=7)
    ax.plot(Q, y_inv, color=vis.COLORS["inv"], linestyle=vis.LS["inv"], linewidth=2.0, label=r"Invisible $H_{\mathrm{inv}}$", zorder=5)
    std_label = r"Standard spectral $H_{\mathrm{std}}$ (global)" if std_holdout_mode == "global" else r"Standard spectral $H_{\mathrm{std}}$ (high-value)"
    ax.plot(Q, y_std, color="#555555", linestyle="-.", linewidth=1.9, label=std_label, zorder=4)
    ax.plot(Q, y_u, color=vis.COLORS["uniform"], linestyle=vis.LS["uniform"], linewidth=1.6, label="Uniform", zorder=3)
    ax.axhline(1.0, color=vis.COLORS["uniform"], linestyle=":", alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.80),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        borderaxespad=0.15,
        handlelength=1.8,
        fontsize=6.8,
    )
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "quick_visibility_plus_standard_spectral"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--score-level", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--sigma-grid", type=str, default="0.5,1,2,3")
    ap.add_argument(
        "--fallback-sigma-grid",
        type=str,
        default="0.5,1,2,3",
        help="Extra sigma values tried automatically if requested grid cannot build a strong invisible set.",
    )
    ap.add_argument(
        "--max-inv-vis-ratio",
        type=float,
        default=0.30,
        help="Require |Vis_ref(H_inv)| <= ratio * |Vis_ref(H_vis)| after holdout construction.",
    )
    ap.add_argument(
        "--disable-auto-fallback",
        action="store_true",
        help="Disable automatic fallback to fallback-sigma-grid when invisible set is too visible.",
    )
    ap.add_argument("--std-holdout-mode", type=str, default="global", choices=["high_value", "global"])
    ap.add_argument("--Qmax", type=int, default=10000)
    ap.add_argument("--improve-steps", type=int, default=2000)
    args = ap.parse_args()

    vis.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sigma_grid = _parse_floats(args.sigma_grid) if str(args.sigma_grid).strip() else [1.0]
    built = _build_visible_invisible(
        n=args.n,
        beta=args.beta,
        good_frac=args.good_frac,
        score_level=args.score_level,
        holdout_k=args.holdout_k,
        train_m=args.train_m,
        K=args.K,
        seed=args.seed,
        improve_steps=args.improve_steps,
        sigma_grid=sigma_grid,
    )
    ratio_before = abs(float(built["Vis_inv_ref"])) / max(abs(float(built["Vis_vis_ref"])), 1e-12)
    fallback_used = False
    if (not args.disable_auto_fallback) and (ratio_before > float(args.max_inv_vis_ratio)):
        fallback = _parse_floats(args.fallback_sigma_grid) if str(args.fallback_sigma_grid).strip() else []
        merged = list(sigma_grid)
        for s in fallback:
            if all(abs(float(s) - float(t)) > 1e-12 for t in merged):
                merged.append(float(s))
        if len(merged) > len(sigma_grid):
            built_fb = _build_visible_invisible(
                n=args.n,
                beta=args.beta,
                good_frac=args.good_frac,
                score_level=args.score_level,
                holdout_k=args.holdout_k,
                train_m=args.train_m,
                K=args.K,
                seed=args.seed,
                improve_steps=args.improve_steps,
                sigma_grid=merged,
            )
            ratio_after = abs(float(built_fb["Vis_inv_ref"])) / max(abs(float(built_fb["Vis_vis_ref"])), 1e-12)
            if ratio_after < ratio_before:
                built = built_fb
                sigma_grid = merged
                fallback_used = True

    p_star = np.array(built["p_star"], dtype=np.float64)
    bits_table = np.array(built["bits_table"], dtype=np.int8)
    support = np.array(built["support"], dtype=bool)
    good_mask = np.array(built["good_mask"], dtype=bool)
    mask_vis = np.array(built["mask_vis"], dtype=bool)
    mask_inv = np.array(built["mask_inv"], dtype=bool)
    q_vis = np.array(built["q_vis"], dtype=np.float64)
    q_inv = np.array(built["q_inv"], dtype=np.float64)
    completion_for_holdout_idxs = built["completion_for_holdout_idxs"]  # callable

    # Standard spectral holdout baseline under the same feature family P.
    candidate_mask = support.astype(bool) if args.std_holdout_mode == "global" else good_mask
    mask_std = hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate_mask,
        bits_table=bits_table,
        m_train=args.train_m,
        holdout_k=args.holdout_k,
        pool_size=args.holdout_pool,
        seed=args.seed + 111,
    )
    H_std = np.where(mask_std)[0].astype(int)
    q_std, _, vis_std_train = completion_for_holdout_idxs(H_std, seed_offset=321)

    hv.save_holdout_list(mask_vis, bits_table, p_star, np.array(built["scores"]), str(outdir), name="holdout_strings_visible_plus_std.txt")
    hv.save_holdout_list(mask_inv, bits_table, p_star, np.array(built["scores"]), str(outdir), name="holdout_strings_invisible_plus_std.txt")
    hv.save_holdout_list(mask_std, bits_table, p_star, np.array(built["scores"]), str(outdir), name="holdout_strings_standard_plus_std.txt")

    plot_pdf = outdir / "6a_adversarial_curves_with_standard_spectral.pdf"
    plot_png = outdir / "6a_adversarial_curves_with_standard_spectral.png"
    _plot_curves(
        out_pdf=plot_pdf,
        out_png=plot_png,
        p_star=p_star,
        mask_vis=mask_vis,
        mask_inv=mask_inv,
        mask_std=mask_std,
        q_vis=q_vis,
        q_inv=q_inv,
        q_std=np.array(q_std, dtype=np.float64),
        qmax=args.Qmax,
        std_holdout_mode=args.std_holdout_mode,
    )

    q_unif = np.ones_like(p_star) / p_star.size
    qH_unif = float(q_unif[mask_vis].sum())
    summary: Dict[str, object] = {
        "config": {
            "n": args.n,
            "beta": args.beta,
            "train_m": args.train_m,
            "holdout_k": args.holdout_k,
            "holdout_pool": args.holdout_pool,
            "good_frac": args.good_frac,
            "score_level": args.score_level,
            "seed": args.seed,
            "K": args.K,
            "sigma_grid": sigma_grid,
            "sigma_used": float(built["sigma_used"]),
            "max_inv_vis_ratio": float(args.max_inv_vis_ratio),
            "auto_fallback_disabled": bool(args.disable_auto_fallback),
            "auto_fallback_used": bool(fallback_used),
            "std_holdout_mode": args.std_holdout_mode,
        },
        "holdout": {
            "size_vis": int(np.sum(mask_vis)),
            "size_inv": int(np.sum(mask_inv)),
            "size_std": int(np.sum(mask_std)),
            "pH_vis": float(p_star[mask_vis].sum()),
            "pH_inv": float(p_star[mask_inv].sum()),
            "pH_std": float(p_star[mask_std].sum()),
        },
        "visibility": {
            "Vis_ref_vis": float(built["Vis_vis_ref"]),
            "Vis_ref_inv": float(built["Vis_inv_ref"]),
            "Vis_train_vis": float(built["Vis_vis_train"]),
            "Vis_train_inv": float(built["Vis_inv_train"]),
            "Vis_train_std": float(vis_std_train),
        },
        "mass": {
            "qH_vis": float(q_vis[mask_vis].sum()),
            "qH_inv": float(q_inv[mask_inv].sum()),
            "qH_std": float(np.array(q_std, dtype=np.float64)[mask_std].sum()),
            "qH_unif": qH_unif,
        },
        "files": {
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
            "visible_holdout": str(outdir / "holdout_strings_visible_plus_std.txt"),
            "invisible_holdout": str(outdir / "holdout_strings_invisible_plus_std.txt"),
            "standard_holdout": str(outdir / "holdout_strings_standard_plus_std.txt"),
        },
    }
    with (outdir / "summary_with_standard_spectral.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {plot_pdf}")
    print(f"[Saved] {plot_png}")
    print(f"[Saved] {outdir / 'summary_with_standard_spectral.json'}")


if __name__ == "__main__":
    main()
