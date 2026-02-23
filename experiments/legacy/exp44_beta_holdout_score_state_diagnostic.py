#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused beta holdout diagnostic plot for score-vs-state recovery behavior.

Creates a 4-panel figure for a fixed configuration (default beta=0.8, global holdout,
low-data m=200, holdout_k=20, IQP parity):
  1) Target score-mass p*(S=s) vs training score-mass p_train(S=s)
  2) Holdout recovery R(Q): target vs IQP-parity vs uniform
  3) Holdout score/state diagnostics (score-mass match vs state coverage/concentration)
  4) Full-support score/state diagnostics (score-share + within-level state diversity)

Also writes reproducibility tables and a summary JSON with an explicit verdict:
  - "State-diverse Match"
  - "Score-hit but State-collapse"
  - "Missed holdout structure"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _beta_tag(beta: float) -> str:
    return f"{beta:.2f}".replace("-", "m").replace(".", "p")


def _parse_list_ints(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


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


def _build_seed_artifact(args: argparse.Namespace, seed: int) -> Dict[str, object]:
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
        seed=int(seed),
        protect_max_score=bool(int(args.protect_max_score)),
        dense_global_levels_only=bool(int(args.dense_global_levels_only)),
        dense_global_min_states_per_level=int(args.dense_global_min_states_per_level),
    )

    holdout_size = int(np.sum(holdout_mask))
    if holdout_size == 0:
        raise RuntimeError(f"Holdout is empty for seed={seed}; cannot run diagnostics.")

    p_train = p_star.copy()
    p_train[holdout_mask] = 0.0
    p_train /= float(np.sum(p_train))

    cfg = hv.Config(
        n=int(args.n),
        beta=float(args.beta),
        train_m=int(args.train_m),
        holdout_k=int(args.holdout_k),
        holdout_pool=int(args.holdout_pool),
        seed=int(seed),
        good_frac=float(args.good_frac),
        sigmas=[float(args.sigma)],
        Ks=[int(args.K)],
        Qmax=int(args.qmax),
        Q80_thr=float(args.q80_thr),
        Q80_search_max=int(args.q80_search_max),
        target_family="paper_even",
        adversarial=False,
        use_iqp=True,
        use_classical=False,
        iqp_steps=int(args.iqp_steps),
        iqp_lr=float(args.iqp_lr),
        iqp_eval_every=int(args.iqp_eval_every),
        iqp_layers=int(args.layers),
        iqp_loss="parity_mse",
        outdir=str(args.outdir),
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
    if not isinstance(q_iqp, np.ndarray):
        raise RuntimeError(f"IQP model distribution missing for seed={seed}.")

    return {
        "seed": int(seed),
        "bits_table": bits_table,
        "p_star": p_star,
        "support": support,
        "scores": scores,
        "good_mask": good_mask,
        "holdout_mask": holdout_mask,
        "p_train": p_train,
        "q_iqp": q_iqp,
    }


def _simulate_recovery_paths(
    q: np.ndarray,
    holdout_mask: np.ndarray,
    q_vals: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    qv = np.asarray(q, dtype=np.float64)
    qv = np.clip(qv, 0.0, None)
    s = float(np.sum(qv))
    if s <= 0.0:
        qv = np.ones_like(qv, dtype=np.float64) / qv.size
    else:
        qv = qv / s

    qgrid = np.unique(np.asarray(q_vals, dtype=int))
    if qgrid.size == 0:
        raise ValueError("q_vals is empty.")
    if int(np.min(qgrid)) < 0:
        raise ValueError("q_vals must be non-negative.")

    H_idx = np.where(np.asarray(holdout_mask, dtype=bool))[0]
    H = int(H_idx.size)
    out = np.zeros((int(repeats), int(qgrid.size)), dtype=np.float64)
    if H == 0:
        return out

    N = int(qv.size)
    lut = np.full(N, -1, dtype=np.int32)
    lut[H_idx] = np.arange(H, dtype=np.int32)

    nonzero_pos = np.where(qgrid > 0)[0]
    if nonzero_pos.size == 0:
        return out
    q_nonzero = qgrid[nonzero_pos]
    q_max = int(q_nonzero[-1])

    for r in range(int(repeats)):
        draws = rng.choice(N, size=q_max, replace=True, p=qv)
        seen = np.zeros(H, dtype=bool)
        seen_count = 0
        ptr = 0
        for t in range(1, q_max + 1):
            hp = int(lut[int(draws[t - 1])])
            if hp >= 0 and (not seen[hp]):
                seen[hp] = True
                seen_count += 1
            while ptr < q_nonzero.size and t >= int(q_nonzero[ptr]):
                out[r, int(nonzero_pos[ptr])] = float(seen_count) / float(H)
                ptr += 1
            if ptr >= q_nonzero.size:
                break
    return out


def _mc_mean_ci(samples: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("samples must be shape [n_paths, n_q].")
    lo_q = float(alpha / 2.0)
    hi_q = float(1.0 - alpha / 2.0)
    return {
        "mean": np.mean(arr, axis=0),
        "ci_lo": np.quantile(arr, lo_q, axis=0),
        "ci_hi": np.quantile(arr, hi_q, axis=0),
    }


def _build_holdout(
    holdout_mode: str,
    holdout_selection: str,
    p_star: np.ndarray,
    support: np.ndarray,
    scores: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    m_train_for_holdout: int,
    holdout_k: int,
    holdout_pool: int,
    seed: int,
    protect_max_score: bool,
    dense_global_levels_only: bool,
    dense_global_min_states_per_level: int,
) -> np.ndarray:
    if holdout_mode == "global":
        candidate = support.astype(bool)
        if dense_global_levels_only:
            s_int = scores.astype(np.int64)
            levels = np.sort(np.unique(s_int[support]))
            dense_levels: List[int] = []
            for lv in levels:
                cnt = int(np.sum(support & (s_int == int(lv))))
                if cnt >= int(dense_global_min_states_per_level):
                    dense_levels.append(int(lv))
            if dense_levels:
                dense_mask = support & np.isin(s_int, np.array(dense_levels, dtype=np.int64))
                if int(np.sum(dense_mask)) >= int(holdout_k):
                    candidate = dense_mask

        if protect_max_score:
            s_int = scores.astype(np.int64)
            max_score = int(np.max(s_int[support]))
            top_mask = support & (s_int == max_score)
            protected_count = int(np.sum(top_mask))
            if protected_count > 0:
                candidate_wo_top = candidate & (~top_mask)
                if int(np.sum(candidate_wo_top)) >= int(holdout_k):
                    candidate = candidate_wo_top
    elif holdout_mode == "high_value":
        candidate = good_mask
    else:
        raise ValueError(f"Unsupported holdout_mode: {holdout_mode}")

    sel = str(holdout_selection).strip().lower()
    if sel == "smart":
        return hv.select_holdout_smart(
            p_star=p_star,
            good_mask=candidate,
            bits_table=bits_table,
            m_train=m_train_for_holdout,
            holdout_k=holdout_k,
            pool_size=holdout_pool,
            seed=seed + 111,
        )
    if sel == "random":
        cand_idx = np.where(candidate)[0]
        if int(cand_idx.size) < int(holdout_k):
            raise RuntimeError(
                f"Random holdout failed: candidate size {int(cand_idx.size)} < holdout_k={int(holdout_k)}."
            )
        rng = np.random.default_rng(seed + 111)
        pick = rng.choice(cand_idx, size=int(holdout_k), replace=False)
        holdout = np.zeros_like(candidate, dtype=bool)
        holdout[pick] = True
        return holdout
    raise ValueError(f"Unsupported holdout_selection: {holdout_selection}")


def _score_levels(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s_int = scores.astype(np.int64)
    return np.sort(np.unique(s_int[mask]))


def score_mass_by_level(
    probs: np.ndarray,
    scores: np.ndarray,
    levels: np.ndarray,
    base_mask: np.ndarray,
) -> np.ndarray:
    s_int = scores.astype(np.int64)
    masses = np.zeros(levels.shape[0], dtype=np.float64)
    for i, lv in enumerate(levels):
        m = base_mask & (s_int == int(lv))
        masses[i] = float(np.sum(probs[m]))
    return masses


def build_score_mass_rows(
    p_star: np.ndarray,
    p_train: np.ndarray,
    q_model: np.ndarray,
    holdout_mask: np.ndarray,
    scores: np.ndarray,
) -> List[Dict[str, object]]:
    support_mask = p_star > 0.0
    levels = _score_levels(scores, support_mask)
    p_star_mass = score_mass_by_level(p_star, scores, levels, support_mask)
    p_train_mass = score_mass_by_level(p_train, scores, levels, support_mask)
    q_model_mass = score_mass_by_level(q_model, scores, levels, support_mask)

    rows: List[Dict[str, object]] = []
    s_int = scores.astype(np.int64)
    for i, lv in enumerate(levels):
        h_mask = holdout_mask & (s_int == int(lv))
        rows.append(
            {
                "score_level": int(lv),
                "p_star_mass": float(p_star_mass[i]),
                "p_train_mass": float(p_train_mass[i]),
                "q_model_mass": float(q_model_mass[i]),
                "removed_holdout_mass": float(np.sum(p_star[h_mask])),
                "holdout_state_count": int(np.sum(h_mask)),
            }
        )
    return rows


def _normalized_shares(mass: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    total = float(np.sum(mass))
    if total <= eps:
        return np.zeros_like(mass, dtype=np.float64)
    return mass / total


def holdout_score_diagnostics(
    p_star: np.ndarray,
    q_model: np.ndarray,
    holdout_mask: np.ndarray,
    scores: np.ndarray,
    qdiag: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    s_int = scores.astype(np.int64)
    levels = _score_levels(scores, holdout_mask)

    p_h = float(np.sum(p_star[holdout_mask]))
    q_h = float(np.sum(q_model[holdout_mask]))

    target_mass = np.zeros(levels.shape[0], dtype=np.float64)
    model_mass = np.zeros(levels.shape[0], dtype=np.float64)
    coverage = np.zeros(levels.shape[0], dtype=np.float64)
    dominance = np.zeros(levels.shape[0], dtype=np.float64)

    rows: List[Dict[str, object]] = []
    for i, lv in enumerate(levels):
        level_mask = holdout_mask & (s_int == int(lv))
        idx = np.where(level_mask)[0]
        target_mass[i] = float(np.sum(p_star[idx]))
        model_mass[i] = float(np.sum(q_model[idx]))

        cov = hv.expected_unique_fraction(q_model, level_mask, np.array([int(qdiag)], dtype=int))
        coverage[i] = float(cov[0]) if cov.size else 0.0

        if model_mass[i] > 0.0 and idx.size > 0:
            dominance[i] = float(np.max(q_model[idx]) / model_mass[i])
        else:
            dominance[i] = float("nan")

        rows.append(
            {
                "score_level": int(lv),
                "holdout_states": int(idx.size),
                "target_mass_level": float(target_mass[i]),
                "model_mass_level": float(model_mass[i]),
                "target_score_share": float(target_mass[i] / p_h) if p_h > 0.0 else 0.0,
                "model_score_share": float(model_mass[i] / q_h) if q_h > 0.0 else 0.0,
                "state_coverage_qdiag": float(coverage[i]),
                "dominance_in_level": float(dominance[i]),
                "max_state_prob_in_level": float(np.max(q_model[idx])) if idx.size > 0 else 0.0,
            }
        )

    target_share = _normalized_shares(target_mass)
    model_share = _normalized_shares(model_mass)
    score_tv = 0.5 * float(np.sum(np.abs(target_share - model_share)))

    dom_clean = np.nan_to_num(dominance, nan=1.0)
    w = model_share if float(np.sum(model_share)) > 0.0 else np.ones_like(model_share) / max(1, model_share.size)
    weighted_dom = float(np.sum(w * dom_clean))

    target_recovery_qdiag = float(hv.expected_unique_fraction(p_star, holdout_mask, np.array([int(qdiag)]))[0])
    model_recovery_qdiag = float(hv.expected_unique_fraction(q_model, holdout_mask, np.array([int(qdiag)]))[0])
    recovery_ratio = float(model_recovery_qdiag / max(1e-12, target_recovery_qdiag))

    min_level_coverage = float(np.min(coverage)) if coverage.size else 0.0

    summary = {
        "score_tv": float(score_tv),
        "weighted_dominance": float(weighted_dom),
        "holdout_recovery_qdiag": float(model_recovery_qdiag),
        "target_recovery_qdiag": float(target_recovery_qdiag),
        "recovery_ratio_vs_target": float(recovery_ratio),
        "min_level_coverage_qdiag": float(min_level_coverage),
        "p_star_holdout": float(p_h),
        "q_model_holdout": float(q_h),
    }
    return rows, summary


def full_support_score_diagnostics(
    p_star: np.ndarray,
    q_model: np.ndarray,
    support_mask: np.ndarray,
    scores: np.ndarray,
    qdiag: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    s_int = scores.astype(np.int64)
    levels = _score_levels(scores, support_mask)

    target_mass = score_mass_by_level(p_star, scores, levels, support_mask)
    model_mass_raw = score_mass_by_level(q_model, scores, levels, support_mask)

    target_share = _normalized_shares(target_mass)
    model_share = _normalized_shares(model_mass_raw)

    rows: List[Dict[str, object]] = []
    dominance = np.zeros(levels.shape[0], dtype=np.float64)
    neff_ratio = np.zeros(levels.shape[0], dtype=np.float64)
    support_hit = np.zeros(levels.shape[0], dtype=np.float64)

    for i, lv in enumerate(levels):
        level_mask = support_mask & (s_int == int(lv))
        idx = np.where(level_mask)[0]
        count = int(idx.size)

        q_level = q_model[idx]
        q_mass = float(np.sum(q_level))
        if q_mass > 0.0 and count > 0:
            p_level = np.clip(q_level / q_mass, 1e-15, 1.0)
            dominance[i] = float(np.max(p_level))
            ent = -float(np.sum(p_level * np.log(p_level)))
            neff = float(np.exp(ent))
            neff_ratio[i] = float(neff / count)
        else:
            dominance[i] = float("nan")
            neff = 0.0
            neff_ratio[i] = 0.0

        hit = hv.expected_unique_fraction(q_model, level_mask, np.array([int(qdiag)], dtype=int))
        support_hit[i] = float(hit[0]) if hit.size else 0.0

        rows.append(
            {
                "score_level": int(lv),
                "support_states": int(count),
                "target_score_share_support": float(target_share[i]),
                "model_score_share_support": float(model_share[i]),
                "support_hit_qdiag": float(support_hit[i]),
                "dominance_in_level_support": float(dominance[i]),
                "n_eff_support": float(neff),
                "n_eff_ratio_support": float(neff_ratio[i]),
            }
        )

    score_tv = 0.5 * float(np.sum(np.abs(target_share - model_share)))
    dom_clean = np.nan_to_num(dominance, nan=1.0)
    weighted_dom = float(np.sum(model_share * dom_clean))
    weighted_neff_ratio = float(np.sum(model_share * neff_ratio))
    weighted_support_hit = float(np.sum(target_share * support_hit))

    summary = {
        "support_score_tv": float(score_tv),
        "support_weighted_dominance": float(weighted_dom),
        "support_weighted_neff_ratio": float(weighted_neff_ratio),
        "support_weighted_hit_qdiag": float(weighted_support_hit),
    }
    return rows, summary


def classify_holdout_learning(
    diag_rows: Sequence[Dict[str, object]],
    summary: Dict[str, float],
) -> Tuple[str, str]:
    if len(diag_rows) == 0:
        return "Missed holdout structure", "No holdout score levels available for diagnosis."

    score_tv = float(summary["score_tv"])
    weighted_dom = float(summary["weighted_dominance"])
    rec_ratio = float(summary["recovery_ratio_vs_target"])
    min_cov = float(summary["min_level_coverage_qdiag"])

    dominant_row = max(diag_rows, key=lambda r: float(r["dominance_in_level"]))
    dom_level = int(dominant_row["score_level"])
    dom_val = float(dominant_row["dominance_in_level"])

    score_match = score_tv <= 0.12
    strong_coverage = (rec_ratio >= 0.75) and (min_cov >= 0.45)
    collapse = weighted_dom >= 0.70

    if score_match and strong_coverage and weighted_dom <= 0.60:
        verdict = "State-diverse Match"
        reason = (
            "Score-level shares are close to target and recovery is supported by broad state-level coverage, "
            "not a single dominant state."
        )
    elif score_match and collapse:
        verdict = "Score-hit but State-collapse"
        reason = (
            f"Score-level shares look matched, but sampling is concentrated (highest dominance at s={dom_level}: "
            f"{dom_val:.2f}), indicating few states carry the score match."
        )
    else:
        verdict = "Missed holdout structure"
        reason = (
            "Model does not reproduce holdout score/state structure: score-share mismatch and/or insufficient "
            "state-level recovery across holdout levels."
        )

    return verdict, reason


def _plot_panel_a_score_mass(
    ax,
    rows: Sequence[Dict[str, object]],
    legend_outside_right: bool = False,
) -> None:
    levels = np.array([int(r["score_level"]) for r in rows], dtype=int)
    p_star_mass = np.array([float(r["p_star_mass"]) for r in rows], dtype=np.float64)
    p_train_mass = np.array([float(r["p_train_mass"]) for r in rows], dtype=np.float64)
    q_model_mass = np.array([float(r["q_model_mass"]) for r in rows], dtype=np.float64)
    removed = np.array([float(r["removed_holdout_mass"]) for r in rows], dtype=np.float64)

    x = np.arange(levels.size)
    w = 0.24
    ax.bar(x - w, p_star_mass, width=w, color=hv.COLORS["target"], alpha=0.8, label=r"$p^*(S=s)$")
    ax.bar(x, p_train_mass, width=w, color=hv.COLORS["gray"], alpha=0.85, label=r"$p_{\mathrm{train}}(S=s)$, Holdout entfernt")
    ax.bar(x + w, q_model_mass, width=w, color=hv.COLORS["model"], alpha=0.65, label=r"$q_{\mathrm{IQP}}(S=s)$")
    ax.plot(x, removed, color="#1F77B4", marker="o", linewidth=1.4, label="Removed holdout mass")

    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Score level s")
    ax.set_ylabel("Mass")
    ax.set_title("Target vs train score-mass")
    if legend_outside_right:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=6.4,
            frameon=True,
        )
    else:
        ax.legend(loc="upper right", fontsize=6.8, frameon=True)


def _plot_panel_b_recovery(
    ax,
    p_star: np.ndarray,
    q_iqp: np.ndarray,
    holdout_mask: np.ndarray,
    qmax: int,
) -> Dict[str, float]:
    q_vals = _q_grid(qmax)
    y_star = hv.expected_unique_fraction(p_star, holdout_mask, q_vals)
    y_iqp = hv.expected_unique_fraction(q_iqp, holdout_mask, q_vals)
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, q_vals)

    ax.plot(q_vals, y_star, color=hv.COLORS["target"], linewidth=2.0, label=r"Target $p^*$")
    ax.plot(q_vals, y_iqp, color=hv.COLORS["model"], linewidth=2.2, label="IQP (parity)")
    ax.plot(q_vals, y_unif, color=hv.COLORS["gray"], linewidth=1.5, linestyle="--", label="Uniform")
    ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7)

    ax.set_xlim(0, qmax)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_title("Holdout recovery")
    ax.legend(loc="lower right", fontsize=6.8, frameon=True)

    q80_iqp = float(hv.find_Q_threshold(q_iqp, holdout_mask, thr=0.8, Qmax=max(200000, qmax)))
    q80_target = float(hv.find_Q_threshold(p_star, holdout_mask, thr=0.8, Qmax=max(200000, qmax)))
    q80_unif = float(hv.find_Q_threshold(q_unif, holdout_mask, thr=0.8, Qmax=max(200000, qmax)))
    return {"Q80_iqp": q80_iqp, "Q80_target": q80_target, "Q80_uniform": q80_unif}


def _plot_panel_c_recovery_mc(
    ax,
    q_vals: np.ndarray,
    curves: Dict[str, Dict[str, np.ndarray]],
    qmax: int,
    n_paths: int,
    n_seeds: int,
    repeats: int,
    iqp_sample_paths: np.ndarray | None = None,
) -> None:
    style = {
        "target": {"label": r"Target $p^*$", "color": hv.COLORS["target"], "ls": "-", "lw": 1.9},
        "iqp": {"label": "IQP (parity)", "color": hv.COLORS["model"], "ls": "-", "lw": 2.1},
        "uniform": {"label": "Uniform", "color": hv.COLORS["gray"], "ls": "--", "lw": 1.5},
    }
    for key in ("target", "iqp", "uniform"):
        c = curves[key]
        st = style[key]
        x = np.asarray(q_vals, dtype=np.float64)
        mean = np.asarray(c["mean"], dtype=np.float64)
        lo = np.asarray(c["ci_lo"], dtype=np.float64)
        hi = np.asarray(c["ci_hi"], dtype=np.float64)

        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.16, linewidth=0.0)
        ax.plot(x, mean, color=st["color"], linestyle=st["ls"], linewidth=st["lw"], label=st["label"])

    if iqp_sample_paths is not None:
        paths = np.asarray(iqp_sample_paths, dtype=np.float64)
        if paths.ndim == 2 and paths.shape[1] == len(q_vals) and paths.shape[0] > 0:
            for i in range(paths.shape[0]):
                lbl = f"IQP sample paths (n={int(paths.shape[0])})" if i == 0 else None
                ax.plot(
                    np.asarray(q_vals, dtype=np.float64),
                    paths[i],
                    color=hv.COLORS["model"],
                    linewidth=0.7,
                    alpha=0.20,
                    label=lbl,
                    zorder=1,
                )

    ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlim(0, qmax)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_title("Holdout recovery (Monte Carlo + 95% CI)")
    ax.legend(loc="lower right", fontsize=6.8, frameon=True)
    ax.text(
        0.03,
        0.03,
        f"paths={int(n_paths)} ({int(n_seeds)} seeds x {int(repeats)} repeats)",
        transform=ax.transAxes,
        fontsize=6.2,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.6),
    )


def _plot_panel_c_diagnostic(
    ax,
    diag_rows: Sequence[Dict[str, object]],
    verdict: str,
    qdiag: int,
) -> None:
    levels = np.array([int(r["score_level"]) for r in diag_rows], dtype=int)
    target_share = np.array([float(r["target_score_share"]) for r in diag_rows], dtype=np.float64)
    model_share = np.array([float(r["model_score_share"]) for r in diag_rows], dtype=np.float64)
    coverage = np.array([float(r["state_coverage_qdiag"]) for r in diag_rows], dtype=np.float64)
    dominance = np.array([float(r["dominance_in_level"]) for r in diag_rows], dtype=np.float64)

    x = np.arange(levels.size)
    w = 0.36

    ax.bar(x - 0.5 * w, target_share, width=w, color=hv.COLORS["target"], alpha=0.70, label="Target share in holdout")
    ax.bar(x + 0.5 * w, model_share, width=w, color=hv.COLORS["model"], alpha=0.70, label="Model share in holdout")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Score level s")
    ax.set_ylabel("Score share inside holdout")

    ax2 = ax.twinx()
    ax2.plot(x, coverage, color="#1F77B4", marker="o", linewidth=1.4, label=fr"State coverage @Q={qdiag}")
    ax2.plot(x, dominance, color="#2CA02C", marker="s", linewidth=1.3, linestyle="--", label="Dominance in level")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_ylabel("Coverage / Dominance")

    lines = [
        Line2D([0], [0], color=hv.COLORS["target"], linewidth=6, alpha=0.70, label="Target share in holdout"),
        Line2D([0], [0], color=hv.COLORS["model"], linewidth=6, alpha=0.70, label="Model share in holdout"),
        Line2D([0], [0], color="#1F77B4", marker="o", linewidth=1.4, label=fr"State coverage @Q={qdiag}"),
        Line2D([0], [0], color="#2CA02C", marker="s", linewidth=1.3, linestyle="--", label="Dominance in level"),
    ]
    ax.legend(handles=lines, loc="upper left", fontsize=6.5, frameon=True)
    ax.set_title(f"Score-level diagnosis ({verdict})")


def _plot_panel_d_full_support(
    ax,
    support_rows: Sequence[Dict[str, object]],
    support_summary: Dict[str, float],
    qdiag: int,
) -> None:
    levels = np.array([int(r["score_level"]) for r in support_rows], dtype=int)
    target_share = np.array([float(r["target_score_share_support"]) for r in support_rows], dtype=np.float64)
    model_share = np.array([float(r["model_score_share_support"]) for r in support_rows], dtype=np.float64)
    support_hit = np.array([float(r["support_hit_qdiag"]) for r in support_rows], dtype=np.float64)
    neff_ratio = np.array([float(r["n_eff_ratio_support"]) for r in support_rows], dtype=np.float64)

    x = np.arange(levels.size)
    w = 0.36
    ax.bar(x - 0.5 * w, target_share, width=w, color=hv.COLORS["target"], alpha=0.70, label="Target share in support")
    ax.bar(x + 0.5 * w, model_share, width=w, color=hv.COLORS["model"], alpha=0.70, label="Model share in support")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Score level s")
    ax.set_ylabel("Score share inside support")

    ax2 = ax.twinx()
    ax2.plot(x, support_hit, color="#2CA02C", marker="s", linewidth=1.5, linestyle="-", label=fr"Support hit @Q={qdiag}")
    ax2.plot(x, neff_ratio, color="#1F77B4", marker="o", linewidth=1.4, label=r"$N_{\mathrm{eff}}/|A_s|$")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_ylabel("Diversity metrics")

    lines = [
        Line2D([0], [0], color=hv.COLORS["target"], linewidth=6, alpha=0.70, label="Target share in support"),
        Line2D([0], [0], color=hv.COLORS["model"], linewidth=6, alpha=0.70, label="Model share in support"),
        Line2D([0], [0], color="#2CA02C", marker="s", linewidth=1.5, linestyle="-", label=fr"Support hit @Q={qdiag}"),
        Line2D([0], [0], color="#1F77B4", marker="o", linewidth=1.4, label=r"$N_{\mathrm{eff}}/|A_s|$"),
    ]
    ax.legend(handles=lines, loc="upper left", fontsize=6.2, frameon=True)
    ax.set_title(
        "Full-support state diversity\n"
        + f"TV={support_summary['support_score_tv']:.3f}, "
        + f"w-hit={support_summary['support_weighted_hit_qdiag']:.3f}, "
        + f"w-neff={support_summary['support_weighted_neff_ratio']:.3f}"
    )


def run(args: argparse.Namespace) -> None:
    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP training. Install with `pip install pennylane`.")

    hv.set_style(base=8)

    outdir = _ensure_outdir(Path(args.outdir))
    primary = _build_seed_artifact(args=args, seed=int(args.seed))
    p_star = np.asarray(primary["p_star"], dtype=np.float64)
    support = np.asarray(primary["support"], dtype=bool)
    scores = np.asarray(primary["scores"], dtype=np.float64)
    holdout_mask = np.asarray(primary["holdout_mask"], dtype=bool)
    p_train = np.asarray(primary["p_train"], dtype=np.float64)
    q_iqp = np.asarray(primary["q_iqp"], dtype=np.float64)
    holdout_size = int(np.sum(holdout_mask))
    if holdout_size == 0:
        raise RuntimeError("Holdout is empty; cannot run holdout diagnostics.")

    q80_iqp = float(hv.find_Q_threshold(q_iqp, holdout_mask, thr=float(args.q80_thr), Qmax=int(args.q80_search_max)))
    if args.qdiag is not None:
        qdiag = int(args.qdiag)
    elif np.isfinite(q80_iqp):
        qdiag = max(1, min(int(math.ceil(float(q80_iqp))), int(args.qmax)))
    else:
        qdiag = int(args.qmax)

    score_mass_rows = build_score_mass_rows(
        p_star=p_star,
        p_train=p_train,
        q_model=q_iqp,
        holdout_mask=holdout_mask,
        scores=scores,
    )
    diag_rows, diag_summary = holdout_score_diagnostics(
        p_star=p_star,
        q_model=q_iqp,
        holdout_mask=holdout_mask,
        scores=scores,
        qdiag=qdiag,
    )
    support_rows, support_summary = full_support_score_diagnostics(
        p_star=p_star,
        q_model=q_iqp,
        support_mask=support,
        scores=scores,
        qdiag=qdiag,
    )
    verdict, reason = classify_holdout_learning(diag_rows=diag_rows, summary=diag_summary)

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.15), constrained_layout=False)
    _plot_panel_a_score_mass(axes[0], score_mass_rows)
    q80_pack = _plot_panel_b_recovery(axes[1], p_star=p_star, q_iqp=q_iqp, holdout_mask=holdout_mask, qmax=int(args.qmax))
    _plot_panel_c_diagnostic(axes[2], diag_rows, verdict, qdiag=qdiag)
    _plot_panel_d_full_support(axes[3], support_rows=support_rows, support_summary=support_summary, qdiag=qdiag)

    fig_title = (
        fr"$\beta$={args.beta:g}, holdout={args.holdout_mode}, $m$={int(args.train_m)}, "
        fr"$|H|$={int(args.holdout_k)}, IQP parity"
    )
    if str(args.holdout_selection).strip().lower() == "random":
        fig_title += ", selection=random"
    if bool(int(args.protect_max_score)) and str(args.holdout_mode) == "global":
        fig_title += r", protected max-score state"
    if bool(int(args.dense_global_levels_only)) and str(args.holdout_mode) == "global":
        fig_title += fr", dense levels only (>= {int(args.dense_global_min_states_per_level)} states)"
    fig.suptitle(fig_title, y=0.96)
    fig.subplots_adjust(left=0.045, right=0.988, top=0.86, bottom=0.22, wspace=0.32)

    footer = (
        f"Verdict: {verdict} | score TV={diag_summary['score_tv']:.3f}, "
        f"weighted dominance={diag_summary['weighted_dominance']:.3f}, "
        f"recovery ratio vs target={diag_summary['recovery_ratio_vs_target']:.2f}. "
        f"Interpretation: {reason}"
    )
    fig.text(
        0.056,
        0.08,
        textwrap.fill(footer, width=170),
        ha="left",
        va="center",
        fontsize=7.3,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d0d0d0", alpha=0.95),
    )

    btag = _beta_tag(float(args.beta))
    sel_tag = str(args.holdout_selection).strip().lower()
    legacy_base_prefix = f"beta_{btag}_{args.holdout_mode}_m{int(args.train_m)}_holdout{int(args.holdout_k)}"
    sel_prefix = f"beta_{btag}_{args.holdout_mode}_{sel_tag}_m{int(args.train_m)}_holdout{int(args.holdout_k)}"
    prefix = legacy_base_prefix if sel_tag == "smart" else sel_prefix

    plot_pdf = outdir / f"{prefix}_4panel.pdf"
    plot_png = outdir / f"{prefix}_4panel.png"
    legacy_plot_pdf = outdir / f"{prefix}_3panel.pdf"
    legacy_plot_png = outdir / f"{prefix}_3panel.png"

    fig.savefig(plot_pdf)
    fig.savefig(plot_png, dpi=280)
    # keep legacy filenames for convenience/compatibility
    fig.savefig(legacy_plot_pdf)
    fig.savefig(legacy_plot_png, dpi=280)
    if sel_tag == "smart" and sel_prefix != prefix:
        fig.savefig(outdir / f"{sel_prefix}_4panel.pdf")
        fig.savefig(outdir / f"{sel_prefix}_4panel.png", dpi=280)
        fig.savefig(outdir / f"{sel_prefix}_3panel.pdf")
        fig.savefig(outdir / f"{sel_prefix}_3panel.png", dpi=280)
    plt.close(fig)

    mc_csv = None
    mc_summary_json = None
    mc_plot_pdf = None
    mc_plot_png = None
    if bool(int(args.make_3panel_mc)):
        q_vals_mc = _q_grid(int(args.qmax)) if bool(int(args.mc_use_same_q_grid)) else _q_grid(min(5000, int(args.qmax)))
        mc_seeds = _parse_list_ints(str(args.mc_seeds))
        if len(mc_seeds) == 0:
            mc_seeds = [int(args.seed)]
        repeats = int(args.mc_repeats)
        if repeats <= 0:
            raise ValueError("--mc-repeats must be > 0.")
        alpha = float(args.mc_ci_alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("--mc-ci-alpha must be in (0,1).")

        seed_artifacts: Dict[int, Dict[str, object]] = {int(args.seed): primary}
        for sd in mc_seeds:
            if int(sd) not in seed_artifacts:
                seed_artifacts[int(sd)] = _build_seed_artifact(args=args, seed=int(sd))

        paths_target: List[np.ndarray] = []
        paths_iqp: List[np.ndarray] = []
        paths_uniform: List[np.ndarray] = []

        for sd in mc_seeds:
            seed_art = seed_artifacts[int(sd)]
            p_s = np.asarray(seed_art["p_star"], dtype=np.float64)
            h_s = np.asarray(seed_art["holdout_mask"], dtype=bool)
            q_s = np.asarray(seed_art["q_iqp"], dtype=np.float64)
            u_s = np.ones_like(p_s, dtype=np.float64) / p_s.size

            paths_target.append(
                _simulate_recovery_paths(
                    q=p_s,
                    holdout_mask=h_s,
                    q_vals=q_vals_mc,
                    repeats=repeats,
                    rng=np.random.default_rng(200000 + 13 * int(sd)),
                )
            )
            paths_iqp.append(
                _simulate_recovery_paths(
                    q=q_s,
                    holdout_mask=h_s,
                    q_vals=q_vals_mc,
                    repeats=repeats,
                    rng=np.random.default_rng(300000 + 17 * int(sd)),
                )
            )
            paths_uniform.append(
                _simulate_recovery_paths(
                    q=u_s,
                    holdout_mask=h_s,
                    q_vals=q_vals_mc,
                    repeats=repeats,
                    rng=np.random.default_rng(400000 + 19 * int(sd)),
                )
            )

        target_all = np.concatenate(paths_target, axis=0)
        iqp_all = np.concatenate(paths_iqp, axis=0)
        uniform_all = np.concatenate(paths_uniform, axis=0)
        curves = {
            "target": _mc_mean_ci(target_all, alpha=alpha),
            "iqp": _mc_mean_ci(iqp_all, alpha=alpha),
            "uniform": _mc_mean_ci(uniform_all, alpha=alpha),
        }

        iqp_sample_paths = None
        n_overlay = max(0, int(args.mc_overlay_paths))
        if n_overlay > 0:
            k = min(n_overlay, int(iqp_all.shape[0]))
            idx = np.random.default_rng(500000 + int(args.seed)).choice(int(iqp_all.shape[0]), size=k, replace=False)
            iqp_sample_paths = iqp_all[idx]

        fig3, axes3 = plt.subplots(1, 3, figsize=(15.6, 4.2), constrained_layout=False)
        _plot_panel_a_score_mass(axes3[0], score_mass_rows, legend_outside_right=True)
        _plot_panel_b_recovery(axes3[1], p_star=p_star, q_iqp=q_iqp, holdout_mask=holdout_mask, qmax=int(args.qmax))
        _plot_panel_c_recovery_mc(
            axes3[2],
            q_vals=q_vals_mc,
            curves=curves,
            qmax=int(args.qmax),
            n_paths=int(target_all.shape[0]),
            n_seeds=int(len(mc_seeds)),
            repeats=int(repeats),
            iqp_sample_paths=iqp_sample_paths,
        )
        fig3_title = (
            fr"$\beta$={args.beta:g}, holdout={args.holdout_mode}, $m$={int(args.train_m)}, "
            fr"$|H|$={int(args.holdout_k)}, IQP parity"
        )
        if sel_tag == "random":
            fig3_title += ", selection=random"
        fig3.suptitle(fig3_title, y=0.97)
        fig3.subplots_adjust(left=0.05, right=0.985, top=0.84, bottom=0.16, wspace=0.52)

        mc_plot_pdf = outdir / f"{prefix}_3panel_mc.pdf"
        mc_plot_png = outdir / f"{prefix}_3panel_mc.png"
        fig3.savefig(mc_plot_pdf)
        fig3.savefig(mc_plot_png, dpi=280)
        if sel_tag == "smart" and sel_prefix != prefix:
            fig3.savefig(outdir / f"{sel_prefix}_3panel_mc.pdf")
            fig3.savefig(outdir / f"{sel_prefix}_3panel_mc.png", dpi=280)
        plt.close(fig3)

        if bool(int(args.mc_save_csv)):
            rows_mc: List[Dict[str, object]] = []
            for model_key in ("target", "iqp", "uniform"):
                c = curves[model_key]
                for i, qv in enumerate(np.asarray(q_vals_mc, dtype=int).tolist()):
                    rows_mc.append(
                        {
                            "model": str(model_key),
                            "Q": int(qv),
                            "recovery_mean": float(c["mean"][i]),
                            "recovery_ci_lo": float(c["ci_lo"][i]),
                            "recovery_ci_hi": float(c["ci_hi"][i]),
                            "n_paths": int(target_all.shape[0]),
                            "n_seeds": int(len(mc_seeds)),
                            "repeats_per_seed": int(repeats),
                            "ci_alpha": float(alpha),
                        }
                    )
            mc_csv = outdir / f"beta_{btag}_recovery_mc_curves.csv"
            _write_csv(mc_csv, rows_mc)
            mc_summary_json = outdir / f"beta_{btag}_recovery_mc_config.json"
            with mc_summary_json.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mc_seeds": [int(x) for x in mc_seeds],
                        "mc_repeats": int(repeats),
                        "mc_ci_alpha": float(alpha),
                        "mc_use_same_q_grid": bool(int(args.mc_use_same_q_grid)),
                        "q_grid_size": int(len(q_vals_mc)),
                        "q_grid_max": int(np.max(q_vals_mc)),
                        "paths_per_model": int(target_all.shape[0]),
                        "mc_overlay_paths": int(args.mc_overlay_paths),
                        "plot_pdf": str(mc_plot_pdf) if mc_plot_pdf is not None else None,
                        "plot_png": str(mc_plot_png) if mc_plot_png is not None else None,
                    },
                    f,
                    indent=2,
                )

    score_csv = outdir / f"beta_{btag}_score_mass_target_vs_train.csv"
    diag_csv = outdir / f"beta_{btag}_holdout_score_state_diagnostics.csv"
    support_diag_csv = outdir / f"beta_{btag}_support_score_state_diagnostics.csv"
    summary_json = outdir / f"beta_{btag}_summary.json"

    _write_csv(score_csv, score_mass_rows)
    _write_csv(diag_csv, list(diag_rows))
    _write_csv(support_diag_csv, list(support_rows))

    summary = {
        "config": {
            "n": int(args.n),
            "beta": float(args.beta),
            "holdout_mode": str(args.holdout_mode),
            "holdout_selection": str(args.holdout_selection),
            "train_m": int(args.train_m),
            "holdout_k": int(args.holdout_k),
            "holdout_pool": int(args.holdout_pool),
            "holdout_m_train": int(args.holdout_m_train),
            "seed": int(args.seed),
            "protect_max_score": bool(int(args.protect_max_score)),
            "dense_global_levels_only": bool(int(args.dense_global_levels_only)),
            "dense_global_min_states_per_level": int(args.dense_global_min_states_per_level),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "layers": int(args.layers),
            "qmax": int(args.qmax),
            "qdiag": int(qdiag),
        },
        "holdout": {
            "size": int(holdout_size),
            "p_star_holdout": float(np.sum(p_star[holdout_mask])),
            "q_iqp_holdout": float(np.sum(q_iqp[holdout_mask])),
        },
        "recovery": {
            "Q80_iqp": float(q80_pack["Q80_iqp"]),
            "Q80_target": float(q80_pack["Q80_target"]),
            "Q80_uniform": float(q80_pack["Q80_uniform"]),
        },
        "diagnostic_summary": {k: float(v) for k, v in diag_summary.items()},
        "support_diagnostic_summary": {k: float(v) for k, v in support_summary.items()},
        "verdict": str(verdict),
        "reason": str(reason),
        "outputs": {
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
            "legacy_plot_pdf": str(legacy_plot_pdf),
            "legacy_plot_png": str(legacy_plot_png),
            "score_mass_csv": str(score_csv),
            "diagnostics_csv": str(diag_csv),
            "support_diagnostics_csv": str(support_diag_csv),
            "plot_3panel_mc_pdf": str(mc_plot_pdf) if mc_plot_pdf is not None else None,
            "plot_3panel_mc_png": str(mc_plot_png) if mc_plot_png is not None else None,
            "recovery_mc_csv": str(mc_csv) if mc_csv is not None else None,
            "recovery_mc_config_json": str(mc_summary_json) if mc_summary_json is not None else None,
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[saved] {plot_pdf}")
    print(f"[saved] {plot_png}")
    print(f"[saved] {legacy_plot_pdf}")
    print(f"[saved] {legacy_plot_png}")
    print(f"[saved] {score_csv}")
    print(f"[saved] {diag_csv}")
    print(f"[saved] {support_diag_csv}")
    if mc_plot_pdf is not None:
        print(f"[saved] {mc_plot_pdf}")
    if mc_plot_png is not None:
        print(f"[saved] {mc_plot_png}")
    if mc_csv is not None:
        print(f"[saved] {mc_csv}")
    if mc_summary_json is not None:
        print(f"[saved] {mc_summary_json}")
    print(f"[saved] {summary_json}")
    print(f"[verdict] {verdict}")
    print(f"[reason] {reason}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Beta holdout score/state diagnostic (4-panel figure + optional 3-panel MC recovery figure)."
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "44_beta_holdout_score_state_diagnostic"),
    )

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--holdout-selection", type=str, default="smart", choices=["smart", "random"])
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--protect-max-score", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dense-global-levels-only", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dense-global-min-states-per-level", type=int, default=64)
    ap.add_argument("--seed", type=int, default=46)

    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=400)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)

    ap.add_argument("--qmax", type=int, default=10000)
    ap.add_argument("--qdiag", type=int, default=None)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)

    ap.add_argument("--make-3panel-mc", type=int, default=0, choices=[0, 1])
    ap.add_argument("--mc-seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--mc-repeats", type=int, default=300)
    ap.add_argument("--mc-ci-alpha", type=float, default=0.05)
    ap.add_argument("--mc-save-csv", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mc-use-same-q-grid", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mc-overlay-paths", type=int, default=0)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
