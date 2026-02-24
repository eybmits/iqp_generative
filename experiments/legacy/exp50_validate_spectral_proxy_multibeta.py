#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 50: Validate spectral Q80 as a proxy for IQP Q80 over betas and seeds.

Main output:
- 3-panel summary plot
- long table over (beta, seed, sigma, K)
- replicate-level summary (one row per beta/seed)
- aggregate summary JSON with claim metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402
from experiments.legacy.exp44_beta_holdout_score_state_diagnostic import _build_holdout  # noqa: E402


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _beta_tag(beta: float) -> str:
    return f"{beta:.2f}".replace("-", "m").replace(".", "p")


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


def _settings_grid(sigmas: Sequence[float], Ks: Sequence[int]) -> List[Tuple[float, int]]:
    pairs: List[Tuple[float, int]] = []
    for s in sigmas:
        for k in Ks:
            pairs.append((float(s), int(k)))
    return pairs


def _bootstrap_ci(
    values: Sequence[float],
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"center": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    center = float(np.median(arr))
    if arr.size == 1 or n_boot <= 0:
        return {"center": center, "ci_lo": center, "ci_hi": center}
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(n_boot), dtype=np.float64)
    n = int(arr.size)
    for i in range(int(n_boot)):
        pick = rng.choice(arr, size=n, replace=True)
        boots[i] = float(np.median(pick))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return {"center": center, "ci_lo": lo, "ci_hi": hi}


def _spearman_with_permutation(
    x: Sequence[float],
    y: Sequence[float],
    perm_repeats: int = 0,
    seed: int = 0,
) -> Dict[str, float]:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    out = {
        "rho": float("nan"),
        "p_value": float("nan"),
        "p_perm": float("nan"),
        "n_pairs": int(xa.size),
    }
    if xa.size < 3:
        return out
    if np.allclose(xa, xa[0]) or np.allclose(ya, ya[0]):
        out["rho"] = 0.0
        out["p_value"] = 1.0
        out["p_perm"] = 1.0 if int(perm_repeats) > 0 else float("nan")
        return out
    rho, p_value = spearmanr(xa, ya)
    if not np.isfinite(rho):
        out["rho"] = 0.0
        out["p_value"] = 1.0
        out["p_perm"] = 1.0 if int(perm_repeats) > 0 else float("nan")
        return out
    out["rho"] = float(rho)
    out["p_value"] = float(p_value) if np.isfinite(p_value) else float("nan")
    if int(perm_repeats) > 0 and np.isfinite(rho):
        rng = np.random.default_rng(int(seed))
        cnt = 0
        abs_r = abs(float(rho))
        for _ in range(int(perm_repeats)):
            yperm = rng.permutation(ya)
            r_perm, _ = spearmanr(xa, yperm)
            if np.isfinite(r_perm) and abs(float(r_perm)) >= abs_r:
                cnt += 1
        out["p_perm"] = float((cnt + 1) / (int(perm_repeats) + 1))
    return out


def _pick_best_worst_by_q80_spec(rows: Sequence[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, object]]:
    if len(rows) == 0:
        raise ValueError("rows must be non-empty")
    finite = [r for r in rows if np.isfinite(float(r["Q80_spec"]))]
    if len(finite) >= 2:
        return (
            min(finite, key=lambda r: float(r["Q80_spec"])),
            max(finite, key=lambda r: float(r["Q80_spec"])),
        )
    if len(finite) == 1:
        best = finite[0]
        worst = max(
            rows,
            key=lambda r: float(r["Q80_spec"]) if np.isfinite(float(r["Q80_spec"])) else float("inf"),
        )
        return best, worst
    # Fallback when every Q80 is non-finite.
    return (
        max(rows, key=lambda r: float(r["qH_ratio_spec"])),
        min(rows, key=lambda r: float(r["qH_ratio_spec"])),
    )


def _build_cfg(args: argparse.Namespace, seed: int, beta: float, sigma: float, K: int) -> hv.Config:
    return hv.Config(
        n=int(args.n),
        beta=float(beta),
        train_m=int(args.train_m),
        holdout_k=int(args.holdout_k),
        holdout_pool=int(args.holdout_pool),
        seed=int(seed),
        good_frac=float(args.good_frac),
        sigmas=[float(sigma)],
        Ks=[int(K)],
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


def _iter_groups(rows: Sequence[Dict[str, object]], key: str) -> Iterable[Tuple[object, List[Dict[str, object]]]]:
    by_key: Dict[object, List[Dict[str, object]]] = {}
    for r in rows:
        by_key.setdefault(r[key], []).append(dict(r))
    for k in sorted(by_key.keys()):
        yield k, by_key[k]


def run(args: argparse.Namespace) -> None:
    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP training. Install with `pip install pennylane`.")

    hv.set_style(base=8)
    outdir = _ensure_outdir(Path(args.outdir))

    betas = _parse_list_floats(str(args.betas))
    seeds = _parse_list_ints(str(args.seeds))
    sigmas = _parse_list_floats(str(args.sigmas))
    Ks = _parse_list_ints(str(args.Ks))
    settings = _settings_grid(sigmas=sigmas, Ks=Ks)
    if len(settings) < 2:
        raise RuntimeError("Need at least two (sigma,K) settings.")

    bits_table = hv.make_bits_table(int(args.n))

    long_rows: List[Dict[str, object]] = []
    replicate_rows: List[Dict[str, object]] = []

    total_reps = len(betas) * len(seeds)
    rep_idx = 0
    for beta in betas:
        p_star, support, scores = hv.build_target_distribution_paper(int(args.n), float(beta))
        good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))

        for seed in seeds:
            rep_idx += 1
            print(f"[rep {rep_idx}/{total_reps}] beta={beta:g}, seed={seed}")
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
            H_size = int(np.sum(holdout_mask))
            if H_size <= 0:
                raise RuntimeError(f"Holdout is empty for beta={beta}, seed={seed}.")
            q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
            qH_unif = float(np.sum(q_unif[holdout_mask]))

            rows_rep: List[Dict[str, object]] = []
            for sigma, K in settings:
                cfg = _build_cfg(
                    args=args,
                    seed=int(seed),
                    beta=float(beta),
                    sigma=float(sigma),
                    K=int(K),
                )

                art = hv.rerun_single_setting(
                    cfg=cfg,
                    p_star=p_star,
                    holdout_mask=holdout_mask,
                    bits_table=bits_table,
                    sigma=float(sigma),
                    K=int(K),
                    return_hist=False,
                    iqp_loss="parity_mse",
                )

                q_spec = np.asarray(art["q_spec"], dtype=np.float64)
                q_iqp = art["q_iqp"]
                if not isinstance(q_iqp, np.ndarray):
                    raise RuntimeError(f"IQP distribution missing for beta={beta}, seed={seed}, sigma={sigma}, K={K}.")

                met_spec = hv.compute_metrics_for_q(
                    q=q_spec,
                    holdout_mask=holdout_mask,
                    qH_unif=qH_unif,
                    H_size=H_size,
                    Q80_thr=float(args.q80_thr),
                    Q80_search_max=int(args.q80_search_max),
                )
                met_iqp = hv.compute_metrics_for_q(
                    q=q_iqp,
                    holdout_mask=holdout_mask,
                    qH_unif=qH_unif,
                    H_size=H_size,
                    Q80_thr=float(args.q80_thr),
                    Q80_search_max=int(args.q80_search_max),
                )

                row = {
                    "beta": float(beta),
                    "seed": int(seed),
                    "sigma": float(sigma),
                    "K": int(K),
                    "holdout_size": int(H_size),
                    "qH_ratio_spec": float(met_spec["qH_ratio"]),
                    "Q80_spec": float(met_spec["Q80"]),
                    "R_spec_Q1000": float(met_spec["R_Q1000"]),
                    "R_spec_Q10000": float(met_spec["R_Q10000"]),
                    "qH_ratio_iqp": float(met_iqp["qH_ratio"]),
                    "Q80_iqp": float(met_iqp["Q80"]),
                    "R_iqp_Q1000": float(met_iqp["R_Q1000"]),
                    "R_iqp_Q10000": float(met_iqp["R_Q10000"]),
                    "loss_iqp": float(art["loss_iqp"]),
                }
                rows_rep.append(row)
                long_rows.append(dict(row))

            best, worst = _pick_best_worst_by_q80_spec(rows_rep)
            q80_cap = float(args.q80_search_max) * 1.05
            spec_vals = [
                float(r["Q80_spec"]) if np.isfinite(float(r["Q80_spec"])) else q80_cap
                for r in rows_rep
            ]
            iqp_vals = [
                float(r["Q80_iqp"]) if np.isfinite(float(r["Q80_iqp"])) else q80_cap
                for r in rows_rep
            ]
            stat = _spearman_with_permutation(
                x=spec_vals,
                y=iqp_vals,
                perm_repeats=int(args.perm_repeats),
                seed=int(seed) + int(round(100 * float(beta))),
            )

            best_iqp = float(best["Q80_iqp"])
            worst_iqp = float(worst["Q80_iqp"])
            delta_abs = float(worst_iqp - best_iqp) if np.isfinite(best_iqp) and np.isfinite(worst_iqp) else float("nan")
            delta_rel = float(delta_abs / worst_iqp) if np.isfinite(delta_abs) and worst_iqp > 0 else float("nan")
            best_better = bool(np.isfinite(delta_abs) and delta_abs > 0.0)

            # Practical transfer metric: is spectral top-1 among IQP top-k?
            topk = max(1, min(int(args.topk), len(rows_rep)))
            iqp_sorted = sorted(
                rows_rep,
                key=lambda r: float(r["Q80_iqp"]) if np.isfinite(float(r["Q80_iqp"])) else float("inf"),
            )
            iqp_topk = {(float(r["sigma"]), int(r["K"])) for r in iqp_sorted[:topk]}
            spec_best_key = (float(best["sigma"]), int(best["K"]))
            top1_in_topk = bool(spec_best_key in iqp_topk)

            replicate_rows.append(
                {
                    "beta": float(beta),
                    "seed": int(seed),
                    "n_settings": int(len(rows_rep)),
                    "n_pairs_spearman": int(stat["n_pairs"]),
                    "spearman_rho": float(stat["rho"]),
                    "spearman_p": float(stat["p_value"]),
                    "spearman_p_perm": float(stat["p_perm"]),
                    "best_sigma_spec": float(best["sigma"]),
                    "best_K_spec": int(best["K"]),
                    "worst_sigma_spec": float(worst["sigma"]),
                    "worst_K_spec": int(worst["K"]),
                    "Q80_spec_best": float(best["Q80_spec"]),
                    "Q80_spec_worst": float(worst["Q80_spec"]),
                    "Q80_iqp_best_spec_setting": float(best_iqp),
                    "Q80_iqp_worst_spec_setting": float(worst_iqp),
                    "delta_Q80_iqp_worst_minus_best": float(delta_abs),
                    "delta_Q80_iqp_rel": float(delta_rel),
                    "best_spec_beats_worst_in_iqp": int(best_better),
                    f"top1_spec_in_iqp_top{int(topk)}": int(top1_in_topk),
                }
            )

    if len(replicate_rows) == 0:
        raise RuntimeError("No replicate summary rows were produced.")

    # Aggregate statistics.
    rho_all = np.array([float(r["spearman_rho"]) for r in replicate_rows], dtype=np.float64)
    rho_all = rho_all[np.isfinite(rho_all)]
    pperm_all = np.array([float(r["spearman_p_perm"]) for r in replicate_rows], dtype=np.float64)
    pperm_all = pperm_all[np.isfinite(pperm_all)]
    win_vals = np.array([float(r["best_spec_beats_worst_in_iqp"]) for r in replicate_rows], dtype=np.float64)
    delta_rel_vals = np.array([float(r["delta_Q80_iqp_rel"]) for r in replicate_rows], dtype=np.float64)
    delta_rel_vals = delta_rel_vals[np.isfinite(delta_rel_vals)]

    rho_ci = _bootstrap_ci(
        values=rho_all.tolist(),
        alpha=float(args.ci_alpha),
        n_boot=int(args.ci_boot),
        seed=int(args.seed),
    )
    delta_ci = _bootstrap_ci(
        values=delta_rel_vals.tolist(),
        alpha=float(args.ci_alpha),
        n_boot=int(args.ci_boot),
        seed=int(args.seed) + 1000,
    )
    win_rate = float(np.mean(win_vals)) if win_vals.size > 0 else float("nan")

    # Beta-wise summary rows for panel and CSV.
    beta_summary_rows: List[Dict[str, object]] = []
    for beta, rows_beta in _iter_groups(replicate_rows, key="beta"):
        vals = np.array([float(r["spearman_rho"]) for r in rows_beta], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ci = _bootstrap_ci(
            values=vals.tolist(),
            alpha=float(args.ci_alpha),
            n_boot=int(args.ci_boot),
            seed=int(args.seed) + int(round(100 * float(beta))),
        )
        beta_summary_rows.append(
            {
                "beta": float(beta),
                "n_replicates": int(len(rows_beta)),
                "spearman_rho_median": float(ci["center"]),
                "spearman_rho_ci_lo": float(ci["ci_lo"]),
                "spearman_rho_ci_hi": float(ci["ci_hi"]),
                "spearman_rho_mean": float(np.mean(vals)) if vals.size > 0 else float("nan"),
            }
        )

    # Plot: 3 panels.
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.3), constrained_layout=False)

    # Panel A: scatter Q80_spec vs Q80_iqp
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    beta_levels = sorted(list({float(r["beta"]) for r in long_rows}))
    beta_to_color: Dict[float, Tuple[float, float, float, float]] = {}
    if len(beta_levels) == 1:
        beta_to_color[beta_levels[0]] = cmap(0.65)
    else:
        for i, b in enumerate(beta_levels):
            t = i / float(len(beta_levels) - 1)
            beta_to_color[b] = cmap(t)

    x_vals: List[float] = []
    y_vals: List[float] = []
    for b in beta_levels:
        rows_b = [r for r in long_rows if float(r["beta"]) == float(b)]
        x = np.array([float(r["Q80_spec"]) for r in rows_b], dtype=np.float64)
        y = np.array([float(r["Q80_iqp"]) for r in rows_b], dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x = x[m]
        y = y[m]
        x_vals.extend(x.tolist())
        y_vals.extend(y.tolist())
        ax.scatter(
            x,
            y,
            s=24,
            alpha=0.72,
            color=beta_to_color[b],
            edgecolor="white",
            linewidth=0.35,
            label=fr"$\beta$={b:g}",
        )

    if len(x_vals) > 0 and len(y_vals) > 0:
        lo = min(min(x_vals), min(y_vals))
        hi = max(max(x_vals), max(y_vals))
        ax.plot([lo, hi], [lo, hi], color="#666666", linestyle="--", linewidth=1.1, alpha=0.8)
        if hi / max(lo, 1e-9) >= 6.0:
            ax.set_xscale("log")
            ax.set_yscale("log")

    ax.set_xlabel(r"Spectral $Q_{80}$ (lower better)")
    ax.set_ylabel(r"IQP $Q_{80}$ (lower better)")
    ax.set_title("Setting-level proxy relationship")
    ax.legend(loc="upper left", fontsize=6.5, frameon=True)

    # Panel B: rho by beta + overall
    ax = axes[1]
    x_labels: List[str] = []
    x_pos: List[float] = []
    rng_jitter = np.random.default_rng(int(args.seed) + 2222)

    for i, beta in enumerate(beta_levels):
        rows_beta = [r for r in replicate_rows if float(r["beta"]) == float(beta)]
        vals = np.array([float(r["spearman_rho"]) for r in rows_beta], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        x = np.full(vals.size, float(i), dtype=np.float64) + rng_jitter.normal(0.0, 0.03, size=vals.size)
        ax.scatter(x, vals, color=beta_to_color[beta], s=28, alpha=0.8, edgecolor="white", linewidth=0.3)
        ci = _bootstrap_ci(vals.tolist(), alpha=float(args.ci_alpha), n_boot=int(args.ci_boot), seed=int(args.seed) + i)
        if np.isfinite(ci["center"]):
            ax.errorbar(
                [float(i)],
                [float(ci["center"])],
                yerr=[[float(ci["center"]) - float(ci["ci_lo"])], [float(ci["ci_hi"]) - float(ci["center"])]],
                fmt="o",
                color="black",
                markersize=3.8,
                linewidth=1.2,
                capsize=2.5,
                zorder=5,
            )
        x_labels.append(fr"$\beta$={beta:g}")
        x_pos.append(float(i))

    # Overall column.
    i_all = float(len(beta_levels))
    rho_overall = np.array([float(r["spearman_rho"]) for r in replicate_rows], dtype=np.float64)
    rho_overall = rho_overall[np.isfinite(rho_overall)]
    x_all = np.full(rho_overall.size, i_all, dtype=np.float64) + rng_jitter.normal(0.0, 0.03, size=rho_overall.size)
    ax.scatter(x_all, rho_overall, color="#222222", s=24, alpha=0.75, edgecolor="white", linewidth=0.3)
    if np.isfinite(rho_ci["center"]):
        ax.errorbar(
            [i_all],
            [float(rho_ci["center"])],
            yerr=[[float(rho_ci["center"]) - float(rho_ci["ci_lo"])], [float(rho_ci["ci_hi"]) - float(rho_ci["center"])]],
            fmt="o",
            color="#111111",
            markersize=4.2,
            linewidth=1.2,
            capsize=2.8,
            zorder=5,
        )
    x_labels.append("overall")
    x_pos.append(i_all)

    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(r"Spearman $\rho(Q80_{spec},Q80_{iqp})$")
    ax.set_title("Rank correlation across replicates")

    # Panel C: paired best-vs-worst transfer
    ax = axes[2]
    y_best = np.array([float(r["Q80_iqp_best_spec_setting"]) for r in replicate_rows], dtype=np.float64)
    y_worst = np.array([float(r["Q80_iqp_worst_spec_setting"]) for r in replicate_rows], dtype=np.float64)
    m = np.isfinite(y_best) & np.isfinite(y_worst) & (y_best > 0) & (y_worst > 0)
    y_best = y_best[m]
    y_worst = y_worst[m]
    for i in range(y_best.size):
        good = bool(y_best[i] < y_worst[i])
        col = "#2CA02C" if good else "#D62728"
        ax.plot([0, 1], [float(y_best[i]), float(y_worst[i])], color=col, alpha=0.35, linewidth=1.1)
    ax.scatter(np.zeros_like(y_best), y_best, color="#2CA02C", s=28, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.scatter(np.ones_like(y_worst), y_worst, color="#D62728", s=28, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["IQP @ spectral-best", "IQP @ spectral-worst"])
    ax.set_ylabel(r"$Q_{80}$ (lower better)")
    if y_best.size > 0 and (max(np.max(y_best), np.max(y_worst)) / max(min(np.min(y_best), np.min(y_worst)), 1e-9) >= 6.0):
        ax.set_yscale("log")

    legend_lines = [
        Line2D([0], [0], color="#2CA02C", linewidth=1.4, alpha=0.8, label="Best beats worst"),
        Line2D([0], [0], color="#D62728", linewidth=1.4, alpha=0.8, label="Worst beats best"),
    ]
    ax.legend(handles=legend_lines, loc="upper right", fontsize=6.4, frameon=True)
    ax.set_title(
        "Best-vs-worst practical transfer\n"
        + f"win-rate={win_rate:.2f}, median rel-improve={float(delta_ci['center']):.3f}"
    )

    fig.suptitle(
        f"Spectral proxy validation | holdout={args.holdout_mode}/{args.holdout_selection}, "
        + f"m={int(args.train_m)}, settings={len(settings)}, betas={len(betas)}, seeds={len(seeds)}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.18, wspace=0.30)

    prefix = (
        f"spectral_proxy_validation_{args.holdout_mode}_{args.holdout_selection}_m{int(args.train_m)}_"
        + f"betas{len(betas)}_seeds{len(seeds)}_settings{len(settings)}"
    )
    plot_pdf = outdir / f"{prefix}.pdf"
    plot_png = outdir / f"{prefix}.png"
    long_csv = outdir / f"{prefix}_long.csv"
    rep_csv = outdir / f"{prefix}_replicate_summary.csv"
    beta_csv = outdir / f"{prefix}_beta_summary.csv"
    summary_json = outdir / f"{prefix}_summary.json"

    fig.savefig(plot_pdf)
    fig.savefig(plot_png, dpi=300)
    plt.close(fig)

    _write_csv(long_csv, long_rows)
    _write_csv(rep_csv, replicate_rows)
    _write_csv(beta_csv, beta_summary_rows)

    claim_primary = bool(np.isfinite(rho_ci["ci_lo"]) and float(rho_ci["ci_lo"]) > 0.0)
    claim_practical = bool(np.isfinite(win_rate) and win_rate >= float(args.win_rate_threshold))

    summary = {
        "config": {
            "n": int(args.n),
            "betas": [float(x) for x in betas],
            "seeds": [int(x) for x in seeds],
            "train_m": int(args.train_m),
            "holdout_mode": str(args.holdout_mode),
            "holdout_selection": str(args.holdout_selection),
            "holdout_k": int(args.holdout_k),
            "sigmas": [float(x) for x in sigmas],
            "Ks": [int(x) for x in Ks],
            "iqp_steps": int(args.iqp_steps),
            "iqp_lr": float(args.iqp_lr),
            "layers": int(args.layers),
            "topk": int(args.topk),
            "perm_repeats": int(args.perm_repeats),
        },
        "aggregate": {
            "n_replicates": int(len(replicate_rows)),
            "rho_median": float(rho_ci["center"]),
            "rho_ci_lo": float(rho_ci["ci_lo"]),
            "rho_ci_hi": float(rho_ci["ci_hi"]),
            "perm_p_median": float(np.median(pperm_all)) if pperm_all.size > 0 else float("nan"),
            "win_rate_best_over_worst": float(win_rate),
            "delta_rel_median": float(delta_ci["center"]),
            "delta_rel_ci_lo": float(delta_ci["ci_lo"]),
            "delta_rel_ci_hi": float(delta_ci["ci_hi"]),
        },
        "claim_checks": {
            "primary_proxy_positive_ci": bool(claim_primary),
            "practical_win_rate": bool(claim_practical),
            "win_rate_threshold": float(args.win_rate_threshold),
        },
        "outputs": {
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
            "long_csv": str(long_csv),
            "replicate_summary_csv": str(rep_csv),
            "beta_summary_csv": str(beta_csv),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[saved] {plot_pdf}")
    print(f"[saved] {plot_png}")
    print(f"[saved] {long_csv}")
    print(f"[saved] {rep_csv}")
    print(f"[saved] {beta_csv}")
    print(f"[saved] {summary_json}")
    print(f"[summary] rho_median={summary['aggregate']['rho_median']:.3f}, win_rate={summary['aggregate']['win_rate_best_over_worst']:.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Validate spectral Q80 proxy over multiple betas and seeds.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "50_spectral_proxy_validation"),
    )

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--betas", type=str, default="0.7,0.8,0.9,1.0")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
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

    ap.add_argument("--sigmas", type=str, default="0.5,1,1.5,2,3")
    ap.add_argument("--Ks", type=str, default="128,256,512")

    ap.add_argument("--iqp-steps", type=int, default=300)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--layers", type=int, default=1)

    ap.add_argument("--qmax", type=int, default=10000)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)

    ap.add_argument("--perm-repeats", type=int, default=1000)
    ap.add_argument("--ci-alpha", type=float, default=0.05)
    ap.add_argument("--ci-boot", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--win-rate-threshold", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=46)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
