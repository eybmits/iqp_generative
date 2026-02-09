#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 11: Beta-family sweep (paper_even) under configurable holdout mode.

What this script produces:
1) Per-beta plots (all strong baselines, same style family as exp02/exp10):
   - recovery curve:        beta_<tag>_recovery_all_baselines.pdf
   - fit-distance bars:     beta_<tag>_fit_distance_all_baselines.pdf
   - target score profile:  beta_<tag>_target_score_distribution.pdf
2) Sweep summary plots:
   - beta_sweep_Q80_vs_beta.pdf
   - beta_sweep_qH_vs_beta.pdf
   - beta_sweep_fit_tv_vs_beta.pdf
3) One full collage (all betas x all plot types):
   - collage_all_plots_all_betas.pdf
4) CSVs:
   - beta_sweep_metrics_long.csv
   - one metrics CSV per beta

Run:
  python3 experiments/exp11_beta_sweep_global_holdout.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402
from experiments import exp10_strong_classical_recovery as exp10  # noqa: E402


LEGEND_STYLE = dict(
    loc="lower right",
    fontsize=7.0,
    frameon=True,
    framealpha=0.90,
    facecolor="white",
    edgecolor="none",
    handlelength=1.6,
    labelspacing=0.25,
    borderpad=0.25,
    handletextpad=0.5,
    borderaxespad=0.2,
)

MODEL_SPECS = [
    ("iqp_parity_mse", "IQP (parity)", hv.COLORS["model"], "-", 2.2),
    ("iqp_prob_mse", "IQP (prob-MSE)", hv.COLORS["model_prob_mse"], "--", 2.0),
    ("classical_nnn_fields_parity", "Ising+fields (NN+NNN)", "#005A9C", "-", 1.9),
    ("classical_dense_fields_xent", "Dense Ising+fields (xent)", "#8C564B", (0, (5, 2)), 1.9),
    ("classical_transformer_mle", "AR Transformer (MLE)", "#1AA7A1", "--", 2.0),
    ("classical_maxent_parity", "MaxEnt parity (P,z)", "#9467BD", "--", 2.1),
]

MODEL_LABEL_SHORT = {
    "iqp_parity_mse": "IQP-par",
    "iqp_prob_mse": "IQP-prob",
    "classical_nnn_fields_parity": "NNN+f",
    "classical_dense_fields_xent": "Dense-xent",
    "classical_transformer_mle": "AR-Tr",
    "classical_maxent_parity": "MaxEnt",
}


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _beta_tag(beta: float) -> str:
    return f"{beta:.2f}".replace("-", "m").replace(".", "p")


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


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _mass_cov_curve(q: np.ndarray, holdout_mask: np.ndarray, p_star: np.ndarray, Qvals: np.ndarray) -> np.ndarray:
    idx = np.where(holdout_mask)[0]
    if idx.size == 0:
        return np.zeros_like(Qvals, dtype=np.float64)
    qh = np.clip(q[idx], 0.0, 1.0)
    ph = p_star[idx]
    # C_m(Q) = sum_{x in H} p*(x) * [1 - (1 - q(x))^Q]
    base = np.clip(1.0 - qh[None, :], 0.0, 1.0)
    seen = 1.0 - np.power(base, Qvals[:, None])
    return (seen * ph[None, :]).sum(axis=1)


def _build_holdout(
    holdout_mode: str,
    p_star: np.ndarray,
    support: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    m_train_for_holdout: int,
    holdout_k: int,
    holdout_pool: int,
    seed: int,
) -> np.ndarray:
    if holdout_mode == "global":
        candidate = support.astype(bool)
    elif holdout_mode == "high_value":
        candidate = good_mask
    else:
        raise ValueError(f"Unsupported holdout_mode: {holdout_mode}")
    return hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate,
        bits_table=bits_table,
        m_train=m_train_for_holdout,
        holdout_k=holdout_k,
        pool_size=holdout_pool,
        seed=seed + 111,
    )


def _train_models_for_beta(
    holdout_mode: str,
    n: int,
    beta: float,
    seed: int,
    train_m: int,
    sigma: float,
    K: int,
    layers: int,
    holdout_k: int,
    holdout_pool: int,
    holdout_m_train: int,
    good_frac: float,
    iqp_steps: int,
    iqp_lr: float,
    iqp_eval_every: int,
    q80_thr: float,
    q80_search_max: int,
    artr_epochs: int,
    artr_d_model: int,
    artr_heads: int,
    artr_layers: int,
    artr_ff: int,
    artr_lr: float,
    artr_batch_size: int,
    maxent_steps: int,
    maxent_lr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]], List[Dict[str, float]]]:
    bits_table = hv.make_bits_table(n)
    p_star, support, scores = hv.build_target_distribution_paper(n, beta)
    good_mask = hv.topk_mask_by_scores(scores, support, frac=good_frac)

    holdout_mask = _build_holdout(
        holdout_mode=holdout_mode,
        p_star=p_star,
        support=support,
        good_mask=good_mask,
        bits_table=bits_table,
        m_train_for_holdout=holdout_m_train,
        holdout_k=holdout_k,
        holdout_pool=holdout_pool,
        seed=seed,
    )

    cfg = hv.Config(
        n=n,
        beta=beta,
        train_m=train_m,
        holdout_k=holdout_k,
        holdout_pool=holdout_pool,
        seed=seed,
        good_frac=good_frac,
        sigmas=[sigma],
        Ks=[K],
        Qmax=10000,
        Q80_thr=q80_thr,
        Q80_search_max=q80_search_max,
        target_family="paper_even",
        adversarial=False,
        use_iqp=True,
        use_classical=True,
        iqp_steps=iqp_steps,
        iqp_lr=iqp_lr,
        iqp_eval_every=iqp_eval_every,
        iqp_layers=layers,
        iqp_loss="parity_mse",
        outdir=str(ROOT / "outputs" / "paper_even_final"),
    )

    art = hv.rerun_single_setting(
        cfg=cfg,
        p_star=p_star,
        holdout_mask=holdout_mask,
        bits_table=bits_table,
        sigma=sigma,
        K=K,
        return_hist=False,
        iqp_loss="parity_mse",
    )
    q_iqp_parity = art["q_iqp"]
    P = art["P"]
    z_data = art["z"]
    assert isinstance(q_iqp_parity, np.ndarray)
    assert isinstance(P, np.ndarray)
    assert isinstance(z_data, np.ndarray)

    cfg_prob = hv.Config(**{**cfg.__dict__, "use_classical": False, "iqp_loss": "prob_mse"})
    art_prob = hv.rerun_single_setting(
        cfg=cfg_prob,
        p_star=p_star,
        holdout_mask=holdout_mask,
        bits_table=bits_table,
        sigma=sigma,
        K=K,
        return_hist=False,
        iqp_loss="prob_mse",
    )
    q_iqp_prob = art_prob["q_iqp"]
    assert isinstance(q_iqp_prob, np.ndarray)

    p_train = p_star.copy()
    p_train[holdout_mask] = 0.0
    p_train /= p_train.sum()
    idxs_train = hv.sample_indices(p_train, train_m, seed=seed + 7)
    emp = hv.empirical_dist(idxs_train, p_star.size)

    q_nnn_fields = exp10._train_classical_boltzmann(
        n=n,
        layers=layers,
        steps=iqp_steps,
        lr=iqp_lr,
        seed_init=seed + 30001,
        P=P,
        z_data=z_data,
        loss_mode="parity_mse",
        emp_dist=emp,
        topology="nn_nnn",
        include_fields=True,
    )
    q_dense_xent = exp10._train_classical_boltzmann(
        n=n,
        layers=layers,
        steps=iqp_steps,
        lr=iqp_lr,
        seed_init=seed + 30004,
        P=P,
        z_data=z_data,
        loss_mode="xent",
        emp_dist=emp,
        topology="dense",
        include_fields=True,
    )
    q_transformer = exp10._train_transformer_autoregressive(
        bits_table=bits_table,
        idxs_train=idxs_train,
        n=n,
        seed=seed + 35501,
        epochs=artr_epochs,
        d_model=artr_d_model,
        nhead=artr_heads,
        num_layers=artr_layers,
        dim_ff=artr_ff,
        lr=artr_lr,
        batch_size=artr_batch_size,
    )
    q_maxent = exp10._train_maxent_parity(
        P=P,
        z_data=z_data,
        seed=seed + 36001,
        steps=maxent_steps,
        lr=maxent_lr,
    )

    q_by_key = {
        "iqp_parity_mse": q_iqp_parity,
        "iqp_prob_mse": q_iqp_prob,
        "classical_nnn_fields_parity": q_nnn_fields,
        "classical_dense_fields_xent": q_dense_xent,
        "classical_transformer_mle": q_transformer,
        "classical_maxent_parity": q_maxent,
    }

    model_rows: List[Dict[str, object]] = []
    for key, label, color, ls, lw in MODEL_SPECS:
        model_rows.append({"key": key, "label": label, "q": q_by_key[key], "color": color, "ls": ls, "lw": lw})

    N = p_star.size
    H_size = int(np.sum(holdout_mask))
    q_unif = np.ones(N, dtype=np.float64) / N
    qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0

    metrics_rows: List[Dict[str, float]] = []
    for row in model_rows:
        q = row["q"]
        assert isinstance(q, np.ndarray)
        met = hv.compute_metrics_for_q(q, holdout_mask, qH_unif, H_size, q80_thr, q80_search_max)
        fit = exp10._distribution_fit_metrics(q=q, p_star=p_star)
        metrics_rows.append(
            {
                "beta": float(beta),
                "model_key": str(row["key"]),
                "model_label": str(row["label"]),
                "qH": float(met["qH"]),
                "qH_ratio": float(met["qH_ratio"]),
                "Q80": float(met["Q80"]),
                "Q80_pred": float(met["Q80_pred"]),
                "R_Q1000": float(met["R_Q1000"]),
                "R_Q10000": float(met["R_Q10000"]),
                "fit_tv_to_pstar": float(fit["tv"]),
                "fit_js_dist_to_pstar": float(fit["js_dist"]),
                "fit_prob_mse_to_pstar": float(fit["prob_mse"]),
                "holdout_size": float(H_size),
                "p_star_holdout": float(p_star[holdout_mask].sum()),
                "sigma": float(sigma),
                "K": float(K),
                "seed": float(seed),
                "train_m": float(train_m),
            }
        )

    return p_star, scores, holdout_mask, model_rows, metrics_rows


def _plot_recovery_single(ax, p_star: np.ndarray, holdout_mask: np.ndarray, model_rows: List[Dict[str, object]], Q: np.ndarray) -> None:
    y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, Q)
    ax.plot(Q, y_star, color=hv.COLORS["target"], linewidth=2.0, zorder=10)
    for idx, row in enumerate(model_rows):
        q = row["q"]
        assert isinstance(q, np.ndarray)
        y = hv.expected_unique_fraction(q, holdout_mask, Q)
        ax.plot(
            Q,
            y,
            color=str(row["color"]),
            linestyle=row.get("ls", "-"),
            linewidth=float(row.get("lw", 1.9)),
            alpha=float(row.get("alpha", 1.0)),
            zorder=9 - idx,
        )
    ax.plot(Q, y_unif, color=hv.COLORS["gray"], linewidth=1.5, linestyle="--", alpha=0.9, zorder=1)
    ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlim(0, 10000)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")


def _plot_mass_single(ax, p_star: np.ndarray, holdout_mask: np.ndarray, model_rows: List[Dict[str, object]], Q: np.ndarray) -> None:
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    pH = float(p_star[holdout_mask].sum())
    c_target = _mass_cov_curve(p_star, holdout_mask, p_star, Q) / max(1e-15, pH)
    c_unif = _mass_cov_curve(q_unif, holdout_mask, p_star, Q) / max(1e-15, pH)
    ax.plot(Q, c_target, color=hv.COLORS["target"], linewidth=2.0, zorder=10)
    for idx, row in enumerate(model_rows):
        q = row["q"]
        assert isinstance(q, np.ndarray)
        c = _mass_cov_curve(q, holdout_mask, p_star, Q) / max(1e-15, pH)
        ax.plot(
            Q,
            c,
            color=str(row["color"]),
            linestyle=row.get("ls", "-"),
            linewidth=float(row.get("lw", 1.9)),
            alpha=float(row.get("alpha", 1.0)),
            zorder=9 - idx,
        )
    ax.plot(Q, c_unif, color=hv.COLORS["gray"], linewidth=1.5, linestyle="--", alpha=0.9, zorder=1)
    ax.set_xlim(0, 10000)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Mass coverage $C_m(Q)/p^*(H)$")


def _plot_fit_single(ax, metrics_rows: List[Dict[str, float]]) -> None:
    m_by_key = {str(r["model_key"]): r for r in metrics_rows}
    keys = [k for k, _, _, _, _ in MODEL_SPECS]
    vals = [float(m_by_key[k]["fit_tv_to_pstar"]) for k in keys]
    cols = [c for _, _, c, _, _ in MODEL_SPECS]
    x = np.arange(len(keys))
    ax.bar(x, vals, color=cols, alpha=0.9)
    ax.set_ylabel(r"TV($q,p^*$)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL_SHORT[k] for k in keys], rotation=35, ha="right", fontsize=6)


def _plot_target_score_distribution(ax, p_star: np.ndarray, scores: np.ndarray) -> None:
    mask = p_star > 0.0
    if not np.any(mask):
        ax.text(0.5, 0.5, "No support", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlabel("Score")
        ax.set_ylabel(r"$p^*(s)$")
        return

    score_vals = scores[mask]
    masses = p_star[mask]
    uniq = np.unique(score_vals)
    uniq = np.sort(uniq)
    pmass = np.array([float(masses[score_vals == s].sum()) for s in uniq], dtype=np.float64)

    ax.bar(uniq, pmass, color=hv.COLORS["target"], alpha=0.85, width=0.75)
    ax.plot(uniq, pmass, color=hv.COLORS["target"], marker="o", markersize=3.2, linewidth=1.2, alpha=0.95)
    ax.set_xlabel("Score")
    ax.set_ylabel(r"$p^*(s)$")
    ax.set_ylim(0.0, max(1e-6, float(pmass.max()) * 1.12))
    ax.set_xticks(uniq)


def _save_per_beta_plots(
    out_per_beta: Path,
    holdout_mode: str,
    train_m: int,
    beta: float,
    p_star: np.ndarray,
    scores: np.ndarray,
    holdout_mask: np.ndarray,
    model_rows: List[Dict[str, object]],
    metrics_rows: List[Dict[str, float]],
    Q: np.ndarray,
) -> None:
    btag = _beta_tag(beta)

    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
    _plot_recovery_single(ax, p_star, holdout_mask, model_rows, Q)
    handles = [Line2D([0], [0], color=hv.COLORS["target"], lw=2.0, label=r"Target $p^*$")]
    for row in model_rows:
        handles.append(
            Line2D([0], [0], color=str(row["color"]), lw=float(row.get("lw", 1.9)), ls=row.get("ls", "-"), label=str(row["label"]))
        )
    handles.append(Line2D([0], [0], color=hv.COLORS["gray"], lw=1.5, ls="--", label="Uniform"))
    ax.legend(handles=handles, **LEGEND_STYLE)
    ax.set_title(fr"{holdout_mode} holdout, $m$={int(train_m)}, $\beta$={beta:g}")
    fig.savefig(out_per_beta / f"beta_{btag}_recovery_all_baselines.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
    _plot_fit_single(ax, metrics_rows)
    ax.set_title(fr"Fit distance to target, $\beta$={beta:g}")
    fig.savefig(out_per_beta / f"beta_{btag}_fit_distance_all_baselines.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
    _plot_target_score_distribution(ax, p_star, scores)
    ax.set_title(fr"Target score profile, $\beta$={beta:g}")
    fig.savefig(out_per_beta / f"beta_{btag}_target_score_distribution.pdf")
    plt.close(fig)


def _make_summary_plots(out_summary: Path, rows_long: List[Dict[str, float]]) -> None:
    by_model: Dict[str, List[Dict[str, float]]] = {}
    for r in rows_long:
        by_model.setdefault(str(r["model_key"]), []).append(r)
    for k in by_model:
        by_model[k].sort(key=lambda r: float(r["beta"]))

    # Q80 vs beta
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
    for key, label, color, ls, lw in MODEL_SPECS:
        ys = [float(r["Q80"]) for r in by_model[key]]
        xs = [float(r["beta"]) for r in by_model[key]]
        ax.plot(xs, ys, color=color, ls=ls, lw=lw, marker="o", ms=3.2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$Q_{80}$ (lower is better)")
    ax.legend(**LEGEND_STYLE)
    fig.savefig(out_summary / "beta_sweep_Q80_vs_beta.pdf")
    plt.close(fig)

    # qH vs beta
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
    for key, label, color, ls, lw in MODEL_SPECS:
        ys = [float(r["qH"]) for r in by_model[key]]
        xs = [float(r["beta"]) for r in by_model[key]]
        ax.plot(xs, ys, color=color, ls=ls, lw=lw, marker="o", ms=3.2, label=label)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$q(H)$")
    ax.legend(**LEGEND_STYLE)
    fig.savefig(out_summary / "beta_sweep_qH_vs_beta.pdf")
    plt.close(fig)

    # fit TV vs beta
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
    for key, label, color, ls, lw in MODEL_SPECS:
        ys = [float(r["fit_tv_to_pstar"]) for r in by_model[key]]
        xs = [float(r["beta"]) for r in by_model[key]]
        ax.plot(xs, ys, color=color, ls=ls, lw=lw, marker="o", ms=3.2, label=label)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"TV$(q,p^*)$")
    ax.legend(**LEGEND_STYLE)
    fig.savefig(out_summary / "beta_sweep_fit_tv_vs_beta.pdf")
    plt.close(fig)


def _make_collage(
    out_collage: Path,
    holdout_mode: str,
    betas: List[float],
    artifacts: Dict[float, Dict[str, object]],
    Q: np.ndarray,
) -> None:
    n_b = len(betas)
    fig, axes = plt.subplots(n_b, 3, figsize=(10.5, max(2.3 * n_b, 3.5)), constrained_layout=True)
    if n_b == 1:
        axes = np.array([axes])  # shape (1,3)

    legend_handles = [Line2D([0], [0], color=hv.COLORS["target"], lw=2.0, label=r"Target $p^*$")]
    for key, label, color, ls, lw in MODEL_SPECS:
        legend_handles.append(Line2D([0], [0], color=color, lw=lw, ls=ls, label=label))
    legend_handles.append(Line2D([0], [0], color=hv.COLORS["gray"], lw=1.5, ls="--", label="Uniform"))

    for i, beta in enumerate(betas):
        art = artifacts[beta]
        p_star = art["p_star"]
        scores = art["scores"]
        holdout_mask = art["holdout_mask"]
        model_rows = art["model_rows"]
        metrics_rows = art["metrics_rows"]

        ax0 = axes[i, 0]
        _plot_recovery_single(ax0, p_star, holdout_mask, model_rows, Q)
        ax0.set_title(fr"$\beta$={beta:g} | Recovery ({holdout_mode})")

        ax1 = axes[i, 1]
        _plot_fit_single(ax1, metrics_rows)
        ax1.set_title(fr"$\beta$={beta:g} | Fit TV ({holdout_mode})")

        ax2 = axes[i, 2]
        assert isinstance(scores, np.ndarray)
        _plot_target_score_distribution(ax2, p_star, scores)
        ax2.set_title(fr"$\beta$={beta:g} | Target score profile")

        if i < n_b - 1:
            for ax in (ax0, ax1, ax2):
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    axes[0, 0].legend(handles=legend_handles, loc="upper right", fontsize=6.3, frameon=True, framealpha=0.9)
    fig.savefig(out_collage / "collage_all_plots_all_betas.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "08_claim_beta_sweep_global_m5000"),
    )
    ap.add_argument("--betas", type=str, default="0.2,0.4,0.6,0.8,1.0,1.2,1.4")
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--train-m", type=int, default=5000)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    ap.add_argument("--iqp-steps", type=int, default=400)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--artr-epochs", type=int, default=300)
    ap.add_argument("--artr-d-model", type=int, default=64)
    ap.add_argument("--artr-heads", type=int, default=4)
    ap.add_argument("--artr-layers", type=int, default=2)
    ap.add_argument("--artr-ff", type=int, default=128)
    ap.add_argument("--artr-lr", type=float, default=1e-3)
    ap.add_argument("--artr-batch-size", type=int, default=256)
    ap.add_argument("--maxent-steps", type=int, default=2500)
    ap.add_argument("--maxent-lr", type=float, default=5e-2)
    args = ap.parse_args()

    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")
    if not exp10.HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    hv.set_style(base=8)
    outdir = Path(_ensure_outdir(args.outdir))
    out_per_beta = Path(_ensure_outdir(str(outdir / "per_beta")))
    out_summary = Path(_ensure_outdir(str(outdir / "summary")))
    out_collage = Path(_ensure_outdir(str(outdir / "collage")))

    betas = _parse_list_floats(args.betas)
    Q = _q_grid(10000)

    rows_long: List[Dict[str, float]] = []
    artifacts: Dict[float, Dict[str, object]] = {}

    for beta in betas:
        print(f"[Beta sweep] beta={beta:g}")
        p_star, scores, holdout_mask, model_rows, metrics_rows = _train_models_for_beta(
            holdout_mode=args.holdout_mode,
            n=args.n,
            beta=beta,
            seed=args.seed,
            train_m=args.train_m,
            sigma=args.sigma,
            K=args.K,
            layers=args.layers,
            holdout_k=args.holdout_k,
            holdout_pool=args.holdout_pool,
            holdout_m_train=args.holdout_m_train,
            good_frac=args.good_frac,
            iqp_steps=args.iqp_steps,
            iqp_lr=args.iqp_lr,
            iqp_eval_every=args.iqp_eval_every,
            q80_thr=args.q80_thr,
            q80_search_max=args.q80_search_max,
            artr_epochs=args.artr_epochs,
            artr_d_model=args.artr_d_model,
            artr_heads=args.artr_heads,
            artr_layers=args.artr_layers,
            artr_ff=args.artr_ff,
            artr_lr=args.artr_lr,
            artr_batch_size=args.artr_batch_size,
            maxent_steps=args.maxent_steps,
            maxent_lr=args.maxent_lr,
        )
        rows_long.extend(metrics_rows)
        artifacts[beta] = {
            "p_star": p_star,
            "scores": scores,
            "holdout_mask": holdout_mask,
            "model_rows": model_rows,
            "metrics_rows": metrics_rows,
        }

        btag = _beta_tag(beta)
        _write_csv(out_per_beta / f"beta_{btag}_metrics.csv", metrics_rows)
        _save_per_beta_plots(
            out_per_beta,
            args.holdout_mode,
            args.train_m,
            beta,
            p_star,
            scores,
            holdout_mask,
            model_rows,
            metrics_rows,
            Q,
        )

    _write_csv(out_summary / "beta_sweep_metrics_long.csv", rows_long)
    _make_summary_plots(out_summary, rows_long)
    _make_collage(out_collage, args.holdout_mode, betas, artifacts, Q)

    print(f"Done. Results in {outdir}")


if __name__ == "__main__":
    main()
