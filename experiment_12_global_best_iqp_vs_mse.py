#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 12: global-best IQP parity grid vs IQP MSE over 10 matched seeds.

This driver targets the hardware-selection workflow:

1. use the active 10-seed matched protocol (`111..120`),
2. train the IQP parity family over the full `(sigma, K)` grid offline,
3. select one global-best `(sigma, K)` over the 10 seeds,
4. compare that fixed parity configuration against IQP MSE on the same
   per-seed `D_train`,
5. save enough state to rerender plots or export hardware-ready circuit params.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Rectangle

from experiment_1_kl_diagnostics import (
    CMAP_KL,
    HAS_PENNYLANE,
    K_VALUES,
    SIGMA_VALUES,
    build_parity_matrix,
    build_target_distribution_paper,
    empirical_dist,
    forward_kl,
    get_iqp_pairs_nn_nnn,
    iqp_circuit_zz_only,
    make_bits_table,
    sample_alphas,
    sample_indices,
    qml,
    qnp,
)
from experiment_2_beta_kl_summary import (
    IQP_INIT_OFFSET,
    PARITY_BAND_OFFSET,
    TRAIN_SAMPLE_OFFSET,
    _ci95_halfwidth,
    _reduce_seed_stats,
    _try_rel,
    _write_csv,
    _write_json,
)
from final_plot_style import MSE_COLOR, PARITY_COLOR, TEXT_DARK, apply_final_style, save_pdf
from training_protocol import STANDARD_SEED_IDS_CSV, standard_seed_list, write_training_protocol


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_12_global_best_iqp_vs_mse.py"
OUTPUT_STEM = "experiment_12_global_best_iqp_vs_mse"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM

FIG_W = 540.0 / 72.0
FIG_H = 215.0 / 72.0
MSE_INIT_OFFSET = 20000


def _train_iqp_parity_with_params(
    *,
    n: int,
    layers: int,
    steps: int,
    lr: float,
    P: np.ndarray,
    z_data: np.ndarray,
    seed_init: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP training.")

    dev = qml.device("default.qubit", wires=int(n))
    pairs = get_iqp_pairs_nn_nnn(int(n))

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit_zz_only(W, range(int(n)), pairs, layers=int(layers))
        return qml.probs(wires=range(int(n)))

    P_t = qnp.array(np.asarray(P, dtype=np.float64), requires_grad=False)
    z_t = qnp.array(np.asarray(z_data, dtype=np.float64), requires_grad=False)
    num_params = len(pairs) * int(layers)
    rng = np.random.default_rng(int(seed_init))
    W = qnp.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(float(lr))

    def loss_fn(w):
        q = circuit(w)
        return qnp.mean((z_t - P_t @ q) ** 2)

    for _ in range(1, int(steps) + 1):
        W, _ = opt.step_and_cost(loss_fn, W)

    q_final = np.asarray(circuit(W), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final = q_final / max(1e-15, float(np.sum(q_final)))
    return q_final.astype(np.float64), np.asarray(W, dtype=np.float64)


def _train_iqp_mse_with_params(
    *,
    n: int,
    layers: int,
    steps: int,
    lr: float,
    emp_dist: np.ndarray,
    seed_init: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP training.")

    dev = qml.device("default.qubit", wires=int(n))
    pairs = get_iqp_pairs_nn_nnn(int(n))

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit_zz_only(W, range(int(n)), pairs, layers=int(layers))
        return qml.probs(wires=range(int(n)))

    emp_t = qnp.array(np.asarray(emp_dist, dtype=np.float64), requires_grad=False)
    num_params = len(pairs) * int(layers)
    rng = np.random.default_rng(int(seed_init))
    W = qnp.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(float(lr))

    def loss_fn(w):
        q = circuit(w)
        return qnp.mean((q - emp_t) ** 2)

    for _ in range(1, int(steps) + 1):
        W, _ = opt.step_and_cost(loss_fn, W)

    q_final = np.asarray(circuit(W), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final = q_final / max(1e-15, float(np.sum(q_final)))
    return q_final.astype(np.float64), np.asarray(W, dtype=np.float64)


def _selection_order(value: float, ci95: float, k: int, sigma: float) -> tuple[float, float, int, float]:
    return (float(value), float(ci95), int(k), float(sigma))


def _choose_global_best(
    summary_rows: Sequence[Dict[str, object]],
    *,
    selection_metric: str,
) -> Dict[str, object]:
    metric_key = "mean" if str(selection_metric) == "mean" else "median"
    best_row = None
    best_key = None
    for row in summary_rows:
        key = _selection_order(
            float(row[metric_key]),
            float(row["ci95"]),
            int(row["K"]),
            float(row["sigma"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_row = dict(row)
    if best_row is None:
        raise RuntimeError("Failed to choose a global-best `(sigma, K)` row.")
    return best_row


def _heatmap_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    key: str,
) -> np.ndarray:
    out = np.zeros((len(SIGMA_VALUES), len(K_VALUES)), dtype=np.float64)
    for row in rows:
        i = SIGMA_VALUES.index(float(row["sigma"]))
        j = K_VALUES.index(int(row["K"]))
        out[i, j] = float(row[key])
    return out


def _render_summary_plot(
    *,
    out_pdf: Path,
    out_png: Path,
    mean_grid: np.ndarray,
    best_sigma: float,
    best_k: int,
    per_seed_rows: Sequence[Dict[str, object]],
    selection_metric: str,
) -> None:
    apply_final_style()
    fig, (ax_heat, ax_seed) = plt.subplots(
        1,
        2,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={"width_ratios": [1.0, 1.18]},
        constrained_layout=True,
    )

    norm = colors.Normalize(vmin=float(np.min(mean_grid)), vmax=float(np.max(mean_grid)))
    im = ax_heat.imshow(mean_grid, cmap=CMAP_KL, norm=norm, aspect="auto")
    ax_heat.set_xticks(np.arange(len(K_VALUES)))
    ax_heat.set_xticklabels([str(int(k)) for k in K_VALUES])
    ax_heat.set_yticks(np.arange(len(SIGMA_VALUES)))
    ax_heat.set_yticklabels([f"{float(s):g}" for s in SIGMA_VALUES])
    ax_heat.set_xlabel(r"$K$")
    ax_heat.set_ylabel(r"$\sigma$")
    ax_heat.set_title(r"IQP parity $D_{\mathrm{KL}}(p^{*}\parallel q)$", fontsize=11, pad=6)
    ax_heat.set_xticks(np.arange(-0.5, len(K_VALUES), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(SIGMA_VALUES), 1), minor=True)
    ax_heat.grid(which="minor", color="#FFFFFF", linewidth=1.2)
    ax_heat.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    best_i = SIGMA_VALUES.index(float(best_sigma))
    best_j = K_VALUES.index(int(best_k))
    best_val = float(mean_grid[best_i, best_j])
    best_border_color = "#FFFFFF" if norm(best_val) > 0.25 else TEXT_DARK
    ax_heat.add_patch(
        Rectangle(
            (best_j - 0.5, best_i - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor=best_border_color,
            linewidth=1.2,
        )
    )
    for i, sigma in enumerate(SIGMA_VALUES):
        for j, kval in enumerate(K_VALUES):
            cell_val = float(mean_grid[i, j])
            text_color = TEXT_DARK if norm(cell_val) <= 0.25 else "#FFFFFF"
            ax_heat.text(
                j,
                i,
                f"{cell_val:.3f}",
                ha="center",
                va="center",
                fontsize=9.2,
                color=text_color,
                fontweight="bold" if (i == best_i and j == best_j) else "normal",
            )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label(r"Parity $D_{\mathrm{KL}}(p^{*}\,\|\,q)$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    seed_labels = [str(int(row["seed"])) for row in per_seed_rows]
    y = np.arange(len(per_seed_rows), dtype=np.float64)
    parity_vals = np.asarray([float(row["parity_kl"]) for row in per_seed_rows], dtype=np.float64)
    mse_vals = np.asarray([float(row["mse_kl"]) for row in per_seed_rows], dtype=np.float64)
    for yi, pval, mval in zip(y, parity_vals, mse_vals):
        ax_seed.plot([pval, mval], [yi, yi], color="#CFCFCF", lw=1.0, zorder=1)
    ax_seed.scatter(parity_vals, y, color=PARITY_COLOR, label=f"Parity global-best ({best_sigma:g}, {best_k})", zorder=3)
    ax_seed.scatter(mse_vals, y, color=MSE_COLOR, label="IQP MSE", zorder=3)
    ax_seed.set_yticks(y)
    ax_seed.set_yticklabels(seed_labels)
    ax_seed.invert_yaxis()
    ax_seed.set_xlabel(r"$D_{\mathrm{KL}}(p^{*}\,\|\,q)$")
    ax_seed.set_ylabel("Seed")
    ax_seed.set_title("Per-seed offline KL", fontsize=11)
    ax_seed.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax_seed.grid(False, axis="y")
    ax_seed.legend(loc="upper right", frameon=True, fontsize=8.0)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    save_pdf(fig, out_pdf)


def _write_readme(
    *,
    path: Path,
    selection_metric: str,
    global_best_row: Dict[str, object],
    summary_payload: Dict[str, object],
    pdf_rel: str,
    png_rel: str,
) -> None:
    lines = [
        "# Experiment 12 Summary",
        "",
        "Global-best IQP parity grid selection versus IQP MSE over the active 10 matched seeds.",
        "",
        "Key settings:",
        "",
        f"- beta: `{float(summary_payload['beta']):g}`",
        f"- n: `{int(summary_payload['n'])}`",
        f"- train sample size: `{int(summary_payload['train_m'])}`",
        f"- layers: `{int(summary_payload['layers'])}`",
        f"- IQP budget: `steps={int(summary_payload['iqp_steps'])}`, `lr={float(summary_payload['iqp_lr']):g}`",
        f"- selection metric: `{selection_metric}`",
        f"- matched seeds: `{','.join(str(int(x)) for x in summary_payload['seeds'])}`",
        "",
        "Selected global-best parity grid point:",
        "",
        f"- best `(sigma, K)`: `({float(global_best_row['sigma']):g}, {int(global_best_row['K'])})`",
        f"- parity mean KL: `{float(global_best_row['mean']):.6f}`",
        f"- parity median KL: `{float(global_best_row['median']):.6f}`",
        f"- parity 95% CI: `{float(global_best_row['ci95']):.6f}`",
        "",
        "Parity vs IQP MSE at the selected grid point:",
        "",
        f"- parity mean KL: `{float(summary_payload['parity_mean_kl']):.6f}`",
        f"- MSE mean KL: `{float(summary_payload['mse_mean_kl']):.6f}`",
        f"- parity median KL: `{float(summary_payload['parity_median_kl']):.6f}`",
        f"- MSE median KL: `{float(summary_payload['mse_median_kl']):.6f}`",
        f"- mean delta `(parity - mse)`: `{float(summary_payload['delta_mean']):.6f}`",
        f"- seed wins `(parity < mse)`: `{int(summary_payload['parity_seed_wins'])}/{int(summary_payload['n_seeds'])}`",
        "",
        "Artifacts:",
        "",
        f"- summary plot PDF: `{pdf_rel}`",
        f"- summary plot PNG: `{png_rel}`",
        f"- grid metrics per seed CSV: `{summary_payload['grid_metrics_csv']}`",
        f"- grid summary CSV: `{summary_payload['grid_summary_csv']}`",
        f"- global-best vs MSE CSV: `{summary_payload['global_best_vs_mse_csv']}`",
        f"- data NPZ: `{summary_payload['data_npz']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    ap = argparse.ArgumentParser(description="Experiment 12: global-best IQP parity grid vs IQP MSE.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seeds", type=str, default=STANDARD_SEED_IDS_CSV)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--selection-metric", choices=["mean", "median"], default="mean")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for experiment 12.")

    seed_values = np.asarray([int(x.strip()) for x in str(args.seeds).split(",") if x.strip()], dtype=np.int64)
    p_star, _support, _scores = build_target_distribution_paper(int(args.n), float(args.beta))
    bits_table = make_bits_table(int(args.n))
    num_params = len(get_iqp_pairs_nn_nnn(int(args.n))) * int(args.layers)

    n_seeds = int(seed_values.size)
    n_sigma = len(SIGMA_VALUES)
    n_k = len(K_VALUES)

    parity_kl_grid = np.zeros((n_seeds, n_sigma, n_k), dtype=np.float64)
    parity_weights = np.zeros((n_seeds, n_sigma, n_k, num_params), dtype=np.float64)
    mse_kl = np.zeros(n_seeds, dtype=np.float64)
    mse_weights = np.zeros((n_seeds, num_params), dtype=np.float64)
    train_indices = np.zeros((n_seeds, int(args.train_m)), dtype=np.int64)
    empirical_dists = np.zeros((n_seeds, p_star.size), dtype=np.float64)

    grid_metric_rows: List[Dict[str, object]] = []
    mse_rows: List[Dict[str, object]] = []

    for seed_idx, seed in enumerate(seed_values.tolist()):
        print(f"[experiment12] seed={int(seed)} ({seed_idx + 1}/{n_seeds})", flush=True)
        idxs_train = sample_indices(p_star, int(args.train_m), seed=int(seed) + TRAIN_SAMPLE_OFFSET)
        emp = empirical_dist(idxs_train, p_star.size)
        train_indices[seed_idx] = np.asarray(idxs_train, dtype=np.int64)
        empirical_dists[seed_idx] = np.asarray(emp, dtype=np.float64)

        q_mse, w_mse = _train_iqp_mse_with_params(
            n=int(args.n),
            layers=int(args.layers),
            steps=int(args.iqp_steps),
            lr=float(args.iqp_lr),
            emp_dist=emp,
            seed_init=int(seed) + MSE_INIT_OFFSET + 7 * int(K_VALUES[-1]),
        )
        mse_kl[seed_idx] = float(forward_kl(p_star, q_mse))
        mse_weights[seed_idx] = np.asarray(w_mse, dtype=np.float64)
        mse_rows.append(
            {
                "seed": int(seed),
                "mse_kl": float(mse_kl[seed_idx]),
            }
        )

        for sigma_idx, sigma in enumerate(SIGMA_VALUES):
            for k_idx, kval in enumerate(K_VALUES):
                alphas = sample_alphas(int(args.n), float(sigma), int(kval), seed=int(seed) + PARITY_BAND_OFFSET)
                P = build_parity_matrix(alphas, bits_table)
                z_data = P @ emp
                q_parity, w_parity = _train_iqp_parity_with_params(
                    n=int(args.n),
                    layers=int(args.layers),
                    steps=int(args.iqp_steps),
                    lr=float(args.iqp_lr),
                    P=P,
                    z_data=z_data,
                    seed_init=int(seed) + IQP_INIT_OFFSET + 7 * int(kval),
                )
                kl = float(forward_kl(p_star, q_parity))
                parity_kl_grid[seed_idx, sigma_idx, k_idx] = kl
                parity_weights[seed_idx, sigma_idx, k_idx] = np.asarray(w_parity, dtype=np.float64)
                grid_metric_rows.append(
                    {
                        "seed": int(seed),
                        "sigma": float(sigma),
                        "K": int(kval),
                        "parity_kl": float(kl),
                        "mse_kl_same_seed": float(mse_kl[seed_idx]),
                        "delta_vs_mse": float(kl - mse_kl[seed_idx]),
                    }
                )

    grid_summary_rows: List[Dict[str, object]] = []
    for sigma_idx, sigma in enumerate(SIGMA_VALUES):
        for k_idx, kval in enumerate(K_VALUES):
            vals = parity_kl_grid[:, sigma_idx, k_idx]
            delta_vals = vals - mse_kl
            stats = _reduce_seed_stats(vals)
            delta_stats = _reduce_seed_stats(delta_vals)
            grid_summary_rows.append(
                {
                    "sigma": float(sigma),
                    "K": int(kval),
                    **stats,
                    "delta_mean": float(np.mean(delta_vals)),
                    "delta_median": float(np.median(delta_vals)),
                    "delta_ci95": float(_ci95_halfwidth(np.asarray(delta_vals, dtype=np.float64))),
                    "delta_std": float(np.std(delta_vals, ddof=1)) if delta_vals.size > 1 else 0.0,
                    "parity_seed_wins_vs_mse": int(np.sum(vals < mse_kl)),
                    "mse_seed_wins_vs_parity": int(np.sum(vals > mse_kl)),
                    "ties": int(np.sum(np.isclose(vals, mse_kl))),
                    "delta_q1": float(delta_stats["q1"]),
                    "delta_q3": float(delta_stats["q3"]),
                }
            )

    global_best_row = _choose_global_best(grid_summary_rows, selection_metric=str(args.selection_metric))
    best_sigma = float(global_best_row["sigma"])
    best_k = int(global_best_row["K"])
    best_sigma_idx = SIGMA_VALUES.index(best_sigma)
    best_k_idx = K_VALUES.index(best_k)

    global_best_vs_mse_rows: List[Dict[str, object]] = []
    parity_best_vals = parity_kl_grid[:, best_sigma_idx, best_k_idx]
    parity_best_weights = parity_weights[:, best_sigma_idx, best_k_idx]
    delta_vals = parity_best_vals - mse_kl
    for seed_idx, seed in enumerate(seed_values.tolist()):
        global_best_vs_mse_rows.append(
            {
                "seed": int(seed),
                "sigma": float(best_sigma),
                "K": int(best_k),
                "parity_kl": float(parity_best_vals[seed_idx]),
                "mse_kl": float(mse_kl[seed_idx]),
                "delta_vs_mse": float(delta_vals[seed_idx]),
            }
        )

    representative_idx = int(np.argmin(np.abs(parity_best_vals - float(np.median(parity_best_vals)))))
    best_case_idx = int(np.argmin(delta_vals))
    worst_case_idx = int(np.argmax(delta_vals))

    grid_metrics_csv = outdir / f"{OUTPUT_STEM}_grid_metrics_per_seed.csv"
    grid_summary_csv = outdir / f"{OUTPUT_STEM}_grid_summary.csv"
    global_best_vs_mse_csv = outdir / f"{OUTPUT_STEM}_global_best_vs_mse_per_seed.csv"
    summary_json = outdir / f"{OUTPUT_STEM}_summary.json"
    run_config_json = outdir / "RUN_CONFIG.json"
    data_npz = outdir / f"{OUTPUT_STEM}_data.npz"
    readme = outdir / "README.md"
    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}.png"

    _write_csv(grid_metrics_csv, grid_metric_rows)
    _write_csv(grid_summary_csv, grid_summary_rows)
    _write_csv(global_best_vs_mse_csv, global_best_vs_mse_rows)

    mean_grid = _heatmap_from_rows(grid_summary_rows, key="mean")
    _render_summary_plot(
        out_pdf=out_pdf,
        out_png=out_png,
        mean_grid=mean_grid,
        best_sigma=best_sigma,
        best_k=best_k,
        per_seed_rows=global_best_vs_mse_rows,
        selection_metric=str(args.selection_metric),
    )

    summary_payload = {
        "script": SCRIPT_REL,
        "beta": float(args.beta),
        "n": int(args.n),
        "train_m": int(args.train_m),
        "layers": int(args.layers),
        "iqp_steps": int(args.iqp_steps),
        "iqp_lr": float(args.iqp_lr),
        "selection_metric": str(args.selection_metric),
        "seeds": seed_values.tolist(),
        "n_seeds": int(n_seeds),
        "sigma_values": [float(x) for x in SIGMA_VALUES],
        "k_values": [int(x) for x in K_VALUES],
        "global_best_sigma": float(best_sigma),
        "global_best_k": int(best_k),
        "global_best_mean": float(global_best_row["mean"]),
        "global_best_median": float(global_best_row["median"]),
        "global_best_ci95": float(global_best_row["ci95"]),
        "global_best_seed_wins_vs_mse": int(global_best_row["parity_seed_wins_vs_mse"]),
        "parity_mean_kl": float(np.mean(parity_best_vals)),
        "parity_median_kl": float(np.median(parity_best_vals)),
        "parity_ci95_kl": float(_ci95_halfwidth(np.asarray(parity_best_vals, dtype=np.float64))),
        "mse_mean_kl": float(np.mean(mse_kl)),
        "mse_median_kl": float(np.median(mse_kl)),
        "mse_ci95_kl": float(_ci95_halfwidth(np.asarray(mse_kl, dtype=np.float64))),
        "delta_mean": float(np.mean(delta_vals)),
        "delta_median": float(np.median(delta_vals)),
        "delta_ci95": float(_ci95_halfwidth(np.asarray(delta_vals, dtype=np.float64))),
        "parity_seed_wins": int(np.sum(parity_best_vals < mse_kl)),
        "mse_seed_wins": int(np.sum(parity_best_vals > mse_kl)),
        "ties": int(np.sum(np.isclose(parity_best_vals, mse_kl))),
        "representative_seed": int(seed_values[representative_idx]),
        "representative_seed_parity_kl": float(parity_best_vals[representative_idx]),
        "representative_seed_mse_kl": float(mse_kl[representative_idx]),
        "best_case_seed": int(seed_values[best_case_idx]),
        "best_case_delta_vs_mse": float(delta_vals[best_case_idx]),
        "worst_case_seed": int(seed_values[worst_case_idx]),
        "worst_case_delta_vs_mse": float(delta_vals[worst_case_idx]),
        "grid_metrics_csv": _try_rel(grid_metrics_csv),
        "grid_summary_csv": _try_rel(grid_summary_csv),
        "global_best_vs_mse_csv": _try_rel(global_best_vs_mse_csv),
        "data_npz": _try_rel(data_npz),
        "plot_pdf": _try_rel(out_pdf),
        "plot_png": _try_rel(out_png),
    }

    _write_json(summary_json, summary_payload)
    _write_json(
        run_config_json,
        {
            "script": SCRIPT_REL,
            "outdir": _try_rel(outdir),
            "beta": float(args.beta),
            "n": int(args.n),
            "seeds": seed_values.tolist(),
            "train_m": int(args.train_m),
            "layers": int(args.layers),
            "iqp_steps": int(args.iqp_steps),
            "iqp_lr": float(args.iqp_lr),
            "selection_metric": str(args.selection_metric),
            "sigma_values": [float(x) for x in SIGMA_VALUES],
            "k_values": [int(x) for x in K_VALUES],
            "grid_metrics_csv": _try_rel(grid_metrics_csv),
            "grid_summary_csv": _try_rel(grid_summary_csv),
            "global_best_vs_mse_csv": _try_rel(global_best_vs_mse_csv),
            "summary_json": _try_rel(summary_json),
            "data_npz": _try_rel(data_npz),
            "plot_pdf": _try_rel(out_pdf),
            "plot_png": _try_rel(out_png),
            "command": (
                f"python {SCRIPT_REL} --beta {float(args.beta):g} --n {int(args.n)} "
                f"--seeds {str(args.seeds)} --train-m {int(args.train_m)} --layers {int(args.layers)} "
                f"--iqp-steps {int(args.iqp_steps)} --iqp-lr {float(args.iqp_lr):g} "
                f"--selection-metric {str(args.selection_metric)} --outdir {str(_try_rel(outdir))}"
            ),
        },
    )

    np.savez(
        data_npz,
        seeds=seed_values,
        sigma_values=np.asarray(SIGMA_VALUES, dtype=np.float64),
        k_values=np.asarray(K_VALUES, dtype=np.int64),
        p_star=np.asarray(p_star, dtype=np.float64),
        parity_kl_grid=parity_kl_grid,
        mse_kl=np.asarray(mse_kl, dtype=np.float64),
        parity_weights=parity_weights,
        mse_weights=mse_weights,
        train_indices=train_indices,
        empirical_dists=empirical_dists,
        global_best_parity_weights=parity_best_weights,
        global_best_sigma=np.asarray([best_sigma], dtype=np.float64),
        global_best_k=np.asarray([best_k], dtype=np.int64),
    )

    write_training_protocol(
        outdir,
        experiment_name="Experiment 12 global-best IQP parity vs IQP MSE",
        note=(
            "Select a single `(sigma, K)` globally over the active 10 matched seeds, "
            "then compare that fixed parity setting against IQP MSE on the same `D_train`."
        ),
        source_relpath=SCRIPT_REL,
        metrics_note="Selection is based on the chosen parity grid statistic only; the parity-vs-MSE comparison is then evaluated at that fixed selected grid point.",
    )
    _write_readme(
        path=readme,
        selection_metric=str(args.selection_metric),
        global_best_row=global_best_row,
        summary_payload=summary_payload,
        pdf_rel=_try_rel(out_pdf),
        png_rel=_try_rel(out_png),
    )
    print(f"[experiment12] wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    run()
