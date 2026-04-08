#!/usr/bin/env python3
"""Experiment 16: mean/std recovery curves for seedwise-best IQP parity vs IQP MSE."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from final_plot_style import apply_final_style, save_pdf
from training_protocol import write_training_protocol

from experiment_1_kl_diagnostics import build_parity_matrix
from experiment_3_beta_quality_coverage import (
    ELITE_FRAC,
    PARITY_BAND_OFFSET,
    build_target_distribution_paper,
    empirical_dist,
    make_bits_table,
    sample_alphas,
    topk_mask_by_scores,
)
from experiment_4_recovery_sigmak_triplet import (
    FIG_H,
    FIG_W,
    PARITY_BEST_COLOR,
    PARITY_MSE_COLOR,
    TARGET_COLOR,
    UNIFORM_COLOR,
    _expected_unique_fraction,
    _q_grid,
    _reconstruct_bandlimited,
    _style_ax,
    apply_final_style as apply_experiment4_style,
)


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = Path(__file__).name
OUTPUT_STEM = "experiment_16_seedwise_best_hardware_recovery_curve"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM
DEFAULT_EXPERIMENT12_DIR = ROOT / "plots" / "experiment_12_global_best_iqp_vs_mse"
DEFAULT_EXPERIMENT15_DIR = ROOT / "plots" / "experiment_15_ibm_hardware_seedwise_best_coverage"

PARITY_COLOR = PARITY_BEST_COLOR
MSE_COLOR = PARITY_MSE_COLOR
SPECTRAL_COLOR = "#666666"


def _try_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _common_completed_seeds(rows: List[Dict[str, str]]) -> List[int]:
    by_model: Dict[str, set[int]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_key"]), set()).add(int(row["seed"]))
    return sorted(by_model.get("parity_seedwise_best", set()) & by_model.get("iqp_mse", set()))


def _render_plot(
    *,
    out_pdf: Path,
    out_png: Path,
    q_grid: np.ndarray,
    curve_mean: Dict[str, np.ndarray],
    curve_std: Dict[str, np.ndarray],
    title: str,
    hardware_only: bool,
    hide_spectral: bool,
    paper: bool,
) -> None:
    if paper:
        apply_experiment4_style()
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    else:
        apply_final_style()
        fig, ax = plt.subplots(figsize=(3.55, 2.7), constrained_layout=True)

    ax.plot(q_grid, curve_mean["target"], color=TARGET_COLOR, lw=2.2, label=r"Target $p^*$", zorder=6)
    ax.plot(q_grid, curve_mean["uniform"], color=UNIFORM_COLOR, lw=1.7, ls=":", label="Uniform", zorder=1)
    if not hide_spectral:
        ax.plot(q_grid, curve_mean["spectral"], color=SPECTRAL_COLOR, lw=1.8, ls="-.", label="Spectral completion", zorder=2)
        if np.any(curve_std["spectral"] > 0):
            ax.fill_between(
                q_grid,
                np.maximum(0.0, curve_mean["spectral"] - curve_std["spectral"]),
                curve_mean["spectral"] + curve_std["spectral"],
                color=SPECTRAL_COLOR,
                alpha=0.08,
                linewidth=0.0,
            )

    ax.plot(q_grid, curve_mean["parity_hw"], color=PARITY_COLOR, lw=2.5, label="IQP parity hardware", zorder=5)
    ax.fill_between(
        q_grid,
        np.maximum(0.0, curve_mean["parity_hw"] - curve_std["parity_hw"]),
        curve_mean["parity_hw"] + curve_std["parity_hw"],
        color=PARITY_COLOR,
        alpha=0.12,
        linewidth=0.0,
    )
    ax.plot(q_grid, curve_mean["mse_hw"], color=MSE_COLOR, lw=2.3, label="IQP MSE hardware", zorder=4)
    ax.fill_between(
        q_grid,
        np.maximum(0.0, curve_mean["mse_hw"] - curve_std["mse_hw"]),
        curve_mean["mse_hw"] + curve_std["mse_hw"],
        color=MSE_COLOR,
        alpha=0.12,
        linewidth=0.0,
    )

    if not hardware_only:
        ax.plot(q_grid, curve_mean["parity_sim"], color=PARITY_COLOR, lw=2.0, ls="--", alpha=0.85, label="IQP parity simulation", zorder=3)
        ax.fill_between(
            q_grid,
            np.maximum(0.0, curve_mean["parity_sim"] - curve_std["parity_sim"]),
            curve_mean["parity_sim"] + curve_std["parity_sim"],
            color=PARITY_COLOR,
            alpha=0.07,
            linewidth=0.0,
        )
        ax.plot(q_grid, curve_mean["mse_sim"], color=MSE_COLOR, lw=1.9, ls="--", alpha=0.85, label="IQP MSE simulation", zorder=3)
        ax.fill_between(
            q_grid,
            np.maximum(0.0, curve_mean["mse_sim"] - curve_std["mse_sim"]),
            curve_mean["mse_sim"] + curve_std["mse_sim"],
            color=MSE_COLOR,
            alpha=0.07,
            linewidth=0.0,
        )

    _style_ax(ax, q_grid)
    if not paper:
        ax.set_title(title)
    if paper:
        ax.set_ylim(0.0, 1.0)
    if paper:
        handles = [
            Line2D([0], [0], color=TARGET_COLOR, lw=2.1, label=r"Target $p^*$"),
            Line2D([0], [0], color=PARITY_COLOR, lw=2.5, label="IQP parity (hw)"),
            Line2D([0], [0], color=MSE_COLOR, lw=2.3, label="IQP MSE (hw)"),
            Line2D([0], [0], color=PARITY_COLOR, lw=2.0, ls="--", label="IQP parity (sim)"),
            Line2D([0], [0], color=MSE_COLOR, lw=1.9, ls="--", label="IQP MSE (sim)"),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.7, ls=":", label="Uniform"),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="#bfbfbf",
            framealpha=1.0,
            fontsize=6.8,
        ).set_zorder(20)
    else:
        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="#bfbfbf",
            framealpha=1.0,
            fontsize=6.5,
        ).set_zorder(20)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    save_pdf(fig, out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 16: mean/std recovery curves from Experiment 15 hardware counts.")
    ap.add_argument("--experiment12-dir", type=str, default=str(DEFAULT_EXPERIMENT12_DIR))
    ap.add_argument("--experiment15-dir", type=str, default=str(DEFAULT_EXPERIMENT15_DIR))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--qmax", type=int, default=2000)
    ap.add_argument("--elite-frac", type=float, default=ELITE_FRAC)
    ap.add_argument("--hardware-only", action="store_true")
    ap.add_argument("--hide-spectral", action="store_true")
    ap.add_argument("--average-common-seeds", action="store_true")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()

    exp12_dir = Path(args.experiment12_dir).expanduser()
    if not exp12_dir.is_absolute():
        exp12_dir = ROOT / exp12_dir
    exp15_dir = Path(args.experiment15_dir).expanduser()
    if not exp15_dir.is_absolute():
        exp15_dir = ROOT / exp15_dir
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    exp12_summary = json.loads((exp12_dir / "experiment_12_global_best_iqp_vs_mse_summary.json").read_text(encoding="utf-8"))
    exp12_data = np.load(exp12_dir / "experiment_12_global_best_iqp_vs_mse_data.npz", allow_pickle=False)
    exp15_summary = json.loads((exp15_dir / "experiment_15_ibm_hardware_seedwise_best_coverage_summary.json").read_text(encoding="utf-8"))
    exp15_data = np.load(exp15_dir / "experiment_15_ibm_hardware_seedwise_best_coverage_data.npz", allow_pickle=False)
    exp15_rows = _load_csv_rows(exp15_dir / "experiment_15_ibm_hardware_seedwise_best_coverage_per_seed_metrics.csv")

    common_completed_seeds = _common_completed_seeds(exp15_rows)
    if not common_completed_seeds:
        raise RuntimeError("No common completed parity/MSE hardware seeds available in Experiment 15.")

    if args.average_common_seeds:
        selected_seeds = [int(s) for s in common_completed_seeds]
        selected_seed_reason = "average_common_completed_seeds"
    else:
        requested_seed = int(args.seed) if args.seed is not None else int(common_completed_seeds[0])
        if requested_seed not in common_completed_seeds:
            raise ValueError(f"Requested seed {requested_seed} is not available in the completed hardware pairs: {common_completed_seeds}")
        selected_seeds = [requested_seed]
        selected_seed_reason = "explicit_or_first_completed"

    seeds = np.asarray(exp12_data["seeds"], dtype=np.int64)
    train_indices = np.asarray(exp12_data["train_indices"], dtype=np.int64)
    n = int(exp12_summary["n"])
    beta = float(exp12_summary["beta"])
    sigma_values = np.asarray(exp12_data["sigma_values"], dtype=np.float64)
    k_values = np.asarray(exp12_data["k_values"], dtype=np.int64)
    parity_kl_grid = np.asarray(exp12_data["parity_kl_grid"], dtype=np.float64)
    qideal_cube = np.asarray(exp15_data["qideal_cube"], dtype=np.float64)
    qhat_cube = np.asarray(exp15_data["qhat_cube"], dtype=np.float64)
    model_keys = [str(x) for x in exp15_data["model_keys"].tolist()]

    flat_idx = np.argmin(parity_kl_grid.reshape(parity_kl_grid.shape[0], -1), axis=1)
    sigma_idx, k_idx = np.unravel_index(flat_idx, parity_kl_grid.shape[1:])
    seedwise_sigma = np.asarray(sigma_values[sigma_idx], dtype=np.float64)
    seedwise_k = np.asarray(k_values[k_idx], dtype=np.int64)

    q_grid = _q_grid(int(args.qmax))
    p_star, support, scores = build_target_distribution_paper(n, beta)
    elite_mask = topk_mask_by_scores(scores, support, frac=float(args.elite_frac))
    q_uniform = np.ones_like(p_star, dtype=np.float64) / float(p_star.size)
    bits_table = make_bits_table(n)

    parity_idx = model_keys.index("parity_seedwise_best")
    mse_idx = model_keys.index("iqp_mse")

    curve_stack: Dict[str, List[np.ndarray]] = {
        "target": [],
        "uniform": [],
        "spectral": [],
        "parity_sim": [],
        "parity_hw": [],
        "mse_sim": [],
        "mse_hw": [],
    }
    elite_unseen_counts: List[int] = []

    for seed in selected_seeds:
        seed_idx = int(np.where(seeds == int(seed))[0][0])
        idxs_train = np.asarray(train_indices[seed_idx], dtype=np.int64)
        emp = empirical_dist(idxs_train, p_star.size)
        seen_mask = np.zeros_like(support, dtype=bool)
        seen_mask[np.unique(idxs_train)] = True
        elite_unseen_mask = np.asarray(elite_mask & (~seen_mask), dtype=bool)
        elite_unseen_count = int(np.sum(elite_unseen_mask))
        if elite_unseen_count <= 0:
            raise RuntimeError(f"Elite unseen set is empty for seed {seed}.")
        elite_unseen_counts.append(int(elite_unseen_count))

        if not args.hide_spectral:
            sigma = float(seedwise_sigma[seed_idx])
            kval = int(seedwise_k[seed_idx])
            alphas = sample_alphas(n, sigma, kval, seed=int(seed) + int(PARITY_BAND_OFFSET))
            parity_matrix = build_parity_matrix(alphas, bits_table)
            z_data = parity_matrix @ emp
            q_spectral = _reconstruct_bandlimited(parity_matrix, z_data, n)
        else:
            q_spectral = np.zeros_like(p_star, dtype=np.float64)

        curve_stack["target"].append(_expected_unique_fraction(p_star, elite_unseen_mask, q_grid))
        curve_stack["uniform"].append(_expected_unique_fraction(q_uniform, elite_unseen_mask, q_grid))
        curve_stack["spectral"].append(_expected_unique_fraction(q_spectral, elite_unseen_mask, q_grid))
        curve_stack["parity_sim"].append(_expected_unique_fraction(np.asarray(qideal_cube[parity_idx, seed_idx], dtype=np.float64), elite_unseen_mask, q_grid))
        curve_stack["parity_hw"].append(_expected_unique_fraction(np.asarray(qhat_cube[parity_idx, seed_idx], dtype=np.float64), elite_unseen_mask, q_grid))
        curve_stack["mse_sim"].append(_expected_unique_fraction(np.asarray(qideal_cube[mse_idx, seed_idx], dtype=np.float64), elite_unseen_mask, q_grid))
        curve_stack["mse_hw"].append(_expected_unique_fraction(np.asarray(qhat_cube[mse_idx, seed_idx], dtype=np.float64), elite_unseen_mask, q_grid))

    curve_arrays = {key: np.asarray(curve_stack[key], dtype=np.float64) for key in curve_stack}
    curve_mean = {key: np.mean(vals, axis=0) for key, vals in curve_arrays.items()}
    curve_std = {key: np.std(vals, axis=0, ddof=1) if vals.shape[0] > 1 else np.zeros_like(vals[0]) for key, vals in curve_arrays.items()}
    elite_unseen_mean = float(np.mean(np.asarray(elite_unseen_counts, dtype=np.float64)))

    curve_rows: List[Dict[str, object]] = []
    for idx, q_value in enumerate(q_grid.tolist()):
        for key in curve_mean:
            curve_rows.append(
                {
                    "seed_group": ",".join(str(int(s)) for s in selected_seeds),
                    "curve_key": key,
                    "Q": int(q_value),
                    "mean_R_Q": float(curve_mean[key][idx]),
                    "std_R_Q": float(curve_std[key][idx]),
                    "n_aggregated_seeds": int(len(selected_seeds)),
                }
            )

    if len(selected_seeds) > 1:
        stem = f"{OUTPUT_STEM}_avg_common_seeds"
    else:
        stem = f"{OUTPUT_STEM}_seed{int(selected_seeds[0])}"
    if args.hardware_only:
        stem = f"{stem}_hardware_only"
    if args.hide_spectral:
        stem = f"{stem}_no_spectral"
    if args.paper:
        stem = f"{stem}_paper"

    curve_csv = outdir / f"{stem}.csv"
    curve_npz = outdir / f"{stem}.npz"
    summary_json = outdir / f"{stem}_summary.json"
    plot_pdf = outdir / f"{stem}.pdf"
    plot_png = outdir / f"{stem}.png"

    _write_csv(curve_csv, curve_rows)
    np.savez_compressed(
        curve_npz,
        selected_seeds=np.asarray(selected_seeds, dtype=np.int64),
        q_grid=q_grid,
        elite_unseen_count_mean=np.asarray([elite_unseen_mean], dtype=np.float64),
        target_mean=np.asarray(curve_mean["target"], dtype=np.float64),
        target_std=np.asarray(curve_std["target"], dtype=np.float64),
        uniform_mean=np.asarray(curve_mean["uniform"], dtype=np.float64),
        uniform_std=np.asarray(curve_std["uniform"], dtype=np.float64),
        spectral_mean=np.asarray(curve_mean["spectral"], dtype=np.float64),
        spectral_std=np.asarray(curve_std["spectral"], dtype=np.float64),
        parity_sim_mean=np.asarray(curve_mean["parity_sim"], dtype=np.float64),
        parity_sim_std=np.asarray(curve_std["parity_sim"], dtype=np.float64),
        parity_hw_mean=np.asarray(curve_mean["parity_hw"], dtype=np.float64),
        parity_hw_std=np.asarray(curve_std["parity_hw"], dtype=np.float64),
        mse_sim_mean=np.asarray(curve_mean["mse_sim"], dtype=np.float64),
        mse_sim_std=np.asarray(curve_std["mse_sim"], dtype=np.float64),
        mse_hw_mean=np.asarray(curve_mean["mse_hw"], dtype=np.float64),
        mse_hw_std=np.asarray(curve_std["mse_hw"], dtype=np.float64),
    )

    title = f"Seedwise-best recovery curve ({len(selected_seeds)} seed{'s' if len(selected_seeds) != 1 else ''})"
    _render_plot(
        out_pdf=plot_pdf,
        out_png=plot_png,
        q_grid=q_grid,
        curve_mean=curve_mean,
        curve_std=curve_std,
        title=title,
        hardware_only=bool(args.hardware_only),
        hide_spectral=bool(args.hide_spectral),
        paper=bool(args.paper),
    )

    summary_payload = {
        "script": SCRIPT_REL,
        "status": str(exp15_summary.get("status", "unknown")),
        "selected_seed_reason": selected_seed_reason,
        "selected_seeds": selected_seeds,
        "n_selected_seeds": int(len(selected_seeds)),
        "common_completed_seeds": common_completed_seeds,
        "beta": float(beta),
        "n": int(n),
        "train_m": int(exp12_summary["train_m"]),
        "elite_frac": float(args.elite_frac),
        "elite_unseen_count_mean": float(elite_unseen_mean),
        "experiment12_dir": _try_rel(exp12_dir),
        "experiment15_dir": _try_rel(exp15_dir),
        "curve_csv": _try_rel(curve_csv),
        "curve_npz": _try_rel(curve_npz),
        "plot_pdf": _try_rel(plot_pdf),
        "plot_png": _try_rel(plot_png),
        "hardware_only": bool(args.hardware_only),
        "hide_spectral": bool(args.hide_spectral),
        "paper": bool(args.paper),
    }
    _write_json(summary_json, summary_payload)
    _write_json(
        outdir / "RUN_CONFIG.json",
        {
            "script": SCRIPT_REL,
            "experiment12_dir": _try_rel(exp12_dir),
            "experiment15_dir": _try_rel(exp15_dir),
            "outdir": _try_rel(outdir),
            "seed": int(selected_seeds[0]) if len(selected_seeds) == 1 else None,
            "average_common_seeds": bool(args.average_common_seeds),
            "qmax": int(args.qmax),
            "elite_frac": float(args.elite_frac),
            "hardware_only": bool(args.hardware_only),
            "hide_spectral": bool(args.hide_spectral),
            "paper": bool(args.paper),
            "curve_csv": _try_rel(curve_csv),
            "curve_npz": _try_rel(curve_npz),
            "plot_pdf": _try_rel(plot_pdf),
            "plot_png": _try_rel(plot_png),
        },
    )
    write_training_protocol(
        outdir,
        experiment_name="Experiment 16 seedwise-best hardware recovery curve",
        note=(
            "Aggregate recovery curves over the completed Experiment 15 seedwise-best parity hardware runs, "
            "and report mean and standard deviation across seeds for simulation and hardware."
        ),
        source_relpath=SCRIPT_REL,
        metrics_note="Stores mean/std recovery fraction R(Q) over the elite-unseen state set for target, uniform, spectral completion, and IQP simulation/hardware curves.",
    )
    print(f"[experiment16] wrote {plot_pdf}", flush=True)


if __name__ == "__main__":
    main()
