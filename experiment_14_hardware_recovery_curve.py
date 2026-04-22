#!/usr/bin/env python3
"""Experiment 14: recovery curve from Experiment 13 hardware counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from final_plot_style import apply_final_style, save_pdf
from model_labels import IQP_MSE_LABEL, IQP_PARITY_LABEL
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
    _expected_unique_fraction,
    _q_grid,
    _reconstruct_bandlimited,
    _style_ax,
    apply_final_style as apply_experiment4_style,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = Path(__file__).name
OUTPUT_STEM = "experiment_14_hardware_recovery_curve"
DEFAULT_OUTDIR = ROOT / "plots" / "experiment_14_hardware_recovery_curve"
DEFAULT_EXPERIMENT12_DIR = ROOT / "plots" / "experiment_12_global_best_iqp_vs_mse"
DEFAULT_EXPERIMENT13_DIR = ROOT / "plots" / "experiment_13_ibm_hardware_global_best_coverage"

TARGET_COLOR = "#2F2A2B"
UNIFORM_COLOR = "#C6C9CF"
PARITY_COLOR = "#DC2626"
MSE_COLOR = "#4A90E2"
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


def _choose_seed(
    *,
    requested_seed: int | None,
    common_completed_seeds: List[int],
    exp12_summary: Dict[str, object],
) -> int:
    if requested_seed is not None:
        if requested_seed not in common_completed_seeds:
            raise ValueError(
                f"Requested seed {requested_seed} is not available in the completed hardware pairs: {common_completed_seeds}"
            )
        return int(requested_seed)
    representative_seed = int(exp12_summary.get("representative_seed", common_completed_seeds[0]))
    if representative_seed in common_completed_seeds:
        return representative_seed
    return int(common_completed_seeds[0])


def _render_plot(
    *,
    out_pdf: Path,
    out_png: Path,
    q_grid: np.ndarray,
    curves: Dict[str, np.ndarray],
    labels: Dict[str, str],
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
        fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)

    ax.plot(q_grid, curves["target"], color=TARGET_COLOR, lw=2.1, zorder=6)
    ax.plot(q_grid, curves["uniform"], color=UNIFORM_COLOR, lw=1.7, ls=":", zorder=2)
    if not hide_spectral:
        ax.plot(q_grid, curves["spectral"], color=SPECTRAL_COLOR, lw=2.0, ls="-.", zorder=4)
    ax.plot(q_grid, curves["parity_hw"], color=PARITY_COLOR, lw=2.6, zorder=5)
    ax.plot(q_grid, curves["mse_hw"], color=MSE_COLOR, lw=2.3, zorder=4)
    if not hardware_only:
        ax.plot(q_grid, curves["parity_sim"], color=PARITY_COLOR, lw=2.0, ls="--", alpha=0.75, zorder=4)
        ax.plot(q_grid, curves["mse_sim"], color=MSE_COLOR, lw=1.8, ls="--", alpha=0.75, zorder=3)

    _style_ax(ax, q_grid)
    if not paper:
        ax.set_title(title)
    if paper:
        ax.set_ylim(0.0, 1.0)
    if hardware_only:
        handles = [
            Line2D([0], [0], color=TARGET_COLOR, lw=2.1, label=labels["target"]),
            Line2D([0], [0], color=PARITY_COLOR, lw=2.6, label=labels["parity_hw"]),
            Line2D([0], [0], color=MSE_COLOR, lw=2.3, label=labels["mse_hw"]),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.7, ls=":", label=labels["uniform"]),
        ]
        if not hide_spectral:
            handles.insert(3, Line2D([0], [0], color=SPECTRAL_COLOR, lw=2.0, ls="-.", label=labels["spectral"]))
    else:
        handles = [
            Line2D([0], [0], color=TARGET_COLOR, lw=2.1, label=labels["target"]),
            Line2D([0], [0], color=PARITY_COLOR, lw=2.0, ls="--", label=labels["parity_sim"]),
            Line2D([0], [0], color=PARITY_COLOR, lw=2.6, label=labels["parity_hw"]),
            Line2D([0], [0], color=MSE_COLOR, lw=1.8, ls="--", label=labels["mse_sim"]),
            Line2D([0], [0], color=MSE_COLOR, lw=2.3, label=labels["mse_hw"]),
            Line2D([0], [0], color=UNIFORM_COLOR, lw=1.7, ls=":", label=labels["uniform"]),
        ]
        if not hide_spectral:
            handles.insert(5, Line2D([0], [0], color=SPECTRAL_COLOR, lw=2.0, ls="-.", label=labels["spectral"]))
    ax.legend(
        handles=handles,
        loc="lower right" if paper else "upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#bfbfbf",
        fontsize=6.8 if paper else 6.6,
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    save_pdf(fig, out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 14: recovery curve from Experiment 13 hardware counts.")
    ap.add_argument("--experiment12-dir", type=str, default=str(DEFAULT_EXPERIMENT12_DIR))
    ap.add_argument("--experiment13-dir", type=str, default=str(DEFAULT_EXPERIMENT13_DIR))
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
    exp13_dir = Path(args.experiment13_dir).expanduser()
    if not exp13_dir.is_absolute():
        exp13_dir = ROOT / exp13_dir
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    exp12_summary = json.loads((exp12_dir / "experiment_12_global_best_iqp_vs_mse_summary.json").read_text(encoding="utf-8"))
    exp12_data = np.load(exp12_dir / "experiment_12_global_best_iqp_vs_mse_data.npz", allow_pickle=False)
    exp13_data = np.load(exp13_dir / "experiment_13_ibm_hardware_global_best_coverage_data.npz", allow_pickle=False)
    exp13_rows = _load_csv_rows(exp13_dir / "experiment_13_ibm_hardware_global_best_coverage_per_seed_metrics.csv")
    exp13_summary = json.loads((exp13_dir / "experiment_13_ibm_hardware_global_best_coverage_summary.json").read_text(encoding="utf-8"))

    by_model: Dict[str, List[Dict[str, str]]] = {}
    for row in exp13_rows:
        by_model.setdefault(str(row["model_key"]), []).append(row)
    common_completed_seeds = sorted(
        set(int(r["seed"]) for r in by_model.get("parity_global_best", []))
        & set(int(r["seed"]) for r in by_model.get("iqp_mse", []))
    )
    if not common_completed_seeds:
        raise RuntimeError("No common completed parity/MSE hardware seeds available in Experiment 13.")

    seeds = np.asarray(exp12_data["seeds"], dtype=np.int64)

    n = int(exp12_summary["n"])
    beta = float(exp12_summary["beta"])
    global_best_sigma = float(exp12_summary["global_best_sigma"])
    global_best_k = int(exp12_summary["global_best_k"])
    train_indices = np.asarray(exp12_data["train_indices"], dtype=np.int64)
    qideal_cube = np.asarray(exp13_data["qideal_cube"], dtype=np.float64)
    qhat_cube = np.asarray(exp13_data["qhat_cube"], dtype=np.float64)
    model_keys = [str(x) for x in exp13_data["model_keys"].tolist()]

    bits_table = make_bits_table(n)
    q_grid = _q_grid(int(args.qmax))
    parity_model_idx = model_keys.index("parity_global_best")
    mse_model_idx = model_keys.index("iqp_mse")
    p_star, support, scores = build_target_distribution_paper(n, beta)
    elite_mask = topk_mask_by_scores(scores, support, frac=float(args.elite_frac))
    q_uniform = np.ones_like(p_star, dtype=np.float64) / float(p_star.size)

    if bool(args.average_common_seeds):
        selected_seeds = [int(s) for s in common_completed_seeds]
        selected_seed_reason = "average_common_completed_seeds"
    else:
        selected_seed = _choose_seed(
            requested_seed=args.seed,
            common_completed_seeds=common_completed_seeds,
            exp12_summary=exp12_summary,
        )
        selected_seeds = [int(selected_seed)]
        selected_seed_reason = (
            "experiment12_representative_seed"
            if int(selected_seed) == int(exp12_summary.get("representative_seed", -1))
            else "explicit_or_first_completed"
        )

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

        alphas = sample_alphas(n, global_best_sigma, global_best_k, seed=int(seed) + int(PARITY_BAND_OFFSET))
        parity_matrix = build_parity_matrix(alphas, bits_table)
        z_data = parity_matrix @ emp
        q_spectral = _reconstruct_bandlimited(parity_matrix, z_data, n)

        q_parity_sim = np.asarray(qideal_cube[parity_model_idx, seed_idx], dtype=np.float64)
        q_parity_hw = np.asarray(qhat_cube[parity_model_idx, seed_idx], dtype=np.float64)
        q_mse_sim = np.asarray(qideal_cube[mse_model_idx, seed_idx], dtype=np.float64)
        q_mse_hw = np.asarray(qhat_cube[mse_model_idx, seed_idx], dtype=np.float64)

        curve_stack["target"].append(_expected_unique_fraction(p_star, elite_unseen_mask, q_grid))
        curve_stack["uniform"].append(_expected_unique_fraction(q_uniform, elite_unseen_mask, q_grid))
        curve_stack["spectral"].append(_expected_unique_fraction(q_spectral, elite_unseen_mask, q_grid))
        curve_stack["parity_sim"].append(_expected_unique_fraction(q_parity_sim, elite_unseen_mask, q_grid))
        curve_stack["parity_hw"].append(_expected_unique_fraction(q_parity_hw, elite_unseen_mask, q_grid))
        curve_stack["mse_sim"].append(_expected_unique_fraction(q_mse_sim, elite_unseen_mask, q_grid))
        curve_stack["mse_hw"].append(_expected_unique_fraction(q_mse_hw, elite_unseen_mask, q_grid))

    curves = {
        key: np.mean(np.asarray(curve_stack[key], dtype=np.float64), axis=0)
        for key in curve_stack
    }
    elite_unseen_count = int(round(float(np.mean(elite_unseen_counts))))

    curve_rows: List[Dict[str, object]] = []
    for idx, q_value in enumerate(q_grid.tolist()):
        curve_rows.extend(
            [
                {
                    "seed": ",".join(str(int(s)) for s in selected_seeds) if len(selected_seeds) > 1 else int(selected_seeds[0]),
                    "curve_key": key,
                    "Q": int(q_value),
                    "R_Q": float(curve[idx]),
                    "n_aggregated_seeds": int(len(selected_seeds)),
                }
                for key, curve in curves.items()
            ]
        )

    if len(selected_seeds) > 1:
        stem = f"{OUTPUT_STEM}_avg_common_seeds"
    else:
        stem = f"{OUTPUT_STEM}_seed{int(selected_seeds[0])}"
    if bool(args.hardware_only):
        stem = f"{stem}_hardware_only"
    if bool(args.hide_spectral):
        stem = f"{stem}_no_spectral"
    if bool(args.paper):
        stem = f"{stem}_paper"
    curve_csv = outdir / f"{stem}.csv"
    curve_npz = outdir / f"{stem}.npz"
    summary_json = outdir / f"{stem}_summary.json"
    plot_pdf = outdir / f"{stem}.pdf"
    plot_png = outdir / f"{stem}.png"

    _write_csv(curve_csv, curve_rows)
    np.savez_compressed(
        curve_npz,
        seed=np.asarray([int(seed)], dtype=np.int64),
        q_grid=q_grid,
        elite_unseen_mask=np.asarray(elite_unseen_mask, dtype=np.int8),
        elite_unseen_count=np.asarray([int(elite_unseen_count)], dtype=np.int64),
        target_curve=np.asarray(curves["target"], dtype=np.float64),
        uniform_curve=np.asarray(curves["uniform"], dtype=np.float64),
        spectral_curve=np.asarray(curves["spectral"], dtype=np.float64),
        parity_sim_curve=np.asarray(curves["parity_sim"], dtype=np.float64),
        parity_hw_curve=np.asarray(curves["parity_hw"], dtype=np.float64),
        mse_sim_curve=np.asarray(curves["mse_sim"], dtype=np.float64),
        mse_hw_curve=np.asarray(curves["mse_hw"], dtype=np.float64),
        global_best_sigma=np.asarray([global_best_sigma], dtype=np.float64),
        global_best_k=np.asarray([global_best_k], dtype=np.int64),
    )

    labels = {
        "target": r"Target $p^*$",
        "uniform": "Uniform",
        "spectral": f"Spectral completion (sigma={global_best_sigma:g}, K={global_best_k})",
        "parity_sim": f"{IQP_PARITY_LABEL} simulation",
        "parity_hw": f"{IQP_PARITY_LABEL} hardware",
        "mse_sim": f"{IQP_MSE_LABEL} simulation",
        "mse_hw": f"{IQP_MSE_LABEL} hardware",
    }
    if len(selected_seeds) > 1:
        title = f"Recovery curve (mean over {len(selected_seeds)} seeds)"
    else:
        title = f"Recovery curve (seed {int(selected_seeds[0])})"
    _render_plot(
        out_pdf=plot_pdf,
        out_png=plot_png,
        q_grid=q_grid,
        curves=curves,
        labels=labels,
        title=title,
        hardware_only=bool(args.hardware_only),
        hide_spectral=bool(args.hide_spectral),
        paper=bool(args.paper),
    )

    summary_payload = {
        "script": SCRIPT_REL,
        "status": str(exp13_summary.get("status", "unknown")),
        "seed": int(selected_seeds[0]) if len(selected_seeds) == 1 else None,
        "aggregated_seeds": selected_seeds,
        "n_aggregated_seeds": int(len(selected_seeds)),
        "common_completed_seeds": common_completed_seeds,
        "selected_seed_reason": selected_seed_reason,
        "beta": float(beta),
        "n": int(n),
        "train_m": int(exp12_summary["train_m"]),
        "elite_frac": float(args.elite_frac),
        "elite_unseen_count": int(elite_unseen_count),
        "global_best_sigma": float(global_best_sigma),
        "global_best_k": int(global_best_k),
        "experiment12_dir": _try_rel(exp12_dir),
        "experiment13_dir": _try_rel(exp13_dir),
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
            "experiment13_dir": _try_rel(exp13_dir),
            "outdir": _try_rel(outdir),
            "seed": int(selected_seeds[0]) if len(selected_seeds) == 1 else None,
            "aggregated_seeds": selected_seeds,
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
        experiment_name="Experiment 14 hardware recovery curve",
        note=(
            "Build a single-seed recovery curve from Experiment 13 hardware counts using the same elite-unseen set as in "
            "Experiment 4, together with simulation curves for the same fixed `(sigma, K)` setting."
        ),
        source_relpath=SCRIPT_REL,
        metrics_note="Stores expected recovery fraction R(Q) over the elite-unseen state set for target, uniform, spectral completion, and IQP simulation/hardware curves.",
    )
    print(f"[experiment14] wrote {plot_pdf}", flush=True)


if __name__ == "__main__":
    main()
