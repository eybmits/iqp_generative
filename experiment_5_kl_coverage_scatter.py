#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 5: standalone KL-vs-coverage scatter across the beta sweep.

This script is self-contained and supports two modes:

1. aggregate: join the per-seed KL metrics from Experiment 2 with the per-seed
   quality-coverage metrics from Experiment 3 and build the fixed-beta scatter.
2. rerender: rebuild the final PDF directly from a saved scatter CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_5_kl_coverage_scatter.py"

DEFAULT_KL_METRICS_CSV = ROOT / "plots" / "experiment_2_beta_kl_summary" / "experiment_2_beta_kl_summary_metrics_per_seed.csv"
DEFAULT_COVERAGE_METRICS_CSV = ROOT / "plots" / "experiment_3_beta_quality_coverage" / "experiment_3_beta_quality_coverage_metrics_per_seed.csv"
DEFAULT_SCATTER_CSV = ""
DEFAULT_OUTDIR = ROOT / "plots" / "experiment_5_kl_coverage_scatter"
STEM = "experiment_5_kl_coverage_scatter"

FIG_W = 270.0 / 72.0
FIG_H = 185.52 / 72.0

MODEL_ORDER = [
    "classical_transformer_mle",
    "classical_dense_fields_xent",
    "iqp_parity_mse",
    "classical_nnn_fields_parity",
    "classical_maxent_parity",
]

MODEL_STYLE = {
    "iqp_parity_mse": {
        "label": "IQP (parity)",
        "color": "#D62728",
    },
    "classical_nnn_fields_parity": {
        "label": "Ising+fields (NN+NNN)",
        "color": "#1f77b4",
    },
    "classical_dense_fields_xent": {
        "label": "Dense Ising+fields (xent)",
        "color": "#8c564b",
    },
    "classical_transformer_mle": {
        "label": "AR Transformer (MLE)",
        "color": "#17becf",
    },
    "classical_maxent_parity": {
        "label": "MaxEnt parity (P,z)",
        "color": "#9467bd",
    },
}


def apply_final_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 7.2,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "gray",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.03,
        }
    )


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _try_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _coverage_column_for_budget(budget_q: int) -> str:
    if int(budget_q) == 10000:
        return "quality_coverage_Q10000"
    return f"quality_coverage_Q{int(budget_q)}"


def load_scatter_rows(kl_metrics_csv: Path, coverage_metrics_csv: Path, *, budget_q: int) -> List[Dict[str, object]]:
    coverage_col = _coverage_column_for_budget(int(budget_q))
    grouped: Dict[tuple[float, str], Dict[str, List[float]]] = defaultdict(lambda: {"kl": [], "cov": []})

    kl_by_instance: Dict[tuple[float, int, str], float] = {}
    with kl_metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"beta", "seed", "model_key", "KL_pstar_to_q"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"KL metrics CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if model_key not in MODEL_STYLE:
                continue
            key = (float(row["beta"]), int(row["seed"]), model_key)
            kl_by_instance[key] = float(row["KL_pstar_to_q"])

    matched_rows = 0
    with coverage_metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"beta", "seed", "model_key", coverage_col}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Coverage metrics CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if model_key not in MODEL_STYLE:
                continue
            key = (float(row["beta"]), int(row["seed"]), model_key)
            if key not in kl_by_instance:
                continue
            grouped[(key[0], model_key)]["kl"].append(float(kl_by_instance[key]))
            grouped[(key[0], model_key)]["cov"].append(float(row[coverage_col]))
            matched_rows += 1

    if matched_rows == 0:
        raise ValueError("No overlapping KL and coverage rows found for the scatter plot.")

    rows_out: List[Dict[str, object]] = []
    for (beta, model_key), vals in sorted(grouped.items()):
        kl_arr = np.asarray(vals["kl"], dtype=np.float64)
        cov_arr = np.asarray(vals["cov"], dtype=np.float64)
        kl_arr = kl_arr[np.isfinite(kl_arr)]
        cov_arr = cov_arr[np.isfinite(cov_arr)]
        if kl_arr.size == 0 or cov_arr.size == 0:
            continue
        rows_out.append(
            {
                "beta": float(beta),
                "model_key": model_key,
                "model_label": str(MODEL_STYLE[model_key]["label"]),
                "n_seeds": int(min(kl_arr.size, cov_arr.size)),
                "kl_mean": float(np.mean(kl_arr)),
                "coverage_mean": float(np.mean(cov_arr)),
            }
        )
    if not rows_out:
        raise ValueError("No usable aggregated rows found for the scatter plot.")
    return rows_out


def load_scatter_csv(scatter_csv: Path) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    with scatter_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"beta", "model_key", "model_label", "n_seeds", "kl_mean", "coverage_mean"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Scatter CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if model_key not in MODEL_STYLE:
                continue
            rows_out.append(
                {
                    "beta": float(row["beta"]),
                    "model_key": model_key,
                    "model_label": str(row["model_label"]),
                    "n_seeds": int(float(row["n_seeds"])),
                    "kl_mean": float(row["kl_mean"]),
                    "coverage_mean": float(row["coverage_mean"]),
                }
            )
    if not rows_out:
        raise ValueError(f"No usable scatter rows found in {scatter_csv}")
    return rows_out


def _legend_handles() -> List[Line2D]:
    handles: List[Line2D] = []
    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=4.6,
                markerfacecolor=str(style["color"]),
                markeredgecolor="none",
                label=str(style["label"]),
            )
        )
    return handles


def _negative_slope_count(rows: Sequence[Dict[str, object]]) -> tuple[int, int]:
    by_beta: Dict[float, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_beta[float(row["beta"])].append(dict(row))
    neg = 0
    total = 0
    for beta, beta_rows in sorted(by_beta.items()):
        xs = np.asarray([float(r["kl_mean"]) for r in beta_rows], dtype=np.float64)
        ys = np.asarray([float(r["coverage_mean"]) for r in beta_rows], dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys)
        if np.sum(valid) < 2:
            continue
        total += 1
        slope = np.polyfit(xs[valid], ys[valid], deg=1)[0]
        if slope < 0:
            neg += 1
    return neg, total


def render_scatter(rows: Sequence[Dict[str, object]], *, budget_q: int, out_pdf: Path) -> None:
    apply_final_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    by_beta: Dict[float, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_beta[float(row["beta"])].append(dict(row))

    all_x: List[float] = []
    all_y: List[float] = []
    for beta, beta_rows in sorted(by_beta.items()):
        beta_rows_sorted = sorted(beta_rows, key=lambda r: float(r["kl_mean"]))
        xs = np.asarray([float(r["kl_mean"]) for r in beta_rows_sorted], dtype=np.float64)
        ys = np.asarray([float(r["coverage_mean"]) for r in beta_rows_sorted], dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys)
        if np.sum(valid) >= 2:
            ax.plot(
                xs[valid],
                ys[valid],
                color="#B5B5B5",
                lw=0.95,
                alpha=0.82,
                zorder=1,
            )
            all_x.extend(xs[valid].tolist())
            all_y.extend(ys[valid].tolist())

    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        xs = np.asarray([float(r["kl_mean"]) for r in model_rows], dtype=np.float64)
        ys = np.asarray([float(r["coverage_mean"]) for r in model_rows], dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys)
        ax.scatter(
            xs[valid],
            ys[valid],
            s=18,
            color=str(style["color"]),
            edgecolors="none",
            alpha=0.95,
            zorder=3,
        )
        all_x.extend(xs[valid].tolist())
        all_y.extend(ys[valid].tolist())

    ax.set_xlabel(r"$D_{\mathrm{KL}}(p^{*}\,\|\,q)$")
    budget_label = f"{int(budget_q):,}".replace(",", "{,}")
    ax.set_ylabel(rf"$C_q({budget_label})$")
    ax.grid(True, alpha=0.16, linestyle="--")
    ax.ticklabel_format(axis="both", style="plain")

    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        xr = max(1e-6, xmax - xmin)
        ax.set_xlim(max(0.0, xmin - 0.05 * xr), xmax + 0.08 * xr)
    if all_y:
        ymin, ymax = min(all_y), max(all_y)
        yr = max(1e-6, ymax - ymin)
        ax.set_ylim(max(0.0, ymin - 0.08 * yr), ymax + 0.10 * yr)

    legend = ax.legend(
        handles=_legend_handles(),
        loc="upper right",
        frameon=True,
        fontsize=6.8,
        facecolor="white",
        edgecolor="#bfbfbf",
        borderpad=0.24,
        labelspacing=0.20,
        handlelength=0.9,
        handletextpad=0.42,
        borderaxespad=0.20,
    )
    legend.set_zorder(100)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def run() -> None:
    ap = argparse.ArgumentParser(description="Experiment 5: standalone KL-vs-coverage scatter.")
    ap.add_argument("--kl-metrics-csv", type=str, default=str(DEFAULT_KL_METRICS_CSV))
    ap.add_argument("--coverage-metrics-csv", type=str, default=str(DEFAULT_COVERAGE_METRICS_CSV))
    ap.add_argument("--scatter-csv", type=str, default=str(DEFAULT_SCATTER_CSV))
    ap.add_argument("--coverage-budget", type=int, default=1000, choices=[1000, 2000, 5000, 10000])
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--stem", type=str, default=STEM)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    budget_q = int(args.coverage_budget)
    scatter_csv_arg = str(args.scatter_csv).strip()
    if scatter_csv_arg and Path(scatter_csv_arg).exists():
        input_mode = "scatter_csv"
        source_path = Path(scatter_csv_arg)
        scatter_rows = load_scatter_csv(source_path)
    else:
        input_mode = "paired_metrics_csv"
        kl_metrics_csv = Path(args.kl_metrics_csv)
        coverage_metrics_csv = Path(args.coverage_metrics_csv)
        if not kl_metrics_csv.exists():
            raise FileNotFoundError(f"Missing KL metrics csv: {kl_metrics_csv}")
        if not coverage_metrics_csv.exists():
            raise FileNotFoundError(f"Missing coverage metrics csv: {coverage_metrics_csv}")
        source_path = kl_metrics_csv
        scatter_rows = load_scatter_rows(kl_metrics_csv, coverage_metrics_csv, budget_q=budget_q)

    stem = str(args.stem)
    out_pdf = outdir / f"{stem}.pdf"
    out_csv = outdir / f"{stem}_scatter.csv"

    render_scatter(scatter_rows, budget_q=budget_q, out_pdf=out_pdf)
    _write_csv(out_csv, list(scatter_rows))
    _write_json(
        outdir / "RUN_CONFIG.json",
        {
            "script": SCRIPT_REL,
            "input_mode": input_mode,
            "source_path": _try_rel(source_path),
            "kl_metrics_csv": _try_rel(Path(args.kl_metrics_csv)),
            "coverage_metrics_csv": _try_rel(Path(args.coverage_metrics_csv)),
            "outdir": _try_rel(outdir),
            "stem": stem,
            "coverage_budget": budget_q,
            "figure_width_in": FIG_W,
            "figure_height_in": FIG_H,
            "model_order": MODEL_ORDER,
            "pdf_only": True,
        },
    )
    _write_json(
        outdir / "RERENDER_CONFIG.json",
        {
            "script": SCRIPT_REL,
            "scatter_csv": _try_rel(out_csv),
            "outdir": _try_rel(outdir),
            "rerender_command": f"MPLCONFIGDIR=/tmp/mpl-cache python {SCRIPT_REL} --scatter-csv {_try_rel(out_csv)} --outdir {_try_rel(outdir)} --stem {stem}",
        },
    )

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    run()
