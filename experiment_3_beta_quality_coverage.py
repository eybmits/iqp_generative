#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 3: standalone beta-vs-quality-coverage summary plots.

This script is intentionally standalone. It renders three PDF panels for
Q=1000, 2000, 5000 using the already aggregated quality-coverage summary CSVs.
It also stores normalized series CSVs and rerender metadata so the plots can be
adjusted later without recomputing the underlying experiment.
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
SCRIPT_REL = "experiment_3_beta_quality_coverage.py"
DEFAULT_OUTDIR = ROOT / "plots" / "experiment_3_beta_quality_coverage"
STEM_PREFIX = "experiment_3_beta_quality_coverage"

FIG_W = 270.0 / 72.0
FIG_H = 185.52 / 72.0

LEGEND_FONTSIZE = 7.2
BAND_ALPHA = 0.14
MAX_LABELED_X_TICKS = 10
X_LABEL_EVERY_N_BETAS = 3

DEFAULT_METRICS = {
    1000: ROOT / "plots" / "experiment_3_beta_quality_coverage" / "experiment_3_beta_quality_coverage_q1000_series.csv",
    2000: ROOT / "plots" / "experiment_3_beta_quality_coverage" / "experiment_3_beta_quality_coverage_q2000_series.csv",
    5000: ROOT / "plots" / "experiment_3_beta_quality_coverage" / "experiment_3_beta_quality_coverage_q5000_series.csv",
}

MODEL_ORDER = [
    "iqp_parity_mse",
    "classical_nnn_fields_parity",
    "classical_dense_fields_xent",
    "classical_transformer_mle",
    "classical_maxent_parity",
]

MODEL_STYLE = {
    "iqp_parity_mse": {
        "label": "IQP (parity)",
        "color": "#D62728",
        "ls": "-",
        "lw": 2.35,
    },
    "classical_nnn_fields_parity": {
        "label": "Ising+fields (NN+NNN)",
        "color": "#1f77b4",
        "ls": "-",
        "lw": 1.85,
    },
    "classical_dense_fields_xent": {
        "label": "Dense Ising+fields (xent)",
        "color": "#8c564b",
        "ls": (0, (5, 2)),
        "lw": 1.85,
    },
    "classical_transformer_mle": {
        "label": "AR Transformer (MLE)",
        "color": "#17becf",
        "ls": "--",
        "lw": 1.90,
    },
    "classical_maxent_parity": {
        "label": "MaxEnt parity (P,z)",
        "color": "#9467bd",
        "ls": "-.",
        "lw": 1.90,
    },
}


def apply_final_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": LEGEND_FONTSIZE,
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


def _major_beta_ticks(betas: Sequence[float]) -> List[float]:
    beta_vals = [float(beta) for beta in betas]
    if len(beta_vals) <= MAX_LABELED_X_TICKS:
        return beta_vals
    step = max(2, int(X_LABEL_EVERY_N_BETAS))
    major = [beta_vals[idx] for idx in range(0, len(beta_vals) - 1, step)]
    if beta_vals[-1] not in major:
        base_spacing = min(abs(beta_vals[idx + 1] - beta_vals[idx]) for idx in range(len(beta_vals) - 1))
        if major and abs(beta_vals[-1] - major[-1]) <= 1.5 * base_spacing:
            major[-1] = beta_vals[-1]
        else:
            major.append(beta_vals[-1])
    return major


def _load_series_from_metrics_csv(metrics_csv: Path) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"metric", "beta", "model_key", "model_label", "n_seeds", "q1", "mean", "q3", "min", "max"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Metrics CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if str(row["metric"]) != "quality_coverage_Q":
                continue
            if model_key not in MODEL_ORDER:
                continue
            rows_out.append(
                {
                    "beta": float(row["beta"]),
                    "model_key": model_key,
                    "model_label": str(row["model_label"]),
                    "n_seeds": int(float(row["n_seeds"])),
                    "min": float(row["min"]),
                    "q1": float(row["q1"]),
                    "mean": float(row["mean"]),
                    "q3": float(row["q3"]),
                    "max": float(row["max"]),
                }
            )
    if not rows_out:
        raise ValueError(f"No quality-coverage rows found in {metrics_csv}")
    return rows_out


def _load_series_csv(series_csv: Path) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    with series_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"beta", "model_key", "model_label", "n_seeds", "q1", "mean", "q3", "min", "max"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Series CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if model_key not in MODEL_ORDER:
                continue
            rows_out.append(
                {
                    "beta": float(row["beta"]),
                    "model_key": model_key,
                    "model_label": str(row["model_label"]),
                    "n_seeds": int(float(row["n_seeds"])),
                    "min": float(row["min"]),
                    "q1": float(row["q1"]),
                    "mean": float(row["mean"]),
                    "q3": float(row["q3"]),
                    "max": float(row["max"]),
                }
            )
    if not rows_out:
        raise ValueError(f"No usable rows found in {series_csv}")
    return rows_out


def _group_series(series_rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, np.ndarray]]:
    betas = sorted({float(row["beta"]) for row in series_rows})
    grouped: Dict[str, Dict[float, Dict[str, float]]] = defaultdict(dict)
    for row in series_rows:
        grouped[str(row["model_key"])][float(row["beta"])] = {
            "mean": float(row["mean"]),
            "q1": float(row["q1"]),
            "q3": float(row["q3"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
        }

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for model_key in MODEL_ORDER:
        mean = np.full(len(betas), np.nan, dtype=np.float64)
        q1 = np.full(len(betas), np.nan, dtype=np.float64)
        q3 = np.full(len(betas), np.nan, dtype=np.float64)
        vmin = np.full(len(betas), np.nan, dtype=np.float64)
        vmax = np.full(len(betas), np.nan, dtype=np.float64)
        for idx, beta in enumerate(betas):
            if beta not in grouped.get(model_key, {}):
                continue
            rec = grouped[model_key][beta]
            mean[idx] = float(rec["mean"])
            q1[idx] = float(rec["q1"])
            q3[idx] = float(rec["q3"])
            vmin[idx] = float(rec["min"])
            vmax[idx] = float(rec["max"])
        out[model_key] = {
            "betas": np.asarray(betas, dtype=np.float64),
            "mean": mean,
            "q1": q1,
            "q3": q3,
            "min": vmin,
            "max": vmax,
        }
    return out


def _legend_handles() -> List[Line2D]:
    handles: List[Line2D] = []
    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        handles.append(
            Line2D(
                [0],
                [0],
                color=str(style["color"]),
                lw=float(style["lw"]),
                ls=style["ls"],
                label=str(style["label"]),
            )
        )
    return handles


def render_plot(series_rows: Sequence[Dict[str, object]], *, out_pdf: Path, budget_q: int) -> None:
    apply_final_style()
    grouped = _group_series(series_rows)
    betas = grouped[MODEL_ORDER[0]]["betas"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    all_y = []
    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        payload = grouped[model_key]
        mean = payload["mean"]
        q1 = payload["q1"]
        q3 = payload["q3"]
        valid_line = np.isfinite(mean)
        valid_band = np.isfinite(q1) & np.isfinite(q3)
        if np.any(valid_band):
            ax.fill_between(
                betas[valid_band],
                q1[valid_band],
                q3[valid_band],
                color=str(style["color"]),
                alpha=BAND_ALPHA,
                linewidth=0.0,
                zorder=1,
            )
            all_y.extend(q1[valid_band].tolist())
            all_y.extend(q3[valid_band].tolist())
        if np.sum(valid_line) >= 2:
            ax.plot(
                betas[valid_line],
                mean[valid_line],
                color=str(style["color"]),
                lw=float(style["lw"]),
                ls=style["ls"],
                zorder=3,
            )
            all_y.extend(mean[valid_line].tolist())

    ax.set_xlim(float(betas[0]), float(betas[-1]))
    ax.set_xlabel(r"$\beta$")
    budget_label = f"{int(budget_q):,}".replace(",", "{,}")
    ax.set_ylabel(rf"$C_q(Q={budget_label})$")
    ax.grid(True, alpha=0.16, linestyle="--")
    major_ticks = _major_beta_ticks(betas.tolist())
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in major_ticks])
    ax.ticklabel_format(axis="y", style="plain")

    if all_y:
        ymax = max(all_y)
        ymin = min(all_y)
        lower = min(0.0, ymin - 0.05 * max(ymax - ymin, 1e-6))
        ax.set_ylim(lower, 1.04 * ymax)

    legend = ax.legend(
        handles=_legend_handles(),
        loc="upper right",
        frameon=True,
        fontsize=6.8,
        facecolor="white",
        edgecolor="#bfbfbf",
        borderpad=0.24,
        labelspacing=0.20,
        handlelength=1.65,
        handletextpad=0.42,
        borderaxespad=0.20,
    )
    legend.set_zorder(100)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def run() -> None:
    ap = argparse.ArgumentParser(description="Experiment 3: standalone beta-vs-quality-coverage summary plots.")
    ap.add_argument("--metrics-q1000", type=str, default=str(DEFAULT_METRICS[1000]))
    ap.add_argument("--metrics-q2000", type=str, default=str(DEFAULT_METRICS[2000]))
    ap.add_argument("--metrics-q5000", type=str, default=str(DEFAULT_METRICS[5000]))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--series-q1000", type=str, default=str(DEFAULT_METRICS[1000]))
    ap.add_argument("--series-q2000", type=str, default=str(DEFAULT_METRICS[2000]))
    ap.add_argument("--series-q5000", type=str, default=str(DEFAULT_METRICS[5000]))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    budget_inputs = {
        1000: {"metrics": Path(args.metrics_q1000), "series": str(args.series_q1000).strip()},
        2000: {"metrics": Path(args.metrics_q2000), "series": str(args.series_q2000).strip()},
        5000: {"metrics": Path(args.metrics_q5000), "series": str(args.series_q5000).strip()},
    }

    run_rows = []
    rerender_payload = {
        "script": SCRIPT_REL,
        "outdir": _try_rel(outdir),
        "plots": [],
    }

    for budget_q, spec in budget_inputs.items():
        series_arg = str(spec["series"])
        if series_arg:
            input_mode = "series_csv"
            source_path = Path(series_arg)
            if not source_path.exists():
                raise FileNotFoundError(f"Missing series csv: {source_path}")
            series_rows = _load_series_csv(source_path)
        else:
            input_mode = "metrics_csv"
            source_path = Path(spec["metrics"])
            if not source_path.exists():
                raise FileNotFoundError(f"Missing metrics csv: {source_path}")
            series_rows = _load_series_from_metrics_csv(source_path)

        stem = f"{STEM_PREFIX}_q{int(budget_q)}"
        out_pdf = outdir / f"{stem}.pdf"
        out_series_csv = outdir / f"{stem}_series.csv"
        render_plot(series_rows, out_pdf=out_pdf, budget_q=int(budget_q))
        _write_csv(out_series_csv, list(series_rows))

        run_rows.append(
            {
                "budget_q": int(budget_q),
                "input_mode": input_mode,
                "source_path": _try_rel(source_path),
                "pdf": _try_rel(out_pdf),
                "series_csv": _try_rel(out_series_csv),
            }
        )
        rerender_payload["plots"].append(
            {
                "budget_q": int(budget_q),
                "series_csv": _try_rel(out_series_csv),
                "rerender_command": (
                    f"MPLCONFIGDIR=/tmp/mpl-cache python {SCRIPT_REL} "
                    f"--series-q{int(budget_q)} {_try_rel(out_series_csv)} --outdir {_try_rel(outdir)}"
                ),
            }
        )

        print(f"[saved] {out_pdf}")
        print(f"[saved] {out_series_csv}")

    _write_csv(outdir / f"{STEM_PREFIX}_manifest.csv", run_rows)
    _write_json(
        outdir / "RUN_CONFIG.json",
        {
            "script": SCRIPT_REL,
            "outdir": _try_rel(outdir),
            "figure_width_in": FIG_W,
            "figure_height_in": FIG_H,
            "model_order": MODEL_ORDER,
            "band": "iqr",
            "pdf_only": True,
            "budgets": [1000, 2000, 5000],
            "sources": {str(q): row["source_path"] for q, row in zip([1000, 2000, 5000], run_rows)},
        },
    )
    _write_json(outdir / "RERENDER_CONFIG.json", rerender_payload)


if __name__ == "__main__":
    run()
