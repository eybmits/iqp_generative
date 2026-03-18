#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a compact beta-vs-Q80 companion plot from the stored Fig6 multiseed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from paper_benchmark_ledger import is_benchmark_20seed_run, record_benchmark_run

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_REL = "experiments/analysis/plot_fig6_beta_q80_summary.py"
OUTDIR_REL_DEFAULT = "outputs/analysis/fig6_beta_q80_summary"
STEM = "fig6_beta_q80_summary"
DEFAULT_Q80_THR = 0.8

FIG_W = 243.12 / 72.0
FIG_H = 185.52 / 72.0
PNG_DPI = 300
LEGEND_FONTSIZE = 7.2
SEED_TRACE_ALPHA = 0.18
SEED_TRACE_LW = 0.95
BAND_ALPHA = 0.14
MIN_LABELED_LOG_TICK = 1_000.0
MAX_LABELED_X_TICKS = 10
X_LABEL_EVERY_N_BETAS = 3

TARGET_COLOR = "#1C1C1C"
UNIFORM_KEY = "uniform_random"
UNIFORM_LABEL = "Uniform"
UNIFORM_COLOR = "#6E6E6E"
UNIFORM_LS = "--"
UNIFORM_LW = 1.5
ANNOTATION_BG = "#FFFFFF"


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


def make_figure(fig_w: float = FIG_W, fig_h: float = FIG_H) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        return str(path)


def _format_metric(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _q_label(q80_thr: float) -> str:
    return f"Q{int(round(100.0 * float(q80_thr)))}"


def _q_math_label(q80_thr: float) -> str:
    return rf"Q_{{{int(round(100.0 * float(q80_thr)))}}}"


def _output_stem(q80_thr: float) -> str:
    return f"fig6_beta_{_q_label(q80_thr).lower()}_summary"


def _tick_label(value: float) -> str:
    thresholds = [
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "k"),
    ]
    for scale, suffix in thresholds:
        if value >= scale:
            q = value / scale
            if abs(q - round(q)) < 1e-9:
                return f"{int(round(q))}{suffix}"
            return f"{q:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{int(round(value)):,}"


def _format_log_tick(value: float, _pos: object) -> str:
    if value <= 0.0:
        return ""
    return _tick_label(float(value))


def _log_ticks(y_min: float, y_max: float) -> List[float]:
    lo = max(1.0, float(y_min))
    hi = max(lo, float(y_max))
    ticks: List[float] = []
    lo_exp = int(math.floor(math.log10(lo)))
    hi_exp = int(math.floor(math.log10(hi)))

    lead_scale = 10.0**lo_exp
    for mult in (2.0, 5.0):
        tick = mult * lead_scale
        if lo <= tick <= hi and tick >= MIN_LABELED_LOG_TICK:
            ticks.append(float(tick))
            break

    for exp in range(lo_exp, hi_exp + 1):
        tick = 10.0**exp
        if lo <= tick <= hi and tick >= MIN_LABELED_LOG_TICK:
            ticks.append(float(tick))
    ticks = sorted(set(ticks))
    return ticks


def _next_log_cap(value: float) -> float:
    if value <= 1.0:
        return 10.0
    exp = int(math.floor(math.log10(value)))
    for mult in (1.0, 2.0, 5.0, 10.0):
        tick = mult * (10.0**exp)
        if tick >= value:
            return float(tick)
    return float(10.0 ** (exp + 1))


def _major_beta_ticks(betas: Sequence[float]) -> List[float]:
    beta_vals = [float(beta) for beta in betas]
    if len(beta_vals) <= MAX_LABELED_X_TICKS:
        return beta_vals

    step = max(2, int(X_LABEL_EVERY_N_BETAS))
    # Keep dense beta sweeps on a strict modulo grid; the final off-grid beta stays as a minor tick.
    major = [beta_vals[idx] for idx in range(0, len(beta_vals) - 1, step)]
    if not major:
        major = [beta_vals[0]]
    return major


def _first_q_crossing(q: np.ndarray, y: np.ndarray, thr: float) -> float:
    idx = np.where(y >= thr)[0]
    if idx.size == 0:
        return float("inf")
    i = int(idx[0])
    if i == 0:
        return float(q[0])
    q0, q1 = float(q[i - 1]), float(q[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 <= y0 + 1e-12:
        return q1
    t = (float(thr) - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return q0 + t * (q1 - q0)


def _write_readme(
    path: Path,
    *,
    metrics_csv_rel: str,
    data_npz_rel: str,
    style_npz_rel: str,
    q80_thr: float,
    band_stat: str,
    show_seed_traces: bool,
    show_band: bool,
    min_finite_seeds: int,
    y_cap_requested: float,
    y_cap_effective: float,
    has_censored: bool,
) -> None:
    q_label = _q_label(q80_thr)
    stem = _output_stem(q80_thr)
    lines = [
        f"# Fig6 Beta {q_label} Summary",
        "",
        "This directory contains a compact companion summary for the documented Fig6 multiseed rerun.",
        "",
        "Inputs:",
        "",
        f"- per-seed metrics: `{metrics_csv_rel}`",
        f"- multiseed recovery curves: `{data_npz_rel}`",
        f"- style snapshot: `{style_npz_rel}`",
        "- no retraining; all values are derived from the stored multiseed Fig6 artifacts",
        "",
        "Primary summary metric:",
        "",
        f"- `{q_label}`: first interpolated `Q` where `R(Q) >= {q80_thr:.1f}`",
        "- x-axis: target sharpness parameter `beta`",
        f"- y-axis: absolute `{q_label}` on a log scale; lower is better",
        "",
        "Visual encoding:",
        "",
        "- one line per model using the same labels, colors, and line styles as Fig6",
        (
            "- thin background lines show individual seed-specific `Q80(beta)` traces"
            if show_seed_traces
            else "- individual seed traces are hidden in the rendered plot"
        ),
        (
            "- the thick foreground line shows the seed-median and the shaded band shows the IQR"
            if show_band and str(band_stat) == "iqr"
            else (
                "- the thick foreground line shows the seed-mean and the shaded band shows mean ± std"
                if show_band and str(band_stat) == "mean_std"
                else "- the thick foreground line shows the seed-median without a spread band"
            )
        ),
        "- the companion metrics CSV contains `q25/q50/q75/mean/std` for the same runs",
        "- `Uniform` is shown as a deterministic reference line without a seed band",
        "- `Target` is intentionally omitted to keep the plot compact",
        "",
        "Kept files:",
        "",
        f"- `{stem}.pdf`",
        f"- `{stem}.png`",
        f"- `{stem}_metrics.csv`",
        "- `README.md`",
        "- `RUN_CONFIG.json`",
    ]
    if has_censored:
        lines.insert(
            17,
            f"- hollow triangles mark `beta` values where fewer than `{min_finite_seeds}` seeds reach `{q_label}`",
        )
        lines.insert(
            18,
            (
                "- if present, censored points are pinned to the effective plot cap "
                f"`y = {int(round(y_cap_effective))}` with a requested floor of `{int(round(y_cap_requested))}`"
            ),
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_style(style_npz: Path) -> Dict[str, object]:
    with np.load(style_npz, allow_pickle=True) as z:
        return {
            "betas": np.asarray(z["betas"], dtype=np.float64),
            "model_order": [str(x) for x in z["model_order"].tolist()],
            "model_labels": [str(x) for x in z["model_labels"].tolist()],
            "style_color": [str(x) for x in z["style_color"].tolist()],
            "style_ls": z["style_ls"].tolist(),
            "style_lw": [float(x) for x in z["style_lw"].tolist()],
        }


def _load_curve_payload(data_npz: Path) -> Dict[str, object]:
    with np.load(data_npz, allow_pickle=True) as z:
        return {
            "Q": np.asarray(z["Q"], dtype=np.int64),
            "betas": np.asarray(z["betas"], dtype=np.float64),
            "seed_values": np.asarray(z["seed_values"], dtype=np.int64),
            "model_order": [str(x) for x in z["model_order"].tolist()],
            "curves": np.asarray(z["curves"], dtype=np.float64),
            "y_unif": np.asarray(z["y_unif"], dtype=np.float64),
            "holdout_seed": int(np.asarray(z["holdout_seed"]).item()),
            "holdout_mode": str(np.asarray(z["holdout_mode"]).item()),
            "holdout_m_train": int(np.asarray(z["holdout_m_train"]).item()),
        }


def _load_metric_rows(metrics_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"beta", "model_key", "seed", "Q80"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = ", ".join(sorted(required.difference(set(reader.fieldnames or []))))
            raise ValueError(f"Metrics CSV missing required columns: {missing}")
        for row in reader:
            rows.append(
                {
                    "beta": float(row["beta"]),
                    "model_key": str(row["model_key"]),
                    "seed": int(row["seed"]),
                    "Q80": float(row["Q80"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in metrics CSV: {metrics_csv}")
    return rows


def _metric_rows_from_curves(
    *,
    q: np.ndarray,
    betas: Sequence[float],
    model_order: Sequence[str],
    seed_values: Sequence[int],
    curves: np.ndarray,
    q80_thr: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    qf = np.asarray(q, dtype=np.float64)
    for beta_idx, beta in enumerate(betas):
        for model_idx, model_key in enumerate(model_order):
            for seed_idx, seed in enumerate(seed_values):
                q80 = _first_q_crossing(qf, curves[beta_idx, model_idx, seed_idx].astype(np.float64), float(q80_thr))
                rows.append(
                    {
                        "beta": float(beta),
                        "model_key": str(model_key),
                        "seed": int(seed),
                        "Q80": float(q80),
                    }
                )
    if not rows:
        raise ValueError("No per-seed rows could be derived from the multiseed curve payload.")
    return rows


def _aggregate_rows(
    metric_rows: Sequence[Dict[str, object]],
    *,
    betas: Sequence[float],
    model_order: Sequence[str],
    model_labels: Sequence[str],
    min_finite_seeds: int,
) -> List[Dict[str, object]]:
    grouped: Dict[tuple[float, str], List[Dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(float(row["beta"]), str(row["model_key"]))].append(dict(row))

    label_by_key = dict(zip(model_order, model_labels))
    summary_rows: List[Dict[str, object]] = []
    for beta in betas:
        for model_key in model_order:
            rows = grouped.get((float(beta), str(model_key)), [])
            if not rows:
                continue
            q80_vals = np.asarray([float(r["Q80"]) for r in rows], dtype=np.float64)
            finite = q80_vals[np.isfinite(q80_vals)]
            seeds_total = len(rows)
            seeds_reached = int(finite.size)
            if finite.size > 0:
                q25, q50, q75 = np.quantile(finite, [0.25, 0.50, 0.75])
            else:
                q25 = q50 = q75 = float("inf")
            summary_rows.append(
                {
                    "beta": float(beta),
                    "model_key": str(model_key),
                    "model_label": str(label_by_key.get(model_key, model_key)),
                    "series_type": "model",
                    "seeds_total": int(seeds_total),
                    "seeds_reached": seeds_reached,
                    "resolved": int(seeds_reached >= int(min_finite_seeds)),
                    "q80_q25": _format_metric(float(q25)),
                    "q80_median": _format_metric(float(q50)),
                    "q80_q75": _format_metric(float(q75)),
                    "q80_mean": _format_metric(float(np.mean(finite))) if finite.size > 0 else "inf",
                    "q80_std": _format_metric(float(np.std(finite, ddof=0))) if finite.size > 0 else "inf",
                }
            )
    if not summary_rows:
        raise ValueError("No summary rows could be derived from the source per-seed data.")
    return summary_rows


def _uniform_reference_rows(
    *,
    q: np.ndarray,
    betas: Sequence[float],
    y_unif: np.ndarray,
    q80_thr: float,
) -> List[Dict[str, object]]:
    qf = np.asarray(q, dtype=np.float64)
    rows: List[Dict[str, object]] = []
    for beta_idx, beta in enumerate(betas):
        q80 = _first_q_crossing(qf, y_unif[beta_idx].astype(np.float64), float(q80_thr))
        q80_fmt = _format_metric(float(q80))
        rows.append(
            {
                "beta": float(beta),
                "model_key": UNIFORM_KEY,
                "model_label": UNIFORM_LABEL,
                "series_type": "reference",
                "seeds_total": 1,
                "seeds_reached": int(math.isfinite(q80)),
                "resolved": int(math.isfinite(q80)),
                "q80_q25": q80_fmt,
                "q80_median": q80_fmt,
                "q80_q75": q80_fmt,
                "q80_mean": q80_fmt,
                "q80_std": _format_metric(0.0),
            }
        )
    return rows


def _summary_arrays(rows: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    xs = np.asarray([float(r["beta"]) for r in rows], dtype=np.float64)
    q25 = np.asarray([float(r["q80_q25"]) for r in rows], dtype=np.float64)
    q50 = np.asarray([float(r["q80_median"]) for r in rows], dtype=np.float64)
    q75 = np.asarray([float(r["q80_q75"]) for r in rows], dtype=np.float64)
    mean = np.asarray([float(r["q80_mean"]) for r in rows], dtype=np.float64)
    std = np.asarray([float(r["q80_std"]) for r in rows], dtype=np.float64)
    reached = np.asarray([int(r["seeds_reached"]) for r in rows], dtype=np.int64)
    total = np.asarray([int(r["seeds_total"]) for r in rows], dtype=np.int64)
    resolved = np.asarray([bool(int(r["resolved"])) for r in rows], dtype=bool)
    return {
        "x": xs,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "mean": mean,
        "std": std,
        "reached": reached,
        "total": total,
        "resolved": resolved,
    }


def _summary_center_band(arrays: Dict[str, np.ndarray], band_stat: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if str(band_stat) == "mean_std":
        center = np.asarray(arrays["mean"], dtype=np.float64)
        spread = np.asarray(arrays["std"], dtype=np.float64)
        lower = np.where(np.isfinite(center) & np.isfinite(spread) & ((center - spread) > 0.0), center - spread, np.nan)
        upper = np.where(np.isfinite(center) & np.isfinite(spread), center + spread, np.nan)
        return center, lower, upper
    center = np.asarray(arrays["q50"], dtype=np.float64)
    lower = np.asarray(arrays["q25"], dtype=np.float64)
    upper = np.asarray(arrays["q75"], dtype=np.float64)
    return center, lower, upper


def _seed_trace_arrays(
    metric_rows: Sequence[Dict[str, object]],
    *,
    betas: Sequence[float],
    model_order: Sequence[str],
) -> Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]]:
    beta_values = [float(beta) for beta in betas]
    grouped: Dict[Tuple[str, int], Dict[float, float]] = defaultdict(dict)
    for row in metric_rows:
        model_key = str(row["model_key"])
        seed = int(row["seed"])
        grouped[(model_key, seed)][float(row["beta"])] = float(row["Q80"])

    out: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    x = np.asarray(beta_values, dtype=np.float64)
    for model_key in model_order:
        seed_values = sorted({seed for key, seed in grouped if key == str(model_key)})
        for seed in seed_values:
            beta_to_q80 = grouped[(str(model_key), int(seed))]
            y = np.asarray([float(beta_to_q80.get(beta, float("nan"))) for beta in beta_values], dtype=np.float64)
            out[str(model_key)].append((int(seed), x, y))
    return out


def _contiguous_runs(mask: np.ndarray) -> Iterable[slice]:
    start = None
    for idx, is_on in enumerate(mask.tolist()):
        if is_on and start is None:
            start = idx
        elif not is_on and start is not None:
            yield slice(start, idx)
            start = None
    if start is not None:
        yield slice(start, mask.size)


def _legend_handles(
    model_labels: Sequence[str],
    colors: Sequence[str],
    linestyles: Sequence[object],
    linewidths: Sequence[float],
    include_censor: bool,
) -> List[Line2D]:
    handles: List[Line2D] = []
    for label, color, linestyle, linewidth in zip(model_labels, colors, linestyles, linewidths):
        handles.append(
            Line2D(
                [0],
                [0],
                color=str(color),
                linestyle=linestyle,
                linewidth=max(1.1, float(linewidth)),
                label=str(label),
            )
        )
    if include_censor:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="^",
                markersize=5.5,
                markerfacecolor="white",
                markeredgecolor=TARGET_COLOR,
                markeredgewidth=0.9,
                color=TARGET_COLOR,
                label="censored / > budget",
            )
        )
    return handles


def _write_run_config(
    path: Path,
    *,
    outdir: Path,
    metrics_csv: Path,
    data_npz: Path,
    style_npz: Path,
    betas: Sequence[float],
    seed_values: Sequence[int],
    model_order: Sequence[str],
    q80_thr: float,
    band_stat: str,
    show_seed_traces: bool,
    show_band: bool,
    min_finite_seeds: int,
    y_cap_requested: float,
    y_cap_effective: float,
    holdout_seed: int,
    holdout_mode: str,
    holdout_m_train: int,
) -> None:
    stem = _output_stem(q80_thr)
    outdir_rel = _try_rel(outdir)
    metrics_rel = _try_rel(metrics_csv)
    data_rel = _try_rel(data_npz)
    style_rel = _try_rel(style_npz)
    output_files = [
        f"{outdir_rel}/{stem}.pdf",
        f"{outdir_rel}/{stem}.png",
        f"{outdir_rel}/{stem}_metrics.csv",
        f"{outdir_rel}/RUN_CONFIG.json",
        f"{outdir_rel}/README.md",
    ]
    payload = {
        "selected_analysis_run": outdir.resolve() == (ROOT / OUTDIR_REL_DEFAULT).resolve(),
        "script": SCRIPT_REL,
        "outdir": outdir_rel,
        "metrics_csv": metrics_rel,
        "data_npz": data_rel,
        "style_npz": style_rel,
        "betas": [float(x) for x in betas],
        "seed_values": [int(x) for x in seed_values],
        "model_order": [str(x) for x in model_order],
        "reference_series": [UNIFORM_KEY],
        "q80_thr": float(q80_thr),
        "aggregation": "median_over_finite_q80",
        "band_stat": str(band_stat),
        "show_seed_traces": bool(show_seed_traces),
        "show_band": bool(show_band),
        "min_finite_seeds": int(min_finite_seeds),
        "y_scale": "log",
        "visual_style": "fig4_mechanistic_single_panel_style_with_seed_traces",
        "y_cap_requested": float(y_cap_requested),
        "y_cap_effective": float(y_cap_effective),
        "holdout_seed": int(holdout_seed),
        "holdout_mode": str(holdout_mode),
        "holdout_m_train": int(holdout_m_train),
        "selected_output": output_files[0],
        "output_files": output_files,
        "rerun_command": (
            "MPLCONFIGDIR=/tmp/mpl-cache python "
            f"{SCRIPT_REL} "
            f"--metrics-csv {metrics_rel} "
            f"--data-npz {data_rel} "
            f"--style-npz {style_rel} "
            f"--outdir {outdir_rel}"
        ),
    }
    _write_json(path, payload)


def run() -> None:
    ap = argparse.ArgumentParser(description="Render a compact Fig6 beta-vs-Q80 summary plot.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / OUTDIR_REL_DEFAULT),
    )
    ap.add_argument(
        "--metrics-csv",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "analysis"
            / "fig6_multiseed_all600_seeds101_120"
            / "fig6_beta_sweep_recovery_grid_multiseed_metrics.csv"
        ),
    )
    ap.add_argument(
        "--data-npz",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "analysis"
            / "fig6_multiseed_all600_seeds101_120"
            / "fig6_beta_sweep_recovery_grid_multiseed_data.npz"
        ),
    )
    ap.add_argument(
        "--style-npz",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "final_plots"
            / "fig6_beta_sweep_recovery_grid"
            / "fig6_data_default.npz"
        ),
    )
    ap.add_argument("--q80-thr", type=float, default=DEFAULT_Q80_THR)
    ap.add_argument("--band-stat", type=str, default="iqr", choices=["iqr", "mean_std", "none"])
    ap.add_argument("--show-seed-traces", type=int, default=1, choices=[0, 1])
    ap.add_argument("--show-band", type=int, default=1, choices=[0, 1])
    ap.add_argument("--min-finite-seeds", type=int, default=3)
    ap.add_argument("--y-cap", type=float, default=100000.0)
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    if int(args.min_finite_seeds) < 1:
        raise ValueError("--min-finite-seeds must be >= 1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.metrics_csv)
    data_npz = Path(args.data_npz)
    style_npz = Path(args.style_npz)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")
    if not data_npz.exists():
        raise FileNotFoundError(f"Missing multiseed data NPZ: {data_npz}")
    if not style_npz.exists():
        raise FileNotFoundError(f"Missing style NPZ: {style_npz}")

    apply_final_style()

    style_payload = _load_style(style_npz)
    curve_payload = _load_curve_payload(data_npz)
    if [str(x) for x in curve_payload["model_order"]] != [str(x) for x in style_payload["model_order"]]:
        raise ValueError("Multiseed data NPZ model order does not match the style NPZ.")

    if math.isclose(float(args.q80_thr), DEFAULT_Q80_THR, rel_tol=0.0, abs_tol=1e-12):
        metric_rows = _load_metric_rows(metrics_csv)
    else:
        metric_rows = _metric_rows_from_curves(
            q=np.asarray(curve_payload["Q"], dtype=np.int64),
            betas=np.asarray(curve_payload["betas"], dtype=np.float64).tolist(),
            model_order=[str(x) for x in curve_payload["model_order"]],
            seed_values=np.asarray(curve_payload["seed_values"], dtype=np.int64).tolist(),
            curves=np.asarray(curve_payload["curves"], dtype=np.float64),
            q80_thr=float(args.q80_thr),
        )

    betas_present = sorted({float(row["beta"]) for row in metric_rows})
    curve_betas = [float(x) for x in np.asarray(curve_payload["betas"], dtype=np.float64).tolist()]
    betas = [beta for beta in curve_betas if beta in betas_present]
    seen_betas = set(betas)
    for beta in betas_present:
        if beta not in seen_betas:
            betas.append(beta)
            seen_betas.add(beta)
    if not betas:
        raise ValueError("No overlapping beta values were found between the multiseed data NPZ and the source per-seed data.")

    model_order_all = [str(x) for x in style_payload["model_order"]]
    model_labels_all = [str(x) for x in style_payload["model_labels"]]
    style_color_all = [str(x) for x in style_payload["style_color"]]
    style_ls_all = list(style_payload["style_ls"])
    style_lw_all = [float(x) for x in style_payload["style_lw"]]
    model_keys_present = {str(row["model_key"]) for row in metric_rows}
    keep_indices = [idx for idx, key in enumerate(model_order_all) if key in model_keys_present]
    model_order = [model_order_all[idx] for idx in keep_indices]
    model_labels = [model_labels_all[idx] for idx in keep_indices]
    style_color = [style_color_all[idx] for idx in keep_indices]
    style_ls = [style_ls_all[idx] for idx in keep_indices]
    style_lw = [style_lw_all[idx] for idx in keep_indices]
    series_order = model_order + [UNIFORM_KEY]
    series_labels = model_labels + [UNIFORM_LABEL]
    series_color = style_color + [UNIFORM_COLOR]
    series_ls = style_ls + [UNIFORM_LS]
    series_lw = style_lw + [UNIFORM_LW]
    q_label = _q_label(float(args.q80_thr))
    stem = _output_stem(float(args.q80_thr))

    summary_rows = _aggregate_rows(
        metric_rows,
        betas=betas,
        model_order=model_order,
        model_labels=model_labels,
        min_finite_seeds=int(args.min_finite_seeds),
    )
    summary_rows.extend(
        _uniform_reference_rows(
            q=np.asarray(curve_payload["Q"], dtype=np.int64),
            betas=betas,
            y_unif=np.asarray(curve_payload["y_unif"], dtype=np.float64),
            q80_thr=float(args.q80_thr),
        )
    )
    summary_csv = outdir / f"{stem}_metrics.csv"
    _write_csv(summary_csv, summary_rows)

    grouped_summary: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped_summary[str(row["model_key"])].append(dict(row))
    for key in grouped_summary:
        grouped_summary[key].sort(key=lambda row: float(row["beta"]))
    seed_traces_by_model = _seed_trace_arrays(metric_rows, betas=betas, model_order=model_order)

    finite_plot_vals: List[float] = []
    for model_key in series_order:
        rows = grouped_summary.get(model_key, [])
        if not rows:
            continue
        arrays = _summary_arrays(rows)
        center, band_lo, band_hi = _summary_center_band(arrays, str(args.band_stat))
        finite_plot_vals.extend([float(v) for v in center[np.isfinite(center)].tolist()])
        if bool(int(args.show_band)) and str(model_key) != UNIFORM_KEY and str(args.band_stat) != "none":
            finite_plot_vals.extend([float(v) for v in band_lo[np.isfinite(band_lo)].tolist()])
            finite_plot_vals.extend([float(v) for v in band_hi[np.isfinite(band_hi)].tolist()])
    if bool(int(args.show_seed_traces)):
        for model_key in model_order:
            for _seed, _x, y_vals in seed_traces_by_model.get(str(model_key), []):
                finite_plot_vals.extend([float(v) for v in y_vals[np.isfinite(y_vals)].tolist()])
    if not finite_plot_vals:
        raise ValueError("No finite summary values are available for plotting.")
    y_min = min(1500.0, 0.92 * min(finite_plot_vals))
    y_cap_eff = max(float(args.y_cap), _next_log_cap(1.12 * max(finite_plot_vals)))
    y_ticks = _log_ticks(y_min, y_cap_eff)

    fig_w = max(FIG_W, 1.0 + 0.23 * float(len(betas)))
    fig, ax = make_figure(fig_w=fig_w)
    ax.set_yscale("log")
    ax.set_xlim(min(betas) - 0.03, max(betas) + 0.03)
    ax.set_ylim(y_min, y_cap_eff)
    major_betas = _major_beta_ticks(betas)
    minor_betas = [float(beta) for beta in betas if all(not math.isclose(float(beta), float(major), rel_tol=0.0, abs_tol=1e-12) for major in major_betas)]
    ax.xaxis.set_major_locator(FixedLocator(major_betas))
    ax.set_xticklabels([f"{beta:.1f}" for beta in major_betas])
    if minor_betas:
        ax.xaxis.set_minor_locator(FixedLocator(minor_betas))
        ax.tick_params(axis="x", which="minor", length=3.0, width=0.9)
    else:
        ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"Sharpness $\beta$")
    ax.set_ylabel("Q80 (Samples)")

    has_censored = False

    for idx, model_key in enumerate(series_order):
        rows = grouped_summary.get(model_key, [])
        if not rows:
            continue
        arrays = _summary_arrays(rows)
        xs = arrays["x"]
        center, band_lo, band_hi = _summary_center_band(arrays, str(args.band_stat))
        resolved = arrays["resolved"]
        reached = arrays["reached"]
        total = arrays["total"]
        color = series_color[idx]
        linestyle = series_ls[idx]
        linewidth = max(1.1, float(series_lw[idx]))
        is_reference = str(model_key) == UNIFORM_KEY

        if bool(int(args.show_seed_traces)) and not is_reference:
            for _seed, x_seed, y_seed in seed_traces_by_model.get(str(model_key), []):
                finite_seed = np.isfinite(y_seed)
                for run_slice in _contiguous_runs(finite_seed):
                    ax.plot(
                        x_seed[run_slice],
                        y_seed[run_slice],
                        color=color,
                        linestyle=linestyle,
                        linewidth=SEED_TRACE_LW,
                        alpha=SEED_TRACE_ALPHA,
                        zorder=3 + idx,
                    )

        if bool(int(args.show_band)) and not is_reference and str(args.band_stat) != "none":
            band_mask = resolved & np.isfinite(band_lo) & np.isfinite(band_hi)
            for run_slice in _contiguous_runs(band_mask):
                ax.fill_between(
                    xs[run_slice],
                    band_lo[run_slice],
                    band_hi[run_slice],
                    color=color,
                    alpha=BAND_ALPHA,
                    linewidth=0.0,
                    zorder=6 + idx,
                )

        # Break line segments at censored beta values instead of interpolating through them.
        for run_slice in _contiguous_runs(resolved):
            ax.plot(
                xs[run_slice],
                center[run_slice],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=12 + idx,
            )

        censored = ~resolved
        if np.any(censored):
            has_censored = True
            ax.scatter(
                xs[censored],
                np.full(int(np.sum(censored)), y_cap_eff),
                marker="^",
                s=44,
                facecolors="white",
                edgecolors=color,
                linewidths=1.0,
                clip_on=False,
                zorder=18 + idx,
            )
            for x_val, reached_val, total_val in zip(xs[censored], reached[censored], total[censored]):
                ax.text(
                    float(x_val),
                    y_cap_eff / 1.09,
                    f"{int(reached_val)}/{int(total_val)}",
                    ha="center",
                    va="top",
                    fontsize=6.7,
                    color=color,
                    bbox=dict(facecolor=ANNOTATION_BG, edgecolor="none", alpha=0.72, pad=0.18),
                    zorder=19 + idx,
                )

    legend = ax.legend(
        handles=_legend_handles(series_labels, series_color, series_ls, series_lw, include_censor=has_censored),
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        facecolor="white",
        edgecolor="#bfbfbf",
    )
    legend.set_zorder(40)

    out_pdf = outdir / f"{stem}.pdf"
    out_png = outdir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    _write_readme(
        outdir / "README.md",
        metrics_csv_rel=_try_rel(metrics_csv),
        data_npz_rel=_try_rel(data_npz),
        style_npz_rel=_try_rel(style_npz),
        q80_thr=float(args.q80_thr),
        band_stat=str(args.band_stat),
        show_seed_traces=bool(int(args.show_seed_traces)),
        show_band=bool(int(args.show_band)),
        min_finite_seeds=int(args.min_finite_seeds),
        y_cap_requested=float(args.y_cap),
        y_cap_effective=float(y_cap_eff),
        has_censored=has_censored,
    )
    _write_run_config(
        outdir / "RUN_CONFIG.json",
        outdir=outdir,
        metrics_csv=metrics_csv,
        data_npz=data_npz,
        style_npz=style_npz,
        betas=betas,
        seed_values=np.asarray(curve_payload["seed_values"], dtype=np.int64).tolist(),
        model_order=model_order,
        q80_thr=float(args.q80_thr),
        band_stat=str(args.band_stat),
        show_seed_traces=bool(int(args.show_seed_traces)),
        show_band=bool(int(args.show_band)),
        min_finite_seeds=int(args.min_finite_seeds),
        y_cap_requested=float(args.y_cap),
        y_cap_effective=float(y_cap_eff),
        holdout_seed=int(curve_payload["holdout_seed"]),
        holdout_mode=str(curve_payload["holdout_mode"]),
        holdout_m_train=int(curve_payload["holdout_m_train"]),
    )

    seed_values = np.asarray(curve_payload["seed_values"], dtype=np.int64).tolist()
    if is_benchmark_20seed_run(seed_values):
        experiment_id = "fig6_base_q80_summary_20seed"
        title = "Fig6 base Q80 summary"
        if "beta0p1_2p0" in str(outdir):
            band_mode = str(args.band_stat)
            if "seed_traces" in str(outdir):
                experiment_id = "fig6_wide_q80_summary_iqr_seed_traces_20seed"
                title = "Fig6 wide Q80 summary with seed traces"
            elif band_mode == "mean_std":
                experiment_id = "fig6_wide_q80_summary_mean_std_20seed"
                title = "Fig6 wide Q80 summary (mean +/- std)"
            else:
                experiment_id = "fig6_wide_q80_summary_iqr_20seed"
                title = "Fig6 wide Q80 summary (median + IQR)"
        record_benchmark_run(
            experiment_id=experiment_id,
            title=title,
            run_config_path=outdir / "RUN_CONFIG.json",
            output_paths=[out_pdf, out_png, outdir / "README.md"],
            metrics_paths=[summary_csv, metrics_csv, data_npz],
            notes=[
                "Derived 20-seed benchmark-standard Q80 summary.",
                "No retraining is performed in this step; the summary is built from stored multiseed artifacts.",
            ],
        )

    print(f"[saved] {summary_csv}")
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")
    print(f"[saved] {outdir / 'README.md'}")
    print(f"[saved] {outdir / 'RUN_CONFIG.json'}")


if __name__ == "__main__":
    run()
