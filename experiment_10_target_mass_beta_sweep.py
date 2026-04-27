#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example wide target-mass sweep plot for visual comparison with the KL panels.

This plot is derived directly from the target distribution p* across the beta
sweep and mirrors the wider aspect ratio used for the approved Experiment 2/3
pointcloud figures, while keeping a simple line-plot style.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiment_1_kl_diagnostics import build_target_distribution_paper
from final_plot_style import (
    IEEE_COLUMN_W_IN,
    IEEE_COLUMN_W_PT,
    TEXT_DARK,
    apply_ieee_latex_style,
    save_exact_figure,
)


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_10_target_mass_beta_sweep.py"
OUTPUT_STEM = "experiment_10_target_mass_beta_sweep"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM

FIG_W = 320.0 / 72.0
FIG_H = 185.52 / 72.0
PUBLICATION_FIG_W_PT = IEEE_COLUMN_W_PT
PUBLICATION_FIG_H_PT = PUBLICATION_FIG_W_PT * (185.52 / 320.0)
PUBLICATION_FIGSIZE = (PUBLICATION_FIG_W_PT / 72.0, PUBLICATION_FIG_H_PT / 72.0)
WIDER_REFERENCE_FIG_W_PT = 280.0
WIDER_REFERENCE_FIG_H_PT = 168.0
WIDER_REFERENCE_FIGSIZE = (WIDER_REFERENCE_FIG_W_PT / 72.0, WIDER_REFERENCE_FIG_H_PT / 72.0)
PLOT_LEFT = 0.17
PLOT_RIGHT = 0.985
PLOT_BOTTOM = 0.22
PLOT_TOP = 0.93

ALL_BETAS = np.asarray([x / 10.0 for x in range(1, 21)], dtype=np.float64)
HIGHLIGHT_BETAS = np.asarray([0.6, 0.8, 1.0, 1.2, 1.4], dtype=np.float64)

GRAY_SWEEP_COLOR = "#CFCFCF"
HIGHLIGHT_COLORS = ["#151515", "#5D1C18", "#8B2A22", "#BF3A2E", "#E34234"]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def _bucket_masses(dist: np.ndarray, scores: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score_vals = np.asarray(sorted(int(s) for s in np.unique(scores[support])), dtype=np.int64)
    masses = np.zeros(score_vals.size, dtype=np.float64)
    for idx, s in enumerate(score_vals.tolist()):
        masses[idx] = float(np.sum(np.asarray(dist, dtype=np.float64)[(scores == float(s)) & support]))
    return score_vals, masses


def _parse_float_list(text: str) -> np.ndarray:
    return np.asarray([float(x.strip()) for x in str(text).split(",") if x.strip()], dtype=np.float64)


def _load_cached_mass_rows(path: Path, *, n: int, betas: np.ndarray) -> tuple[np.ndarray, Dict[float, np.ndarray], List[Dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(f"Cached target-mass CSV not found: {path}")
    rows_out: List[Dict[str, object]] = []
    grouped: Dict[float, List[tuple[int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_n = int(row["n"])
            row_beta = float(row["beta"])
            if row_n != int(n):
                continue
            grouped.setdefault(row_beta, []).append((int(row["score_level_display"]), float(row["target_mass"])))
            rows_out.append(
                {
                    "n": row_n,
                    "beta": row_beta,
                    "score_level_raw": int(row["score_level_raw"]),
                    "score_level_display": int(row["score_level_display"]),
                    "target_mass": float(row["target_mass"]),
                }
            )
    expected = [float(x) for x in betas.tolist()]
    missing = [beta for beta in expected if beta not in grouped]
    if missing:
        missing_txt = ", ".join(f"{beta:g}" for beta in missing)
        raise RuntimeError(f"Cached CSV is missing beta rows for n={n}: {missing_txt}")
    score_display = np.asarray(sorted(idx for idx, _mass in grouped[expected[0]]), dtype=np.int64)
    masses_by_beta: Dict[float, np.ndarray] = {}
    for beta in expected:
        ordered = sorted(grouped[beta], key=lambda item: item[0])
        masses_by_beta[beta] = np.asarray([mass for _idx, mass in ordered], dtype=np.float64)
    return score_display, masses_by_beta, rows_out


def _render_plot(
    *,
    out_pdf: Path,
    out_png: Path,
    figsize: tuple[float, float],
    all_betas: np.ndarray,
    highlight_betas: np.ndarray,
    score_display: np.ndarray,
    masses_by_beta: Dict[float, np.ndarray],
    subplot_adjust: tuple[float, float, float, float] = (PLOT_LEFT, PLOT_RIGHT, PLOT_BOTTOM, PLOT_TOP),
    use_tex: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_ieee_latex_style(use_tex=bool(use_tex))
    fig, ax = plt.subplots(figsize=figsize)
    left, right, bottom, top = subplot_adjust
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for beta in all_betas.tolist():
        y = masses_by_beta[float(beta)]
        ax.plot(
            score_display,
            y,
            color=GRAY_SWEEP_COLOR,
            lw=0.9,
            alpha=0.33,
            zorder=1,
        )

    for beta, color in zip(highlight_betas.tolist(), HIGHLIGHT_COLORS):
        y = masses_by_beta[float(beta)]
        ax.plot(
            score_display,
            y,
            color=color,
            lw=2.1,
            marker="o",
            markersize=5.0,
            label=rf"$\beta = {beta:g}$",
            zorder=3,
        )

    ax.set_xlabel(r"Score level $s$", labelpad=2.0)
    ax.set_ylabel(r"Target mass $p^*(\ell = s)$", labelpad=4.0)
    ax.set_xlim(float(score_display.min()) - 0.15, float(score_display.max()) + 0.15)
    ax.set_xticks(score_display.tolist())
    ymax = max(float(np.max(vals)) for vals in masses_by_beta.values())
    ax.set_ylim(0.0, ymax * 1.12)
    ax.grid(True, ls="--", lw=0.5, alpha=0.25)

    handles = [
        Line2D([0], [0], color=GRAY_SWEEP_COLOR, lw=1.2, alpha=0.9, label=r"gray: $\beta \in [0.1, 2]$, $\Delta\beta = 0.1$")
    ]
    for beta, color in zip(highlight_betas.tolist(), HIGHLIGHT_COLORS):
        handles.append(
            Line2D([0], [0], color=color, lw=2.1, marker="o", markersize=5.0, label=rf"$\beta = {beta:g}$")
        )
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.6,
        handletextpad=0.45,
    )
    legend.set_zorder(100)

    ax.spines["bottom"].set_color(TEXT_DARK)
    ax.spines["bottom"].set_linewidth(0.95)
    ax.spines["bottom"].set_zorder(10)
    ax.tick_params(axis="both", which="major", pad=1.5)

    save_exact_figure(fig, out_pdf)
    save_exact_figure(fig, out_png, dpi=300)
    plt.close(fig)


def run() -> None:
    ap = argparse.ArgumentParser(description="Wide target-mass beta-sweep example plot.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--betas", type=str, default=",".join(f"{x:.1f}" for x in ALL_BETAS.tolist()))
    ap.add_argument("--highlight-betas", type=str, default=",".join(f"{x:.1f}" for x in HIGHLIGHT_BETAS.tolist()))
    ap.add_argument("--use-tex", type=int, default=1, help="Set to 0 to disable LaTeX text rendering.")
    ap.add_argument("--recompute", type=int, default=0, help="Set to 1 to rebuild the cached CSV instead of rerendering from it.")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    all_betas = _parse_float_list(str(args.betas))
    highlight_betas = _parse_float_list(str(args.highlight_betas))

    masses_by_beta: Dict[float, np.ndarray] = {}
    rows: List[Dict[str, object]] = []

    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}.png"
    out_publication_pdf = outdir / f"{OUTPUT_STEM}_ieee_singlecol.pdf"
    out_publication_png = outdir / f"{OUTPUT_STEM}_ieee_singlecol.png"
    out_wider_pdf = outdir / f"{OUTPUT_STEM}_wider_reference.pdf"
    out_wider_png = outdir / f"{OUTPUT_STEM}_wider_reference.png"
    data_csv = outdir / f"{OUTPUT_STEM}.csv"
    run_json = outdir / "RUN_CONFIG.json"

    if bool(int(args.recompute)):
        masses_by_beta = {}
        rows = []
        raw_score_vals = None
        for beta in all_betas.tolist():
            p_star, support, scores = build_target_distribution_paper(int(args.n), float(beta))
            score_vals, masses = _bucket_masses(p_star, scores, support)
            if raw_score_vals is None:
                raw_score_vals = np.asarray(score_vals, dtype=np.int64)
            masses_by_beta[float(beta)] = np.asarray(masses, dtype=np.float64)
            for idx, score_raw in enumerate(score_vals.tolist()):
                rows.append(
                    {
                        "n": int(args.n),
                        "beta": float(beta),
                        "score_level_raw": int(score_raw),
                        "score_level_display": int(score_raw - score_vals.min()),
                        "target_mass": float(masses[idx]),
                    }
                )
        if raw_score_vals is None:
            raise RuntimeError("No score levels were generated.")
        score_display = np.asarray(raw_score_vals - int(raw_score_vals.min()), dtype=np.int64)
        _write_csv(data_csv, rows)
    else:
        score_display, masses_by_beta, rows = _load_cached_mass_rows(data_csv, n=int(args.n), betas=all_betas)

    _render_plot(
        out_pdf=out_pdf,
        out_png=out_png,
        figsize=(FIG_W, FIG_H),
        all_betas=all_betas,
        highlight_betas=highlight_betas,
        score_display=score_display,
        masses_by_beta=masses_by_beta,
        subplot_adjust=(PLOT_LEFT, PLOT_RIGHT, PLOT_BOTTOM, PLOT_TOP),
        use_tex=bool(int(args.use_tex)),
    )
    _render_plot(
        out_pdf=out_publication_pdf,
        out_png=out_publication_png,
        figsize=PUBLICATION_FIGSIZE,
        all_betas=all_betas,
        highlight_betas=highlight_betas,
        score_display=score_display,
        masses_by_beta=masses_by_beta,
        subplot_adjust=(PLOT_LEFT, PLOT_RIGHT, PLOT_BOTTOM, PLOT_TOP),
        use_tex=bool(int(args.use_tex)),
    )
    _render_plot(
        out_pdf=out_wider_pdf,
        out_png=out_wider_png,
        figsize=WIDER_REFERENCE_FIGSIZE,
        all_betas=all_betas,
        highlight_betas=highlight_betas,
        score_display=score_display,
        masses_by_beta=masses_by_beta,
        subplot_adjust=(PLOT_LEFT, PLOT_RIGHT, PLOT_BOTTOM, PLOT_TOP),
        use_tex=bool(int(args.use_tex)),
    )
    _write_json(
        run_json,
        {
            "script": SCRIPT_REL,
            "outdir": str(outdir.relative_to(ROOT) if outdir.is_relative_to(ROOT) else outdir),
            "n": int(args.n),
            "betas": [float(x) for x in all_betas.tolist()],
            "highlight_betas": [float(x) for x in highlight_betas.tolist()],
            "use_tex": bool(int(args.use_tex)),
            "recompute": bool(int(args.recompute)),
            "pdf": str(out_pdf.relative_to(ROOT) if out_pdf.is_relative_to(ROOT) else out_pdf),
            "png": str(out_png.relative_to(ROOT) if out_png.is_relative_to(ROOT) else out_png),
            "publication_pdf": str(out_publication_pdf.relative_to(ROOT) if out_publication_pdf.is_relative_to(ROOT) else out_publication_pdf),
            "publication_png": str(out_publication_png.relative_to(ROOT) if out_publication_png.is_relative_to(ROOT) else out_publication_png),
            "wider_reference_pdf": str(out_wider_pdf.relative_to(ROOT) if out_wider_pdf.is_relative_to(ROOT) else out_wider_pdf),
            "wider_reference_png": str(out_wider_png.relative_to(ROOT) if out_wider_png.is_relative_to(ROOT) else out_wider_png),
            "publication_natural_width_pt": float(PUBLICATION_FIG_W_PT),
            "publication_natural_height_pt": float(PUBLICATION_FIG_H_PT),
            "wider_reference_natural_width_pt": float(WIDER_REFERENCE_FIG_W_PT),
            "wider_reference_natural_height_pt": float(WIDER_REFERENCE_FIG_H_PT),
            "csv": str(data_csv.relative_to(ROOT) if data_csv.is_relative_to(ROOT) else data_csv),
        },
    )
    print(f"[experiment10] wrote {out_pdf}", flush=True)
    print(f"[experiment10] wrote {out_publication_pdf}", flush=True)
    print(f"[experiment10] wrote {out_wider_pdf}", flush=True)


if __name__ == "__main__":
    run()
