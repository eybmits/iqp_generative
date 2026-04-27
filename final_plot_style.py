#!/usr/bin/env python3
"""Canonical final-figure style extracted from Experiment 1.

This module is intentionally small and reusable. Future final plot scripts can
import these constants/helpers to keep dimensions and styling aligned with the
approved Experiment 1 figure family, while Experiment 1 itself stays standalone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors


FIG_SINGLE_W_PT = 243.12
FIG_SINGLE_H_PT = 185.52
FIG_SINGLE_W_IN = FIG_SINGLE_W_PT / 72.0
FIG_SINGLE_H_IN = FIG_SINGLE_H_PT / 72.0
TEX_PT_PER_IN = 72.27
IEEE_COLUMN_W_PT = 252.0
IEEE_COLUMN_W_IN = IEEE_COLUMN_W_PT / TEX_PT_PER_IN
IEEE_TWOUP_PANEL_W_PT = IEEE_COLUMN_W_PT
IEEE_TWOUP_PANEL_H_PT = FIG_SINGLE_H_PT * (IEEE_TWOUP_PANEL_W_PT / 320.0)
IEEE_TWOUP_PANEL_W_IN = IEEE_TWOUP_PANEL_W_PT / TEX_PT_PER_IN
IEEE_TWOUP_PANEL_H_IN = IEEE_TWOUP_PANEL_H_PT / TEX_PT_PER_IN
IEEE_THREEUP_PANEL_W_PT = 216.0
IEEE_THREEUP_PANEL_H_PT = FIG_SINGLE_H_PT * (IEEE_THREEUP_PANEL_W_PT / FIG_SINGLE_W_PT)
IEEE_THREEUP_PANEL_W_IN = IEEE_THREEUP_PANEL_W_PT / 72.0
IEEE_THREEUP_PANEL_H_IN = IEEE_THREEUP_PANEL_H_PT / 72.0
IEEE_COMPACT_PANEL_W_PT = FIG_SINGLE_W_PT
IEEE_COMPACT_PANEL_H_PT = FIG_SINGLE_H_PT
IEEE_COMPACT_PANEL_W_IN = IEEE_COMPACT_PANEL_W_PT / TEX_PT_PER_IN
IEEE_COMPACT_PANEL_H_IN = IEEE_COMPACT_PANEL_H_PT / TEX_PT_PER_IN
IEEE_BASE_FONTSIZE_PT = 10.0
IEEE_AXIS_LABELSIZE_PT = 11.0
IEEE_TICK_FONTSIZE_PT = 10.0
IEEE_LEGEND_FONTSIZE_PT = 9.0
IEEE_LATEX_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}\usepackage{bm}"

TARGET_COLOR = "#2F2A2B"
PARITY_COLOR = "#E46C5B"
MSE_COLOR = "#5B9BE6"
UNIFORM_COLOR = "#C6C9CF"
HEATMAP_LOW = "#F04B4C"
HEATMAP_MID = "#8E111B"
HEATMAP_HIGH = "#0D0D0F"
ACCENT_DARK = "#171717"
TEXT_DARK = "#222222"
TEXT_MID = "#8A8A8A"
AXIS_ZERO = "#C7C7C7"


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


def apply_ieee_latex_style(*, use_tex: bool = True) -> None:
    """Match IEEEtran-like sizing for figures intended for scale=1.0 inclusion.

    This keeps the canvas at a fixed physical size and avoids `bbox_inches="tight"`
    downstream, so the exported PDF natural size matches the LaTeX inclusion size.
    """
    rc = {
        "font.size": IEEE_BASE_FONTSIZE_PT,
        "axes.labelsize": IEEE_AXIS_LABELSIZE_PT,
        "axes.titlesize": IEEE_AXIS_LABELSIZE_PT,
        "xtick.labelsize": IEEE_TICK_FONTSIZE_PT,
        "ytick.labelsize": IEEE_TICK_FONTSIZE_PT,
        "legend.fontsize": IEEE_LEGEND_FONTSIZE_PT,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#7A7A7A",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.7,
        "grid.linestyle": "-",
        "grid.linewidth": 0.75,
        "grid.color": "#D7D7D7",
    }
    if use_tex:
        rc.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": IEEE_LATEX_PREAMBLE,
            }
        )
    else:
        rc.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": [
                    "STIX Two Text",
                    "Times New Roman",
                    "Times",
                    "Nimbus Roman",
                    "DejaVu Serif",
                ],
                "mathtext.fontset": "stix",
            }
        )
    plt.rcParams.update(rc)


def single_panel_figure():
    apply_final_style()
    return plt.subplots(figsize=(FIG_SINGLE_W_IN, FIG_SINGLE_H_IN), constrained_layout=True)


def ieee_twoup_panel_figure(*, use_tex: bool = True, constrained_layout: bool = False):
    apply_ieee_latex_style(use_tex=use_tex)
    return plt.subplots(
        figsize=(IEEE_TWOUP_PANEL_W_IN, IEEE_TWOUP_PANEL_H_IN),
        constrained_layout=constrained_layout,
    )


def ieee_compact_panel_figure(*, use_tex: bool = True, constrained_layout: bool = False):
    apply_ieee_latex_style(use_tex=use_tex)
    return plt.subplots(
        figsize=(IEEE_COMPACT_PANEL_W_IN, IEEE_COMPACT_PANEL_H_IN),
        constrained_layout=constrained_layout,
    )


def ieee_threeup_panel_figure(*, use_tex: bool = True, constrained_layout: bool = False):
    apply_ieee_latex_style(use_tex=use_tex)
    return plt.subplots(
        figsize=(IEEE_THREEUP_PANEL_W_IN, IEEE_THREEUP_PANEL_H_IN),
        constrained_layout=constrained_layout,
    )


def final_heatmap_cmap():
    return colors.LinearSegmentedColormap.from_list(
        "final_red_black",
        [HEATMAP_LOW, HEATMAP_MID, HEATMAP_HIGH],
    )


def save_pdf(fig, path: str | Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_exact_figure(fig, path: str | Path, *, dpi: int | None = None) -> None:
    """Save without tight-cropping so the PDF natural size stays deterministic."""
    fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
