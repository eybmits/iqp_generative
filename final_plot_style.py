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


def single_panel_figure():
    apply_final_style()
    return plt.subplots(figsize=(FIG_SINGLE_W_IN, FIG_SINGLE_H_IN), constrained_layout=True)


def final_heatmap_cmap():
    return colors.LinearSegmentedColormap.from_list(
        "final_red_black",
        [HEATMAP_LOW, HEATMAP_MID, HEATMAP_HIGH],
    )


def save_pdf(fig, path: str | Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
