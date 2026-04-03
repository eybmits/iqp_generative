#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Toy n=4 parity-signature explainer for the unseen state 1001."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/iqp_generative_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/iqp_generative_cache")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

from experiment_3_beta_quality_coverage import (  # noqa: E402
    build_parity_matrix,
    empirical_dist,
    make_bits_table,
    sample_indices,
)
from experiment_4_recovery_sigmak_triplet import _reconstruct_bandlimited  # noqa: E402
from final_plot_style import MSE_COLOR, PARITY_COLOR  # noqa: E402


SCRIPT_REL = "experiment_11_toy_parity_signature_plot.py"
OUTPUT_STEM = "experiment_11_toy_parity_signature_plot"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM

TOY_N = 4
TOY_TRAIN_M = 12
TOY_TARGET_STATE = "1001"
TOY_PARTNER_STATE = "1100"
TOY_PARITIES = np.asarray(
    [
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    dtype=np.int8,
)
TOY_EVEN_WEIGHTS_BY_LABEL = {
    "0000": 0.04,
    "0011": 0.10,
    "0101": 0.12,
    "0110": 0.10,
    "1001": 0.28,
    "1010": 0.12,
    "1100": 0.18,
    "1111": 0.06,
}
TOY_SEED_SCAN_MAX = 200000

CARD_WSPACE = 0.10
BG = "#FFFFFF"
CARD_FILL = "#FBFBFC"
CARD_EDGE = "#C8D0DA"
TEXT = "#243247"
TEXT_SOFT = "#6C7585"
TARGET_EDGE = "#252525"
MSE_EDGE = "#4B7CC0"
TRAIN_FILL = "#8F9299"
TRAIN_EDGE = "#6E7178"
PARITY_EDGE = "#B45246"
SIGNATURE_NEG_COLOR = "#5C8A74"
SIGNATURE_POS_COLOR = "#D7A24A"
BADGE_FILL = "#EEF3FB"
BADGE_TEXT = "#3B6AA8"
UNSEEN_TEXT = "#9C2F35"
SEEN_TEXT = TEXT
DASH = "#6F7787"
ZERO_STUB = 0.007
BADGE_Y = 0.875
HEADER_Y = 0.875
TARGET_BAR_W = 0.08
TARGET_BAR_MAX_H = 0.27
TARGET_BAR_LW = 2.45
SVG_HASH_SALT = OUTPUT_STEM
REPRO_BUILD_DATETIME = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
REPRO_SVG_DATE = REPRO_BUILD_DATETIME.isoformat().replace("+00:00", "Z")
REPRO_COMMAND = (
    f"python {SCRIPT_REL} --outdir plots/{OUTPUT_STEM} --train-m {TOY_TRAIN_M}"
)


def _try_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _bit_labels(bits_table: np.ndarray) -> np.ndarray:
    return np.asarray(["".join(str(int(b)) for b in row) for row in bits_table], dtype=object)


def _even_mask(bits_table: np.ndarray) -> np.ndarray:
    return (np.sum(np.asarray(bits_table, dtype=np.int8), axis=1) % 2) == 0


def _signature_tuple(col: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(col, dtype=np.int64).tolist())


def _toy_target_distribution(labels: Sequence[str]) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.float64)
    for idx, label in enumerate(labels):
        out[idx] = float(TOY_EVEN_WEIGHTS_BY_LABEL.get(str(label), 0.0))
    total = float(np.sum(out))
    if total <= 0.0:
        raise RuntimeError("Toy target distribution is empty.")
    return out / total


def _find_toy_seed(
    *,
    p_target: np.ndarray,
    P: np.ndarray,
    target_idx: int,
    partner_idx: int,
    train_m: int,
) -> int:
    for seed in range(TOY_SEED_SCAN_MAX):
        idxs_train = sample_indices(p_target, int(train_m), seed=int(seed))
        counts = np.bincount(np.asarray(idxs_train, dtype=np.int64), minlength=p_target.size)
        if int(counts[target_idx]) != 0:
            continue
        if int(counts[partner_idx]) <= 0:
            continue
        emp = empirical_dist(idxs_train, p_target.size)
        q_parity = _reconstruct_bandlimited(P, P @ emp, TOY_N)
        if float(q_parity[target_idx]) <= 0.0:
            continue
        return int(seed)
    raise RuntimeError("Failed to find a toy sample with unseen 1001 and seen 1100.")


def _format_signature(sig: Sequence[int]) -> str:
    return "(" + ", ".join(f"{int(x):+d}" for x in sig) + ")"


def _build_payload(*, train_m: int) -> Dict[str, object]:
    bits_table = make_bits_table(TOY_N)
    labels = [str(x) for x in _bit_labels(bits_table).tolist()]
    p_target = _toy_target_distribution(labels)

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    target_idx = int(label_to_idx[TOY_TARGET_STATE])
    partner_idx = int(label_to_idx[TOY_PARTNER_STATE])

    P = build_parity_matrix(TOY_PARITIES, bits_table)
    target_signature = _signature_tuple(P[:, target_idx])
    partner_signature = _signature_tuple(P[:, partner_idx])
    if target_signature != partner_signature:
        raise RuntimeError("1001 and 1100 must share the same sampled signature.")

    toy_seed = _find_toy_seed(
        p_target=p_target,
        P=P,
        target_idx=target_idx,
        partner_idx=partner_idx,
        train_m=int(train_m),
    )

    idxs_train = sample_indices(p_target, int(train_m), seed=int(toy_seed))
    counts = np.bincount(np.asarray(idxs_train, dtype=np.int64), minlength=p_target.size)
    emp = empirical_dist(idxs_train, p_target.size)
    z_hat = P @ emp
    q_parity = _reconstruct_bandlimited(P, z_hat, TOY_N)

    pair_labels = np.asarray([TOY_TARGET_STATE, TOY_PARTNER_STATE], dtype=object)
    pair_idx = np.asarray([target_idx, partner_idx], dtype=np.int64)
    pair_target = np.asarray(p_target[pair_idx], dtype=np.float64)
    pair_emp = np.asarray(emp[pair_idx], dtype=np.float64)
    pair_parity = np.asarray(q_parity[pair_idx], dtype=np.float64)
    pair_counts = np.asarray(counts[pair_idx], dtype=np.int64)
    shared_target_height = float(np.max(pair_target))
    pair_target_display = np.full(2, shared_target_height, dtype=np.float64)
    # For the didactic plot, show D_train/MSE at the target height on the seen state
    # so the comparison isolates "seen vs unseen" rather than sample-count noise.
    pair_dtrain_display = np.asarray([0.0, shared_target_height], dtype=np.float64)
    pair_eqp_mse = np.asarray(pair_dtrain_display, dtype=np.float64)
    pair_eqp_parity = np.asarray(pair_parity, dtype=np.float64)

    if float(pair_emp[0]) != 0.0:
        raise RuntimeError("1001 unexpectedly appeared in the toy sample.")
    if float(pair_parity[0]) <= 0.0:
        raise RuntimeError("Parity reconstruction failed to assign positive mass to 1001.")

    even_idx = np.where(_even_mask(bits_table))[0]
    same_signature_even = [
        labels[idx]
        for idx in even_idx.tolist()
        if _signature_tuple(P[:, idx]) == target_signature
    ]

    return {
        "labels": np.asarray(labels, dtype=object),
        "pair_labels": pair_labels,
        "pair_idx": pair_idx,
        "pair_target": pair_target,
        "pair_target_display": pair_target_display,
        "pair_emp": pair_emp,
        "pair_parity": pair_parity,
        "pair_dtrain_display": pair_dtrain_display,
        "pair_eqp_mse": pair_eqp_mse,
        "pair_eqp_parity": pair_eqp_parity,
        "pair_counts": pair_counts,
        "signatures_pair": np.asarray([target_signature, partner_signature], dtype=np.int64),
        "toy_seed": int(toy_seed),
        "train_m": int(train_m),
        "moments": np.asarray(z_hat, dtype=np.float64),
        "parities": np.asarray(TOY_PARITIES, dtype=np.int8),
        "odd_mass_parity": float(np.sum(q_parity[~_even_mask(bits_table)])),
        "same_signature_even": np.asarray(same_signature_even, dtype=object),
        "target_signature": np.asarray(target_signature, dtype=np.int64),
    }


def _apply_card_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": SVG_HASH_SALT,
        }
    )


def _mass_height(value: float, *, vmax: float, max_height: float, zero_stub: float = 0.0) -> float:
    value = float(value)
    if value <= 1e-12:
        return float(zero_stub)
    if vmax <= 0.0:
        return 0.0
    return float(max_height) * value / float(vmax)


def _add_card(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.07),
            0.96,
            0.88,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            linewidth=1.35,
            edgecolor=CARD_EDGE,
            facecolor=CARD_FILL,
            zorder=0,
        )
    )


def _add_badge(ax: plt.Axes, n: int) -> None:
    ax.add_patch(Circle((0.085, BADGE_Y), 0.055, facecolor=BADGE_FILL, edgecolor="none", zorder=2))
    ax.text(
        0.085,
        BADGE_Y,
        str(int(n)),
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=BADGE_TEXT,
        zorder=3,
    )


def _outlined_bar(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    fill_color: str | None = None,
    lw: float = 2.35,
    zorder: int = 5,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none" if fill_color is None else fill_color,
            edgecolor=color,
            linewidth=lw,
            zorder=zorder,
        )
    )


def _filled_bar(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    edge_color: str | None = None,
    lw: float = 1.2,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=color,
            edgecolor=color if edge_color is None else edge_color,
            linewidth=0.0 if edge_color is None else lw,
            zorder=zorder,
        )
    )


def _dashed_group(ax: plt.Axes, *, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor=DASH,
            linewidth=1.85,
            linestyle="--",
            zorder=6,
        )
    )


def _state_label(ax: plt.Axes, *, x: float, y: float, label: str, unseen: bool, show_status: bool = True) -> None:
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold" if unseen else "normal",
        color=UNSEEN_TEXT if unseen else SEEN_TEXT,
    )
    if show_status:
        ax.text(
            x,
            y - 0.045,
            "unseen" if unseen else "seen",
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT_SOFT,
        )


def _add_panel1(ax: plt.Axes, payload: Dict[str, object]) -> None:
    labels = [str(x) for x in np.asarray(payload["pair_labels"], dtype=object).tolist()]
    target = np.asarray(payload["pair_target_display"], dtype=np.float64)
    dtrain = np.asarray(payload["pair_dtrain_display"], dtype=np.float64)
    vmax = float(max(np.max(target), np.max(dtrain)))

    _add_card(ax)
    ax.text(0.50, HEADER_Y, "Missing Support", ha="center", va="center", fontsize=21, fontweight="bold", color=TEXT)

    legend_y = 0.692
    legend_w = 0.058
    legend_h = 0.030
    _outlined_bar(ax, x=0.205, y=legend_y, w=legend_w, h=legend_h, color=TARGET_EDGE, lw=2.0)
    ax.text(0.283, legend_y + legend_h / 2.0, "target", ha="left", va="center", fontsize=14.1, color=TEXT)
    _filled_bar(ax, x=0.575, y=legend_y, w=legend_w, h=legend_h, color=TRAIN_FILL, edge_color=TRAIN_EDGE, lw=1.2)
    ax.text(0.653, legend_y + legend_h / 2.0, "train", ha="left", va="center", fontsize=14.1, color=TEXT)

    baseline = 0.34
    state_centers = [0.33, 0.72]
    pair_offset = 0.045

    for idx, value in enumerate(target.tolist()):
        _outlined_bar(
            ax,
            x=float(state_centers[idx] - pair_offset - TARGET_BAR_W / 2.0),
            y=baseline,
            w=TARGET_BAR_W,
            h=_mass_height(value, vmax=vmax, max_height=TARGET_BAR_MAX_H),
            color=TARGET_EDGE,
            lw=TARGET_BAR_LW,
        )
    for idx, value in enumerate(dtrain.tolist()):
        _filled_bar(
            ax,
            x=float(state_centers[idx] + pair_offset - TARGET_BAR_W / 2.0),
            y=baseline,
            w=TARGET_BAR_W,
            h=_mass_height(value, vmax=vmax, max_height=TARGET_BAR_MAX_H, zero_stub=ZERO_STUB if idx == 0 else 0.0),
            color=TRAIN_FILL,
            edge_color=TRAIN_EDGE,
            lw=1.2,
        )

    _state_label(ax, x=float(state_centers[0]), y=0.29, label=labels[0], unseen=True)
    _state_label(ax, x=float(state_centers[1]), y=0.29, label=labels[1], unseen=False)
    ax.text(0.50, 0.16, f"{labels[0]} has zero train support", ha="center", va="center", fontsize=17, color=UNSEEN_TEXT)


def _add_panel2(ax: plt.Axes, payload: Dict[str, object]) -> None:
    labels = [str(x) for x in np.asarray(payload["pair_labels"], dtype=object).tolist()]
    signatures = np.asarray(payload["signatures_pair"], dtype=np.int64)
    parity_masks = ["".join(str(int(b)) for b in row) for row in np.asarray(payload["parities"], dtype=np.int8)]

    _add_card(ax)
    ax.text(0.50, HEADER_Y, "Shared parity cue", ha="center", va="center", fontsize=22, fontweight="bold", color=TEXT)

    x0 = 0.29
    y_rows = [0.49, 0.33]
    cell_w = 0.17
    cell_h = 0.09
    gap = 0.02
    cell_block_w = 3.0 * cell_w + 2.0 * gap
    group_w = 3.0 * cell_w + 2.0 * gap + 0.05
    group_center = x0 + cell_block_w / 2.0
    label_x = x0 - 0.075

    ax.text(group_center - 0.025, 0.685, "same fingerprint", ha="center", va="center", fontsize=18, color=TEXT)

    row_centers = [row_y + cell_h / 2.0 for row_y in y_rows]
    ax.text(label_x, row_centers[0] + 0.015, labels[0], ha="right", va="center", fontsize=19, fontweight="bold", color=UNSEEN_TEXT)
    ax.text(label_x, row_centers[0] - 0.030, "unseen", ha="right", va="center", fontsize=11, color=TEXT_SOFT)
    ax.text(label_x, row_centers[1] + 0.015, labels[1], ha="right", va="center", fontsize=19, color=TEXT)
    ax.text(label_x, row_centers[1] - 0.030, "seen", ha="right", va="center", fontsize=11, color=TEXT_SOFT)

    for row_y in y_rows:
        _dashed_group(ax, x=x0 - 0.025, y=row_y - 0.015, w=group_w, h=cell_h + 0.03)

    for row_idx, row_y in enumerate(y_rows):
        for col_idx in range(signatures.shape[1]):
            value = int(signatures[row_idx, col_idx])
            x = x0 + col_idx * (cell_w + gap)
            ax.add_patch(
                FancyBboxPatch(
                    (x, row_y),
                    cell_w,
                    cell_h,
                    boxstyle="round,pad=0.0,rounding_size=0.012",
                    linewidth=0.0,
                    facecolor=SIGNATURE_NEG_COLOR if value < 0 else SIGNATURE_POS_COLOR,
                    zorder=5,
                )
            )
            ax.text(
                x + cell_w / 2.0,
                row_y + cell_h / 2.0,
                f"{value:+d}",
                ha="center",
                va="center",
                fontsize=18,
                color="white",
                fontweight="bold",
                zorder=6,
            )

    for col_idx, label in enumerate(parity_masks):
        x = x0 + col_idx * (cell_w + gap) + cell_w / 2.0
        ax.text(x, 0.275, label, ha="center", va="center", fontsize=12, color=TEXT_SOFT)
    ax.text(group_center, 0.22, r"sampled parity masks $\alpha$", ha="center", va="center", fontsize=12, color=TEXT_SOFT)


def _add_panel3(ax: plt.Axes, payload: Dict[str, object]) -> None:
    labels = [str(x) for x in np.asarray(payload["pair_labels"], dtype=object).tolist()]
    target = np.asarray(payload["pair_target_display"], dtype=np.float64)
    eqp_mse = np.asarray(payload["pair_eqp_mse"], dtype=np.float64)
    eqp_parity = np.asarray(payload["pair_eqp_parity"], dtype=np.float64)
    vmax = float(max(np.max(target), np.max(eqp_mse), np.max(eqp_parity)))

    _add_card(ax)
    ax.text(0.50, HEADER_Y, "Output behavior", ha="center", va="center", fontsize=21, fontweight="bold", color=TEXT)

    group_centers = [0.29, 0.75]
    state_gap = 0.20
    state_centers_left = [group_centers[0] - state_gap / 2.0, group_centers[0] + state_gap / 2.0]
    state_centers_right = [group_centers[1] - state_gap / 2.0, group_centers[1] + state_gap / 2.0]

    ax.text(group_centers[0], 0.76, "IQP-MSE", ha="center", va="center", fontsize=16.5, fontweight="bold", color=TEXT)
    ax.text(group_centers[1], 0.76, "IQP-Parity", ha="center", va="center", fontsize=16.5, fontweight="bold", color=TEXT)

    legend_y = 0.688
    legend_w = 0.058
    legend_h = 0.030
    text_gap = 0.012
    legend_fs = 14.1
    _outlined_bar(ax, x=0.055, y=legend_y, w=legend_w, h=legend_h, color=TARGET_EDGE, lw=2.0)
    ax.text(0.055 + legend_w + text_gap, legend_y + legend_h / 2.0, "target", ha="left", va="center", fontsize=legend_fs, color=TEXT)
    _filled_bar(ax, x=0.300, y=legend_y, w=legend_w, h=legend_h, color=MSE_COLOR, edge_color=MSE_EDGE, lw=1.5)
    ax.text(0.300 + legend_w + text_gap, legend_y + legend_h / 2.0, "MSE mass", ha="left", va="center", fontsize=legend_fs, color=TEXT)
    _filled_bar(ax, x=0.630, y=legend_y, w=legend_w, h=legend_h, color=PARITY_COLOR, edge_color=PARITY_EDGE, lw=1.5)
    ax.text(0.630 + legend_w + text_gap, legend_y + legend_h / 2.0, "Parity mass", ha="left", va="center", fontsize=legend_fs, color=TEXT)

    baseline = 0.34
    fill_bar_w = TARGET_BAR_W
    max_height = TARGET_BAR_MAX_H
    pair_offset = 0.045

    def draw_group(
        centers: Sequence[float],
        fill_values: np.ndarray,
        fill_color: str,
        outline_color: str,
        fill_edge_color: str | None,
    ) -> None:
        for idx, center in enumerate(centers):
            outline_x = float(center) - pair_offset - TARGET_BAR_W / 2.0
            fill_x = float(center) + pair_offset - fill_bar_w / 2.0
            fill_value = float(fill_values[idx])
            fill_lw = 1.2 if fill_value <= 1e-12 else TARGET_BAR_LW
            _outlined_bar(
                ax,
                x=outline_x,
                y=baseline,
                w=TARGET_BAR_W,
                h=_mass_height(float(target[idx]), vmax=vmax, max_height=max_height),
                color=outline_color,
                lw=TARGET_BAR_LW,
            )
            _filled_bar(
                ax,
                x=fill_x,
                y=baseline,
                w=fill_bar_w,
                h=_mass_height(fill_value, vmax=vmax, max_height=max_height, zero_stub=ZERO_STUB if idx == 0 else 0.0),
                color=fill_color,
                edge_color=fill_edge_color,
                lw=fill_lw,
            )
            _state_label(ax, x=float(center), y=0.225, label=labels[idx], unseen=(idx == 0), show_status=False)

    draw_group(state_centers_left, eqp_mse, MSE_COLOR, TARGET_EDGE, MSE_EDGE)
    draw_group(state_centers_right, eqp_parity, PARITY_COLOR, TARGET_EDGE, PARITY_EDGE)

    ax.text(
        0.50,
        0.125,
        f"Parity-aware fitting\nrestores mass on {labels[0]}",
        ha="center",
        va="center",
        fontsize=14.5,
        color=TEXT,
        linespacing=0.98,
    )


def _render_plot(*, out_pdf: Path, out_png: Path, out_svg: Path, payload: Dict[str, object]) -> None:
    _apply_card_style()
    fig = plt.figure(figsize=(13.6, 4.75), facecolor=BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=CARD_WSPACE)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    _add_panel1(ax1, payload)
    _add_panel2(ax2, payload)
    _add_panel3(ax3, payload)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.035)
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", facecolor=BG)
    fig.savefig(
        out_pdf,
        format="pdf",
        bbox_inches="tight",
        facecolor=BG,
        metadata={
            "Creator": SCRIPT_REL,
            "Producer": "Matplotlib pdf backend",
            "CreationDate": REPRO_BUILD_DATETIME,
        },
    )
    fig.savefig(
        out_svg,
        format="svg",
        bbox_inches="tight",
        facecolor=BG,
        metadata={
            "Date": REPRO_SVG_DATE,
            "Creator": SCRIPT_REL,
        },
    )
    plt.close(fig)


def _write_readme(
    path: Path,
    *,
    out_pdf: Path,
    out_png: Path,
    out_svg: Path,
    data_npz: Path,
    run_json: Path,
    payload: Dict[str, object],
) -> None:
    lines = [
        "# Experiment 11 toy parity signature plot",
        "",
        "This directory contains a three-card n=4 explainer for why a parity-based fit can",
        "assign positive mass to an unseen state such as `1001`.",
        "",
        "Story:",
        "",
        "- card 1: target mass versus `D_train`, highlighting that `1001` has zero train support",
        "- card 2: `1001` and `1100` share the same sampled parity signature",
        "- card 3: `IQP MSE` stays on the seen state, while `IQP Parity` restores mass on `1001`",
        "",
        "Key values:",
        "",
        f"- toy sample seed: `{int(payload['toy_seed'])}`",
        f"- sample size: `{int(payload['train_m'])}`",
        f"- `{TOY_TARGET_STATE}` displayed `D_train` mass: `{float(np.asarray(payload['pair_dtrain_display'])[0]):.4f}`",
        f"- `{TOY_TARGET_STATE}` `IQP Parity` mass: `{float(np.asarray(payload['pair_eqp_parity'])[0]):.4f}`",
        f"- shared signature: `{_format_signature(np.asarray(payload['target_signature'], dtype=np.int64))}`",
        "",
        "Saved artifacts:",
        "",
        f"- PDF: `{_try_rel(out_pdf)}`",
        f"- PNG: `{_try_rel(out_png)}`",
        f"- SVG: `{_try_rel(out_svg)}`",
        f"- data NPZ: `{_try_rel(data_npz)}`",
        f"- run config: `{_try_rel(run_json)}`",
        "",
        "Reproduce:",
        "",
        f"- from repo root: `{REPRO_COMMAND}`",
        "",
        f"- source driver: `{SCRIPT_REL}`",
        f"- outdir: `{_try_rel(path.parent)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    ap = argparse.ArgumentParser(description="Toy n=4 plot for parity-signature coupling of 1001 and 1100.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--train-m", type=int, default=TOY_TRAIN_M)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    payload = _build_payload(train_m=int(args.train_m))

    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}.png"
    out_svg = outdir / f"{OUTPUT_STEM}.svg"
    data_npz = outdir / f"{OUTPUT_STEM}_data.npz"
    run_json = outdir / "RUN_CONFIG.json"
    readme = outdir / "README.md"

    _render_plot(out_pdf=out_pdf, out_png=out_png, out_svg=out_svg, payload=payload)
    np.savez(
        data_npz,
        labels=np.asarray(payload["labels"], dtype=object),
        pair_labels=np.asarray(payload["pair_labels"], dtype=object),
        pair_idx=np.asarray(payload["pair_idx"], dtype=np.int64),
        pair_target=np.asarray(payload["pair_target"], dtype=np.float64),
        pair_target_display=np.asarray(payload["pair_target_display"], dtype=np.float64),
        pair_emp=np.asarray(payload["pair_emp"], dtype=np.float64),
        pair_parity=np.asarray(payload["pair_parity"], dtype=np.float64),
        pair_dtrain_display=np.asarray(payload["pair_dtrain_display"], dtype=np.float64),
        pair_eqp_mse=np.asarray(payload["pair_eqp_mse"], dtype=np.float64),
        pair_eqp_parity=np.asarray(payload["pair_eqp_parity"], dtype=np.float64),
        pair_counts=np.asarray(payload["pair_counts"], dtype=np.int64),
        signatures_pair=np.asarray(payload["signatures_pair"], dtype=np.int64),
        toy_seed=np.asarray([int(payload["toy_seed"])], dtype=np.int64),
        train_m=np.asarray([int(payload["train_m"])], dtype=np.int64),
        moments=np.asarray(payload["moments"], dtype=np.float64),
        parities=np.asarray(payload["parities"], dtype=np.int8),
        odd_mass_parity=np.asarray([float(payload["odd_mass_parity"])], dtype=np.float64),
        same_signature_even=np.asarray(payload["same_signature_even"], dtype=object),
        target_signature=np.asarray(payload["target_signature"], dtype=np.int64),
    )
    _write_json(
        run_json,
        {
            "script": SCRIPT_REL,
            "outdir": _try_rel(outdir),
            "command": REPRO_COMMAND,
            "n": TOY_N,
            "train_m": int(args.train_m),
            "toy_seed": int(payload["toy_seed"]),
            "target_state": TOY_TARGET_STATE,
            "partner_state": TOY_PARTNER_STATE,
            "parities": np.asarray(TOY_PARITIES, dtype=np.int64).tolist(),
            "pdf": _try_rel(out_pdf),
            "png": _try_rel(out_png),
            "svg": _try_rel(out_svg),
            "data_npz": _try_rel(data_npz),
            "readme": _try_rel(readme),
        },
    )
    _write_readme(
        readme,
        out_pdf=out_pdf,
        out_png=out_png,
        out_svg=out_svg,
        data_npz=data_npz,
        run_json=run_json,
        payload=payload,
    )

    print(f"[experiment11-toy] wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    run()
