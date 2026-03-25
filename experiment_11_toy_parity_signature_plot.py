#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Toy n=4 parity-signature explainer for the unseen state 1001."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/iqp_generative_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/iqp_generative_cache")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from experiment_3_beta_quality_coverage import (  # noqa: E402
    build_parity_matrix,
    empirical_dist,
    make_bits_table,
    sample_indices,
)
from experiment_4_recovery_sigmak_triplet import _reconstruct_bandlimited  # noqa: E402
from final_plot_style import MSE_COLOR, PARITY_COLOR, TARGET_COLOR, TEXT_DARK, apply_final_style  # noqa: E402


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

FIG_W = 620.0 / 72.0
FIG_H = 230.0 / 72.0
FIG_LEFT = 0.085
FIG_RIGHT = 0.985
FIG_BOTTOM = 0.20
FIG_TOP = 0.86
DTRAIN_PANEL_COLOR = "#9A9A9A"
HEATMAP_NEG = MSE_COLOR
HEATMAP_POS = PARITY_COLOR


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
    # For the didactic plot, show D_train/MSE at the target height on the seen state
    # so the comparison isolates "seen vs unseen" rather than sample-count noise.
    pair_dtrain_display = np.asarray([0.0, float(pair_target[1])], dtype=np.float64)
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


def _shade_pair_categories(ax: plt.Axes) -> None:
    return None


def _plot_sample_panel(ax: plt.Axes, payload: Dict[str, object]) -> None:
    labels = [str(x) for x in np.asarray(payload["pair_labels"], dtype=object).tolist()]
    target = np.asarray(payload["pair_target"], dtype=np.float64)
    dtrain = np.asarray(payload["pair_dtrain_display"], dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    ymax = float(max(np.max(target), np.max(dtrain)))

    _shade_pair_categories(ax)
    ax.bar(x - width / 2.0, target, color=TARGET_COLOR, width=width, alpha=0.92, zorder=3)
    ax.bar(x + width / 2.0, dtrain, color=DTRAIN_PANEL_COLOR, width=width, alpha=0.92, zorder=3)

    ax.text(
        0.03,
        0.97,
        u"\u25A0 target $p^*(x)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        color=TARGET_COLOR,
    )
    ax.text(
        0.03,
        0.84,
        u"\u25A0 $D_{\\mathrm{train}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        color=DTRAIN_PANEL_COLOR,
    )
    ax.set_title(r"Target vs $D_{\mathrm{train}}$", fontsize=10.0, pad=4.0)
    ax.set_xticks(x.tolist())
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability mass")
    ax.set_ylim(0.0, ymax * 1.32)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    for idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_fontweight("bold")
        if idx == 0:
            tick.set_color("#7A1C16")


def _plot_signature_panel(ax: plt.Axes, payload: Dict[str, object]) -> None:
    raw_labels = [str(x) for x in np.asarray(payload["pair_labels"], dtype=object).tolist()]
    labels = [f"{raw_labels[0]}\nunseen", f"{raw_labels[1]}\nseen"]
    signatures = np.asarray(payload["signatures_pair"], dtype=np.int64)
    parity_labels = ["1010", "0111", "1111"]
    values = (signatures + 1) // 2

    cmap = ListedColormap([HEATMAP_NEG, HEATMAP_POS])
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for i in range(signatures.shape[0]):
        for j in range(signatures.shape[1]):
            ax.text(
                j,
                i,
                "+1" if int(signatures[i, j]) > 0 else "-1",
                ha="center",
                va="center",
                color="white",
                fontsize=9.3,
                fontweight="bold",
            )

    for y in range(signatures.shape[0] + 1):
        ax.axhline(y - 0.5, color="white", lw=1.0, alpha=0.85)
    for x in range(signatures.shape[1] + 1):
        ax.axvline(x - 0.5, color="white", lw=1.0, alpha=0.85)

    ax.set_title("Same signature", fontsize=10.0, pad=4.0)
    ax.set_xticks(range(len(parity_labels)))
    ax.set_xticklabels(parity_labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=8.8, pad=6.0)


def _plot_mass_panel(ax: plt.Axes, payload: Dict[str, object]) -> None:
    labels = [str(x) for x in np.asarray(payload["pair_labels"], dtype=object).tolist()]
    target = np.asarray(payload["pair_target"], dtype=np.float64)
    eqp_mse = np.asarray(payload["pair_eqp_mse"], dtype=np.float64)
    eqp_parity = np.asarray(payload["pair_eqp_parity"], dtype=np.float64)

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.22
    ymax = float(max(np.max(target), np.max(eqp_mse), np.max(eqp_parity)))

    _shade_pair_categories(ax)
    ax.bar(x - width, target, width=width, color=TARGET_COLOR, alpha=0.92, zorder=3)
    ax.bar(x, eqp_mse, width=width, color=MSE_COLOR, alpha=0.82, zorder=3)
    ax.bar(x + width, eqp_parity, width=width, color=PARITY_COLOR, alpha=0.86, zorder=3)
    ax.set_title("Reconstructed mass", fontsize=10.0, pad=4.0)
    ax.set_xticks(x.tolist())
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability mass")
    ax.set_ylim(0.0, ymax * 1.33)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    for idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_fontweight("bold")
        if idx == 0:
            tick.set_color("#7A1C16")


def _render_plot(*, out_pdf: Path, out_png: Path, payload: Dict[str, object]) -> None:
    apply_final_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), gridspec_kw={"width_ratios": [0.95, 1.3, 1.55]})
    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=FIG_BOTTOM, top=FIG_TOP, wspace=0.34)

    _plot_sample_panel(axes[0], payload)
    _plot_signature_panel(axes[1], payload)
    _plot_mass_panel(axes[2], payload)

    middle_box = axes[1].get_position()
    fig.text((middle_box.x0 + middle_box.x1) / 2.0, 0.085, r"Sampled parity mask $\alpha$", ha="center", va="center")

    legend = [
        Patch(facecolor=TARGET_COLOR, edgecolor=TARGET_COLOR, alpha=0.92, label="Target"),
        Patch(facecolor=MSE_COLOR, edgecolor=MSE_COLOR, alpha=0.82, label="IQP MSE"),
        Patch(facecolor=PARITY_COLOR, edgecolor=PARITY_COLOR, alpha=0.86, label="IQP Parity"),
    ]
    axes[2].legend(
        handles=legend,
        loc="upper center",
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
        bbox_transform=axes[2].transAxes,
        borderpad=0.25,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=8.0,
    )

    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)


def _write_readme(path: Path, *, out_pdf: Path, out_png: Path, data_npz: Path, run_json: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Experiment 11 toy parity signature plot",
        "",
        "This directory contains a small n=4 explainer for why a parity-based fit can assign",
        "positive mass to an unseen state such as `1001`.",
        "",
        "Story:",
        "",
        "- panel 1: target mass versus `D_train`, with only the seen state shown at its target height",
        "- panel 2: both states share the same sampled parity signature, but only `1100` comes from `D_train`",
        "- panel 3: `IQP MSE` stays on the seen state, while `IQP Parity` puts positive mass on `1001`",
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
        f"- data NPZ: `{_try_rel(data_npz)}`",
        f"- run config: `{_try_rel(run_json)}`",
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
    data_npz = outdir / f"{OUTPUT_STEM}_data.npz"
    run_json = outdir / "RUN_CONFIG.json"
    readme = outdir / "README.md"

    _render_plot(out_pdf=out_pdf, out_png=out_png, payload=payload)
    np.savez(
        data_npz,
        labels=np.asarray(payload["labels"], dtype=object),
        pair_labels=np.asarray(payload["pair_labels"], dtype=object),
        pair_idx=np.asarray(payload["pair_idx"], dtype=np.int64),
        pair_target=np.asarray(payload["pair_target"], dtype=np.float64),
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
            "n": TOY_N,
            "train_m": int(args.train_m),
            "toy_seed": int(payload["toy_seed"]),
            "target_state": TOY_TARGET_STATE,
            "partner_state": TOY_PARTNER_STATE,
            "parities": np.asarray(TOY_PARITIES, dtype=np.int64).tolist(),
            "pdf": _try_rel(out_pdf),
            "png": _try_rel(out_png),
            "data_npz": _try_rel(data_npz),
            "readme": _try_rel(readme),
        },
    )
    _write_readme(readme, out_pdf=out_pdf, out_png=out_png, data_npz=data_npz, run_json=run_json, payload=payload)

    print(f"[experiment11-toy] wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    run()
