#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 9: single-seed beta sweep in the spirit of Experiment 1 panel (c).

Compares four references over beta for one matched training seed:
- Target p*
- IQP parity (fixed sigma, K)
- IQP MSE
- Uniform
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiment_1_kl_diagnostics import (
    COLOR_IQP,
    COLOR_IQP_MSE,
    COLOR_NEUTRAL,
    COLOR_TARGET,
    K_VALUES,
    SIGMA_VALUES,
    apply_style,
    build_parity_matrix,
    build_target_distribution_paper,
    empirical_dist,
    forward_kl,
    make_bits_table,
    sample_alphas,
    sample_indices,
    train_iqp_qcbm,
    train_iqp_qcbm_prob_mse,
)


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_9_single_seed_beta_panel_c_analogue.py"
OUTPUT_STEM = "experiment_9_single_seed_beta_panel_c_analogue"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM
DEFAULT_BETAS = ",".join(f"{x/10:.1f}" for x in range(1, 21))


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


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


def _render(rows: List[Dict[str, object]], out_pdf: Path, out_png: Path, *, parity_label: str) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(270.0 / 72.0, 185.52 / 72.0), constrained_layout=True)

    families = [
        ("target", "Target p*", COLOR_TARGET, "-"),
        ("iqp_parity", parity_label, COLOR_IQP, "-"),
        ("iqp_mse", "IQP MSE", COLOR_IQP_MSE, "-"),
        ("uniform", "Uniform", COLOR_NEUTRAL, "-."),
    ]

    for key, label, color, ls in families:
        subset = [r for r in rows if str(r["family"]) == key]
        subset = sorted(subset, key=lambda r: float(r["beta"]))
        x = np.asarray([float(r["beta"]) for r in subset], dtype=np.float64)
        y = np.asarray([float(r["KL_pstar_to_q"]) for r in subset], dtype=np.float64)
        ax.plot(x, y, color=color, lw=2.1 if key == "iqp_parity" else 1.9, ls=ls, marker="o", ms=3.6, label=label)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p^\star \parallel q)$")
    ax.set_xlim(0.1, 2.0)
    xticks = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])
    ax.grid(True, ls="--", lw=0.5, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, borderpad=0.25, labelspacing=0.25, handlelength=1.6, handletextpad=0.5)

    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)


def run() -> None:
    ap = argparse.ArgumentParser(description="Experiment 9: single-seed beta sweep in the spirit of panel (c).")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--betas", type=str, default=DEFAULT_BETAS)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--parity-mode", type=str, default="fixed", choices=["fixed", "best_grid"])
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    betas = _parse_float_list(args.betas)
    bits_table = make_bits_table(int(args.n))
    rows: List[Dict[str, object]] = []

    for beta in betas:
        p_star, _support, _scores = build_target_distribution_paper(int(args.n), float(beta))
        idxs_train = sample_indices(p_star, int(args.train_m), seed=int(args.seed))
        emp = empirical_dist(idxs_train, 2 ** int(args.n))
        best_sigma = float(args.sigma)
        best_k = int(args.K)
        best_parity_kl = float("inf")
        best_q_parity = None
        sigma_grid = [float(args.sigma)] if str(args.parity_mode) == "fixed" else [float(x) for x in SIGMA_VALUES]
        k_grid = [int(args.K)] if str(args.parity_mode) == "fixed" else [int(x) for x in K_VALUES]

        for sigma in sigma_grid:
            for kval in k_grid:
                alphas = sample_alphas(int(args.n), float(sigma), int(kval), seed=int(args.seed) + 222)
                P = build_parity_matrix(alphas, bits_table)
                z_data = P @ emp
                q_candidate = train_iqp_qcbm(
                    n=int(args.n),
                    layers=int(args.layers),
                    steps=int(args.steps),
                    lr=float(args.lr),
                    P=P,
                    z_data=z_data,
                    seed_init=int(args.seed) + 10000 + 7 * int(kval),
                )
                kl_candidate = float(forward_kl(p_star, q_candidate))
                if kl_candidate < best_parity_kl:
                    best_parity_kl = kl_candidate
                    best_sigma = float(sigma)
                    best_k = int(kval)
                    best_q_parity = q_candidate

        assert best_q_parity is not None
        q_mse = train_iqp_qcbm_prob_mse(
            n=int(args.n),
            layers=int(args.layers),
            steps=int(args.steps),
            lr=float(args.lr),
            emp_dist=emp,
            seed_init=int(args.seed) + 20000 + 7 * int(args.K),
        )
        q_uniform = np.ones_like(p_star, dtype=np.float64) / float(p_star.size)

        rows.extend(
            [
                {
                    "beta": float(beta),
                    "family": "target",
                    "label": "Target p*",
                    "KL_pstar_to_q": 0.0,
                    "sigma": "",
                    "K": "",
                },
                {
                    "beta": float(beta),
                    "family": "iqp_parity",
                    "label": "Best IQP parity" if str(args.parity_mode) == "best_grid" else "IQP parity",
                    "KL_pstar_to_q": float(best_parity_kl),
                    "sigma": float(best_sigma),
                    "K": int(best_k),
                },
                {
                    "beta": float(beta),
                    "family": "iqp_mse",
                    "label": "IQP MSE",
                    "KL_pstar_to_q": float(forward_kl(p_star, q_mse)),
                    "sigma": "",
                    "K": "",
                },
                {
                    "beta": float(beta),
                    "family": "uniform",
                    "label": "Uniform",
                    "KL_pstar_to_q": float(forward_kl(p_star, q_uniform)),
                    "sigma": "",
                    "K": "",
                },
            ]
        )

    suffix = "best_grid" if str(args.parity_mode) == "best_grid" else "fixed"
    csv_path = outdir / f"{OUTPUT_STEM}_{suffix}.csv"
    pdf_path = outdir / f"{OUTPUT_STEM}_{suffix}.pdf"
    png_path = outdir / f"{OUTPUT_STEM}_{suffix}.png"
    _write_csv(csv_path, rows)
    _render(
        rows,
        pdf_path,
        png_path,
        parity_label="Best IQP parity" if str(args.parity_mode) == "best_grid" else "IQP parity",
    )
    _write_json(
        outdir / f"RUN_CONFIG_{suffix}.json",
        {
            "script": SCRIPT_REL,
            "seed": int(args.seed),
            "betas": [float(x) for x in betas],
            "n": int(args.n),
            "train_m": int(args.train_m),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "layers": int(args.layers),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "parity_mode": str(args.parity_mode),
            "csv": str(csv_path.relative_to(ROOT)),
            "pdf": str(pdf_path.relative_to(ROOT)),
            "png": str(png_path.relative_to(ROOT)),
        },
    )
    print(f"[saved] {pdf_path}")


if __name__ == "__main__":
    run()
