#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illustrative fixed-beta score-bucket fit: best IQP parity vs IQP MSE.

This is a lightweight illustrative figure, not a matched-seed summary panel.
It uses one representative fixed-beta slice and one fixed training seed, then:

1. trains the IQP parity family over the full sigma-K grid,
2. selects the parity model with the lowest forward KL on that slice,
3. trains the IQP MSE control on the same D_train,
4. visualizes target / best parity / IQP MSE mass over score buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiment_1_kl_diagnostics import (
    K_VALUES,
    SIGMA_VALUES,
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
from final_plot_style import (
    MSE_COLOR,
    PARITY_COLOR,
    TARGET_COLOR,
    TEXT_DARK,
    apply_final_style,
    save_pdf,
)


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_8_fixed_beta_bucket_fit_iqp_vs_mse.py"
OUTPUT_STEM = "experiment_8_fixed_beta_bucket_fit_iqp_vs_mse"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM

FIG_W = 270.0 / 72.0
FIG_H = 195.0 / 72.0


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


def _render_plot(
    *,
    out_pdf: Path,
    beta: float,
    seed: int,
    score_vals: np.ndarray,
    target_masses: np.ndarray,
    parity_masses: np.ndarray,
    mse_masses: np.ndarray,
    best_sigma: float,
    best_k: int,
    parity_kl: float,
    mse_kl: float,
) -> None:
    import matplotlib.pyplot as plt

    apply_final_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    x = np.arange(score_vals.size, dtype=np.float64)
    width = 0.26

    ax.bar(
        x - width,
        target_masses,
        width=width,
        color=TARGET_COLOR,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
        label=r"Target $p^*$",
        zorder=3,
    )
    ax.bar(
        x,
        parity_masses,
        width=width,
        color=PARITY_COLOR,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        label=rf"IQP parity ($KL={parity_kl:.3f}$)",
        zorder=3,
    )
    ax.bar(
        x + width,
        mse_masses,
        width=width,
        color=MSE_COLOR,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.8,
        label=rf"IQP MSE ($KL={mse_kl:.3f}$)",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in score_vals.tolist()])
    ax.set_xlabel(r"Score level $s$")
    ax.set_ylabel(r"$p(\ell = s)$")
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.25, zorder=0)
    ax.grid(False, axis="x")

    ymax = float(max(np.max(target_masses), np.max(parity_masses), np.max(mse_masses)))
    ax.set_ylim(0.0, ymax * 1.24)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.45,
        handletextpad=0.5,
        fontsize=7.4,
    )
    legend.set_zorder(100)

    save_pdf(fig, out_pdf)


def run() -> None:
    ap = argparse.ArgumentParser(description="Illustrative fixed-beta score-bucket fit: best IQP parity vs IQP MSE.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    data_csv = outdir / f"{OUTPUT_STEM}_bucket_masses.csv"
    run_json = outdir / "RUN_CONFIG.json"
    rerender_json = outdir / "RERENDER_CONFIG.json"
    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"

    p_star, support, scores = build_target_distribution_paper(int(args.n), float(args.beta))
    bits_table = make_bits_table(int(args.n))
    idxs_train = sample_indices(p_star, int(args.train_m), seed=int(args.seed))
    emp = empirical_dist(idxs_train, p_star.size)

    best = None
    best_q = None
    for sigma in SIGMA_VALUES:
        for kval in K_VALUES:
            alphas = sample_alphas(int(args.n), float(sigma), int(kval), seed=int(args.seed) + 222)
            P = build_parity_matrix(alphas, bits_table)
            z_data = P @ emp
            q_parity = train_iqp_qcbm(
                n=int(args.n),
                layers=int(args.layers),
                steps=int(args.iqp_steps),
                lr=float(args.iqp_lr),
                P=P,
                z_data=z_data,
                seed_init=int(args.seed) + 10000 + 7 * int(kval),
            )
            kl = float(forward_kl(p_star, q_parity))
            if best is None or kl < float(best["kl"]):
                best = {"sigma": float(sigma), "K": int(kval), "kl": kl}
                best_q = np.asarray(q_parity, dtype=np.float64)

    if best is None or best_q is None:
        raise RuntimeError("Failed to identify a best parity model.")

    q_mse = train_iqp_qcbm_prob_mse(
        n=int(args.n),
        layers=int(args.layers),
        steps=int(args.iqp_steps),
        lr=float(args.iqp_lr),
        emp_dist=emp,
        seed_init=int(args.seed) + 20000 + 7 * 512,
    )
    mse_kl = float(forward_kl(p_star, q_mse))

    score_vals, target_masses = _bucket_masses(p_star, scores, support)
    _, parity_masses = _bucket_masses(best_q, scores, support)
    _, mse_masses = _bucket_masses(q_mse, scores, support)

    rows: List[Dict[str, object]] = []
    for idx, s in enumerate(score_vals.tolist()):
        rows.append(
            {
                "beta": float(args.beta),
                "seed": int(args.seed),
                "n": int(args.n),
                "score_level": int(s),
                "target_mass": float(target_masses[idx]),
                "iqp_parity_mass": float(parity_masses[idx]),
                "iqp_mse_mass": float(mse_masses[idx]),
                "best_sigma": float(best["sigma"]),
                "best_K": int(best["K"]),
                "parity_kl": float(best["kl"]),
                "mse_kl": float(mse_kl),
            }
        )

    _write_csv(data_csv, rows)
    _write_json(
        run_json,
        {
            "script": SCRIPT_REL,
            "outdir": str(outdir.relative_to(ROOT)),
            "beta": float(args.beta),
            "seed": int(args.seed),
            "n": int(args.n),
            "train_m": int(args.train_m),
            "layers": int(args.layers),
            "iqp_steps": int(args.iqp_steps),
            "iqp_lr": float(args.iqp_lr),
            "sigma_values": [float(x) for x in SIGMA_VALUES],
            "k_values": [int(x) for x in K_VALUES],
            "best_sigma": float(best["sigma"]),
            "best_K": int(best["K"]),
            "parity_kl": float(best["kl"]),
            "mse_kl": float(mse_kl),
            "data_csv": str(data_csv.relative_to(ROOT)),
            "pdf": str(out_pdf.relative_to(ROOT)),
        },
    )
    _write_json(
        rerender_json,
        {
            "script": SCRIPT_REL,
            "outdir": str(outdir.relative_to(ROOT)),
            "pdf": out_pdf.name,
            "data_csv": data_csv.name,
            "note": "Rerender currently expects recomputation because the style is coupled to the selected best parity configuration for the slice.",
        },
    )

    _render_plot(
        out_pdf=out_pdf,
        beta=float(args.beta),
        seed=int(args.seed),
        score_vals=score_vals,
        target_masses=target_masses,
        parity_masses=parity_masses,
        mse_masses=mse_masses,
        best_sigma=float(best["sigma"]),
        best_k=int(best["K"]),
        parity_kl=float(best["kl"]),
        mse_kl=float(mse_kl),
    )
    print(f"[experiment8] wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    run()
