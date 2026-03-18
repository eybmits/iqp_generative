#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 6: fixed-beta n-sweep with IQP parity vs classical baselines.

This analysis driver evaluates a fixed-beta scaling slice across a user-provided
set of system sizes `n`, exporting per-seed metrics and compact summary plots.
It now follows the shared active analysis protocol with 10 seeds and a uniform
training budget of 600 for all trained model families.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from experiments.analysis.plot_fig3_kl_bshs_dual_axis_boxplot import _kl_pstar_to_q  # noqa: E402
from experiments.analysis.plot_fig6_beta_sweep_recovery_grid_multiseed import (  # noqa: E402
    HAS_PENNYLANE,
    HAS_TORCH,
    _distribution_fit_metrics,
    _parse_int_list,
    _select_holdout_random,
    _train_classical_boltzmann,
    _train_maxent_parity,
    _train_transformer_autoregressive,
    build_parity_matrix,
    build_target_distribution_paper,
    compute_metrics_for_q,
    empirical_dist,
    make_bits_table,
    sample_alphas,
    sample_indices,
    select_holdout_smart,
    topk_mask_by_scores,
    train_iqp_qcbm,
)
from experiments.final_scripts.plot_appendix_ablation_beta0p8_nsweep import (  # noqa: E402
    MODEL_STYLES,
    _add_legend,
    _draw_panel,
    _reduce_seed_stats,
    apply_final_style,
)
from training_protocol import STANDARD_SEED_IDS_CSV, write_training_protocol


OUTPUT_STEM = "experiment6_fixed_beta_nsweep_all_baselines"
MODEL_ORDER = [
    "iqp_parity",
    "classical_nnn_fields_parity",
    "classical_dense_fields_xent",
    "classical_transformer_mle",
    "classical_maxent_parity",
]
PNG_DPI = 300
FIG_W = 6.95
FIG_H = 2.70
SINGLE_FIG_W = 243.12 / 72.0
SINGLE_FIG_H = 185.52 / 72.0


def _metric_label(model_key: str) -> str:
    return str(MODEL_STYLES.get(model_key, {}).get("label", model_key))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_config(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_readme(
    path: Path,
    *,
    beta: float,
    n_values: np.ndarray,
    seed_values: np.ndarray,
    data_npz: Path,
    metrics_csv: Path,
    summary_csv: Path,
    holdout_mode: str,
    holdout_seed: int,
    train_m: int,
    sigma: float,
    K: int,
    iqp_steps: int,
    artr_epochs: int,
    artr_d_model: int,
    artr_heads: int,
    artr_layers: int,
    artr_ff: int,
    maxent_steps: int,
) -> None:
    try:
        data_npz_rel = str(data_npz.relative_to(ROOT))
    except ValueError:
        data_npz_rel = str(data_npz)
    try:
        metrics_csv_rel = str(metrics_csv.relative_to(ROOT))
    except ValueError:
        metrics_csv_rel = str(metrics_csv)
    try:
        summary_csv_rel = str(summary_csv.relative_to(ROOT))
    except ValueError:
        summary_csv_rel = str(summary_csv)
    lines = [
        "# Experiment 6 Fixed-Beta n-Sweep",
        "",
        "This directory contains the fixed-beta n-sweep scaling study for IQP parity versus the classical baselines.",
        "",
        "Scope:",
        "",
        f"- fixed beta: `{beta:g}`",
        f"- n values: `{','.join(str(int(x)) for x in n_values.tolist())}`",
        f"- seeds: `{','.join(str(int(x)) for x in seed_values.tolist())}`",
        "- model set: `IQP parity`, `Ising+fields (NN+NNN)`, `Dense Ising+fields (xent)`, `AR Transformer (MLE)`, `MaxEnt parity`",
        "- this driver follows the active shared analysis protocol: 10 matched seeds and budget 600",
        "",
        "Fixed protocol:",
        "",
        f"- holdout mode: `{holdout_mode}`",
        f"- holdout seed: `{holdout_seed}`",
        f"- train sample count: `m={train_m}`",
        f"- parity band: `sigma={sigma:g}`, `K={K}`",
        f"- budgets: `iqp_steps={iqp_steps}`, `artr_epochs={artr_epochs}`, `maxent_steps={maxent_steps}`",
        (
            f"- selected Transformer config: `d_model={artr_d_model}`, `layers={artr_layers}`, "
            f"`heads={artr_heads}`, `dim_ff={artr_ff}`"
        ),
        "- training protocol file: `TRAINING_PROTOCOL.md`",
        "",
        "Saved artifacts:",
        "",
        f"- data bundle: `{data_npz_rel}`",
        f"- per-seed metrics: `{metrics_csv_rel}`",
        f"- per-n summary: `{summary_csv_rel}`",
        f"- overview figure: `{OUTPUT_STEM}.pdf` / `{OUTPUT_STEM}.png`",
        f"- KL-only figure: `{OUTPUT_STEM}_kl_vs_n.pdf` / `{OUTPUT_STEM}_kl_vs_n.png`",
        f"- support figures: `{OUTPUT_STEM}_qholdout_vs_n.pdf`, `{OUTPUT_STEM}_q80_vs_n.pdf`",
        "",
        "Primary metrics:",
        "",
        "- `KL_pstar_to_q`: exact forward KL `D_KL(p* || q)`",
        "- `TV`: total variation distance between `p*` and `q`",
        "- `qH`: model mass assigned to the fixed holdout set",
        "- `Q80`: first `Q` such that holdout recovery reaches `0.8`",
        "- `R_Q10000`: holdout recovery at `Q=10000`",
        "",
        "Plot semantics:",
        "",
        "- overview left: mean±std forward KL versus `n`",
        "- overview right: mean±std `R(10000)` versus `n`",
        "- support plots show `q(H)` and `Q80` versus `n` with optional seed points",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metric(
    *,
    out_pdf: Path,
    out_png: Path,
    metric_seed: np.ndarray,
    metric_mean: np.ndarray,
    metric_std: np.ndarray,
    model_keys: List[str],
    n_values: np.ndarray,
    ylabel: str,
    show_seed_points: bool,
    dpi: int,
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    legend_loc: str = "upper right",
) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_FIG_W, SINGLE_FIG_H), constrained_layout=True)
    _draw_panel(
        ax=ax,
        n_values=n_values,
        model_keys=model_keys,
        means=metric_mean,
        stds=metric_std,
        ylabel=ylabel,
        show_seed_points=show_seed_points,
        seed_values=np.arange(metric_seed.shape[2], dtype=np.int64),
        seed_data=metric_seed,
    )
    if yscale == "log":
        finite_vals = metric_seed[np.isfinite(metric_seed) & (metric_seed > 0.0)]
        if finite_vals.size > 0:
            ax.set_yscale("log")
            ax.set_ylim(float(np.min(finite_vals) * 0.8), float(np.max(finite_vals) * 1.2))
    elif ylim is not None:
        ax.set_ylim(*ylim)
    _add_legend(ax, model_keys, loc=legend_loc)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


def _build_run_config(
    *,
    args: argparse.Namespace,
    n_values: np.ndarray,
    seed_values: np.ndarray,
    outdir: Path,
    data_npz: Path,
    metrics_csv: Path,
    summary_csv: Path,
) -> Dict[str, object]:
    return {
        "script": "experiments/analysis/plot_experiment6_fixed_beta_nsweep_all_baselines.py",
        "outdir": str(outdir.relative_to(ROOT)),
        "data_npz": str(data_npz.relative_to(ROOT)),
        "metrics_csv": str(metrics_csv.relative_to(ROOT)),
        "summary_csv": str(summary_csv.relative_to(ROOT)),
        "beta": float(args.beta),
        "n_values": [int(x) for x in n_values.tolist()],
        "seed_values": [int(x) for x in seed_values.tolist()],
        "model_order": [str(x) for x in MODEL_ORDER],
        "holdout_mode": str(args.holdout_mode),
        "holdout_seed": int(args.holdout_seed),
        "holdout_m_train": int(args.holdout_m_train),
        "holdout_k": int(args.holdout_k),
        "holdout_pool": int(args.holdout_pool),
        "train_m": int(args.train_m),
        "sigma": float(args.sigma),
        "K": int(args.K),
        "layers": int(args.layers),
        "good_frac": float(args.good_frac),
        "iqp_steps": int(args.iqp_steps),
        "iqp_lr": float(args.iqp_lr),
        "iqp_eval_every": int(args.iqp_eval_every),
        "artr_epochs": int(args.artr_epochs),
        "artr_d_model": int(args.artr_d_model),
        "artr_heads": int(args.artr_heads),
        "artr_layers": int(args.artr_layers),
        "artr_ff": int(args.artr_ff),
        "artr_lr": float(args.artr_lr),
        "artr_batch_size": int(args.artr_batch_size),
        "maxent_steps": int(args.maxent_steps),
        "maxent_lr": float(args.maxent_lr),
        "q80_thr": float(args.q80_thr),
        "q80_search_max": int(args.q80_search_max),
        "rerun_command": (
            "MPLCONFIGDIR=/tmp/mpl-cache python "
            "experiments/analysis/plot_experiment6_fixed_beta_nsweep_all_baselines.py "
            f"--beta {float(args.beta)} "
            f"--n-values {','.join(str(int(x)) for x in n_values.tolist())} "
            f"--seeds {','.join(str(int(x)) for x in seed_values.tolist())} "
            f"--holdout-mode {str(args.holdout_mode)} "
            f"--holdout-seed {int(args.holdout_seed)} "
            f"--iqp-steps {int(args.iqp_steps)} "
            f"--artr-epochs {int(args.artr_epochs)} "
            f"--maxent-steps {int(args.maxent_steps)} "
            f"--outdir {str(outdir.relative_to(ROOT))}"
        ),
        "note": (
            "Fixed-beta n-sweep using the shared active analysis protocol "
            "(10 seeds, training budget 600)."
        ),
    }


def run() -> None:
    ap = argparse.ArgumentParser(description="Experiment 6 fixed-beta n-sweep with all classical baselines.")
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument(
        "--outdir",
        type=str,
        default="",
        help="If omitted, a beta-tagged directory under outputs/analysis/ is used.",
    )
    ap.add_argument("--data-npz", type=str, default="", help="Optional path to an existing data NPZ.")
    ap.add_argument("--n-values", type=str, default="8,10,12,14,16,18")
    ap.add_argument("--seeds", type=str, default=STANDARD_SEED_IDS_CSV)
    ap.add_argument(
        "--holdout-mode",
        type=str,
        default="high_value",
        choices=["global", "high_value", "random_global", "random_high_value"],
    )
    ap.add_argument("--holdout-seed", type=int, default=46)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=20)
    ap.add_argument("--artr-epochs", type=int, default=600)
    ap.add_argument("--artr-d-model", type=int, default=64)
    ap.add_argument("--artr-heads", type=int, default=4)
    ap.add_argument("--artr-layers", type=int, default=2)
    ap.add_argument("--artr-ff", type=int, default=128)
    ap.add_argument("--artr-lr", type=float, default=1e-3)
    ap.add_argument("--artr-batch-size", type=int, default=256)
    ap.add_argument("--maxent-steps", type=int, default=600)
    ap.add_argument("--maxent-lr", type=float, default=5e-2)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=1000000000)
    ap.add_argument("--show-seed-points", type=int, default=0, choices=[0, 1])
    ap.add_argument("--recompute", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    apply_final_style()
    beta_tag = f"{float(args.beta):.2f}".replace(".", "p")
    outdir = (
        Path(args.outdir)
        if str(args.outdir).strip()
        else ROOT / "outputs" / "analysis" / f"{OUTPUT_STEM}_beta{beta_tag}"
    )
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    data_npz = (
        Path(args.data_npz)
        if str(args.data_npz).strip()
        else outdir / f"{OUTPUT_STEM}_data.npz"
    )
    metrics_csv = outdir / f"{OUTPUT_STEM}_metrics.csv"
    summary_csv = outdir / f"{OUTPUT_STEM}_summary.csv"
    run_config_json = outdir / "RUN_CONFIG.json"
    readme_md = outdir / "README.md"

    if data_npz.exists() and not bool(int(args.recompute)):
        with np.load(data_npz, allow_pickle=True) as z:
            beta = float(z["beta"])
            n_values = np.asarray(z["n_values"], dtype=np.int64)
            seed_values = np.asarray(z["seed_values"], dtype=np.int64)
            model_keys = [str(x) for x in z["model_keys"].tolist()]
            holdout_mode = str(z["holdout_mode"].tolist()[0])
            holdout_seed = int(np.asarray(z["holdout_seed"]).ravel()[0])
            train_m = int(np.asarray(z["train_m"]).ravel()[0])
            sigma = float(np.asarray(z["sigma"]).ravel()[0])
            K = int(np.asarray(z["K"]).ravel()[0])
            iqp_steps = int(np.asarray(z["iqp_steps"]).ravel()[0])
            artr_epochs = int(np.asarray(z["artr_epochs"]).ravel()[0])
            maxent_steps = int(np.asarray(z["maxent_steps"]).ravel()[0])
            kl_seed = np.asarray(z["kl_pstar_to_q_seed"], dtype=np.float64)
            kl_mean = np.asarray(z["kl_pstar_to_q_mean"], dtype=np.float64)
            kl_std = np.asarray(z["kl_pstar_to_q_std"], dtype=np.float64)
            q_holdout_seed = np.asarray(z["q_holdout_seed"], dtype=np.float64)
            q_holdout_mean = np.asarray(z["q_holdout_mean"], dtype=np.float64)
            q_holdout_std = np.asarray(z["q_holdout_std"], dtype=np.float64)
            q80_seed = np.asarray(z["q80_seed"], dtype=np.float64)
            q80_mean = np.asarray(z["q80_mean"], dtype=np.float64)
            q80_std = np.asarray(z["q80_std"], dtype=np.float64)
            r_q10000_seed = np.asarray(z["r_q10000_seed"], dtype=np.float64)
            r_q10000_mean = np.asarray(z["r_q10000_mean"], dtype=np.float64)
            r_q10000_std = np.asarray(z["r_q10000_std"], dtype=np.float64)
    else:
        beta = float(args.beta)
        n_values = _parse_int_list(str(args.n_values))
        seed_values = _parse_int_list(str(args.seeds))
        model_keys = [str(x) for x in MODEL_ORDER]
        holdout_mode = str(args.holdout_mode)
        holdout_seed = int(args.holdout_seed)
        train_m = int(args.train_m)
        sigma = float(args.sigma)
        K = int(args.K)
        iqp_steps = int(args.iqp_steps)
        artr_epochs = int(args.artr_epochs)
        maxent_steps = int(args.maxent_steps)
        bits_tables = {int(n): make_bits_table(int(n)) for n in n_values.tolist()}

        kl_seed = np.full((len(model_keys), n_values.size, seed_values.size), np.nan, dtype=np.float64)
        tv_seed = np.full_like(kl_seed, np.nan)
        q_holdout_seed = np.full_like(kl_seed, np.nan)
        q80_seed = np.full_like(kl_seed, np.nan)
        r_q10000_seed = np.full_like(kl_seed, np.nan)
        metric_rows: List[Dict[str, object]] = []

        for ni, n in enumerate(n_values.tolist()):
            print(f"[n {ni + 1}/{n_values.size}] n={int(n)} beta={beta:g}")
            bits_table = bits_tables[int(n)]
            p_star, support, scores = build_target_distribution_paper(int(n), beta)
            good_mask = topk_mask_by_scores(scores, support, frac=float(args.good_frac))
            if str(args.holdout_mode) in {"global", "random_global"}:
                holdout_candidate = support.astype(bool)
            else:
                holdout_candidate = good_mask

            if str(args.holdout_mode).startswith("random_"):
                holdout_mask = _select_holdout_random(
                    holdout_candidate,
                    holdout_k=int(args.holdout_k),
                    seed=int(args.holdout_seed) + 111,
                )
            else:
                holdout_mask = select_holdout_smart(
                    p_star=p_star,
                    good_mask=holdout_candidate,
                    bits_table=bits_table,
                    m_train=int(args.holdout_m_train),
                    holdout_k=int(args.holdout_k),
                    pool_size=int(args.holdout_pool),
                    seed=int(args.holdout_seed) + 111,
                )

            p_train = p_star.copy()
            p_train[holdout_mask] = 0.0
            p_train /= float(np.sum(p_train))

            for si, seed in enumerate(seed_values.tolist()):
                print(f"  [seed {si + 1}/{seed_values.size}] seed={int(seed)}")
                idxs_train = sample_indices(p_train, int(args.train_m), seed=int(seed) + 7)
                emp = empirical_dist(idxs_train, p_star.size)
                alphas = sample_alphas(int(n), float(args.sigma), int(args.K), seed=int(seed) + 222)
                P = build_parity_matrix(alphas, bits_table)
                z_data = P @ emp

                q_by_key = {
                    "iqp_parity": train_iqp_qcbm(
                        n=int(n),
                        layers=int(args.layers),
                        steps=int(args.iqp_steps),
                        lr=float(args.iqp_lr),
                        P=P,
                        z_data=z_data,
                        seed_init=int(seed) + 10000 + 7 * int(args.K),
                        eval_every=int(args.iqp_eval_every),
                    ),
                    "classical_nnn_fields_parity": _train_classical_boltzmann(
                        n=int(n),
                        steps=int(args.iqp_steps),
                        lr=float(args.iqp_lr),
                        seed_init=int(seed) + 30001,
                        P=P,
                        z_data=z_data,
                        loss_mode="parity_mse",
                        emp_dist=emp,
                        topology="nn_nnn",
                        include_fields=True,
                    ),
                    "classical_dense_fields_xent": _train_classical_boltzmann(
                        n=int(n),
                        steps=int(args.iqp_steps),
                        lr=float(args.iqp_lr),
                        seed_init=int(seed) + 30004,
                        P=P,
                        z_data=z_data,
                        loss_mode="xent",
                        emp_dist=emp,
                        topology="dense",
                        include_fields=True,
                    ),
                    "classical_transformer_mle": _train_transformer_autoregressive(
                        bits_table=bits_table,
                        idxs_train=idxs_train,
                        n=int(n),
                        seed=int(seed) + 35501,
                        epochs=int(args.artr_epochs),
                        d_model=int(args.artr_d_model),
                        nhead=int(args.artr_heads),
                        num_layers=int(args.artr_layers),
                        dim_ff=int(args.artr_ff),
                        lr=float(args.artr_lr),
                        batch_size=int(args.artr_batch_size),
                    ),
                    "classical_maxent_parity": _train_maxent_parity(
                        P=P,
                        z_data=z_data,
                        seed=int(seed) + 36001,
                        steps=int(args.maxent_steps),
                        lr=float(args.maxent_lr),
                    ),
                }

                for mi, model_key in enumerate(model_keys):
                    q_model = np.asarray(q_by_key[model_key], dtype=np.float64)
                    kl_val = float(_kl_pstar_to_q(p_star=p_star, q=q_model))
                    tv_val = float(_distribution_fit_metrics(q=q_model, p_star=p_star)["tv"])
                    met = compute_metrics_for_q(
                        q_model,
                        holdout_mask,
                        float(args.q80_thr),
                        int(args.q80_search_max),
                    )
                    kl_seed[mi, ni, si] = kl_val
                    tv_seed[mi, ni, si] = tv_val
                    q_holdout_seed[mi, ni, si] = float(met["qH"])
                    q80_seed[mi, ni, si] = float(met["Q80"])
                    r_q10000_seed[mi, ni, si] = float(met["R_Q10000"])
                    metric_rows.append(
                        {
                            "beta": beta,
                            "n": int(n),
                            "seed": int(seed),
                            "model_key": str(model_key),
                            "model_label": _metric_label(model_key),
                            "holdout_mode": str(args.holdout_mode),
                            "holdout_seed": int(args.holdout_seed),
                            "holdout_m_train": int(args.holdout_m_train),
                            "holdout_k": int(args.holdout_k),
                            "holdout_pool": int(args.holdout_pool),
                            "train_m": int(args.train_m),
                            "sigma": float(args.sigma),
                            "K": int(args.K),
                            "layers": int(args.layers),
                            "iqp_steps": int(args.iqp_steps),
                            "artr_epochs": int(args.artr_epochs),
                            "maxent_steps": int(args.maxent_steps),
                            "KL_pstar_to_q": kl_val,
                            "TV": tv_val,
                            "qH": float(met["qH"]),
                            "Q80": float(met["Q80"]),
                            "R_Q10000": float(met["R_Q10000"]),
                        }
                    )

        kl_mean, kl_std = _reduce_seed_stats(kl_seed)
        tv_mean, tv_std = _reduce_seed_stats(tv_seed)
        q_holdout_mean, q_holdout_std = _reduce_seed_stats(q_holdout_seed)
        q80_mean, q80_std = _reduce_seed_stats(q80_seed)
        r_q10000_mean, r_q10000_std = _reduce_seed_stats(r_q10000_seed)

        np.savez(
            data_npz,
            beta=np.asarray(beta, dtype=np.float64),
            n_values=np.asarray(n_values, dtype=np.int64),
            seed_values=np.asarray(seed_values, dtype=np.int64),
            model_keys=np.asarray(model_keys, dtype=object),
            holdout_mode=np.asarray([str(args.holdout_mode)], dtype=object),
            holdout_seed=np.asarray([int(args.holdout_seed)], dtype=np.int64),
            holdout_m_train=np.asarray([int(args.holdout_m_train)], dtype=np.int64),
            holdout_k=np.asarray([int(args.holdout_k)], dtype=np.int64),
            holdout_pool=np.asarray([int(args.holdout_pool)], dtype=np.int64),
            train_m=np.asarray([int(args.train_m)], dtype=np.int64),
            sigma=np.asarray([float(args.sigma)], dtype=np.float64),
            K=np.asarray([int(args.K)], dtype=np.int64),
            layers=np.asarray([int(args.layers)], dtype=np.int64),
            good_frac=np.asarray([float(args.good_frac)], dtype=np.float64),
            iqp_steps=np.asarray([int(args.iqp_steps)], dtype=np.int64),
            artr_epochs=np.asarray([int(args.artr_epochs)], dtype=np.int64),
            maxent_steps=np.asarray([int(args.maxent_steps)], dtype=np.int64),
            q80_thr=np.asarray([float(args.q80_thr)], dtype=np.float64),
            q80_search_max=np.asarray([int(args.q80_search_max)], dtype=np.int64),
            eval_mode_by_n=np.asarray(["exact"] * int(n_values.size), dtype=object),
            shots_budget=np.asarray([0], dtype=np.int64),
            kl_pstar_to_q_seed=kl_seed,
            kl_pstar_to_q_mean=kl_mean,
            kl_pstar_to_q_std=kl_std,
            tv_seed=tv_seed,
            tv_mean=tv_mean,
            tv_std=tv_std,
            q_holdout_seed=q_holdout_seed,
            q_holdout_mean=q_holdout_mean,
            q_holdout_std=q_holdout_std,
            q80_seed=q80_seed,
            q80_mean=q80_mean,
            q80_std=q80_std,
            r_q10000_seed=r_q10000_seed,
            r_q10000_mean=r_q10000_mean,
            r_q10000_std=r_q10000_std,
        )
        _write_csv(metrics_csv, metric_rows)

        summary_rows: List[Dict[str, object]] = []
        for mi, model_key in enumerate(model_keys):
            for ni, n in enumerate(n_values.tolist()):
                summary_rows.append(
                    {
                        "beta": beta,
                        "n": int(n),
                        "model_key": str(model_key),
                        "model_label": _metric_label(model_key),
                        "seeds_n": int(seed_values.size),
                        "KL_pstar_to_q_mean": float(kl_mean[mi, ni]),
                        "KL_pstar_to_q_std": float(kl_std[mi, ni]),
                        "TV_mean": float(tv_mean[mi, ni]),
                        "TV_std": float(tv_std[mi, ni]),
                        "qH_mean": float(q_holdout_mean[mi, ni]),
                        "qH_std": float(q_holdout_std[mi, ni]),
                        "Q80_mean": float(q80_mean[mi, ni]),
                        "Q80_std": float(q80_std[mi, ni]),
                        "R_Q10000_mean": float(r_q10000_mean[mi, ni]),
                        "R_Q10000_std": float(r_q10000_std[mi, ni]),
                    }
                )
        _write_csv(summary_csv, summary_rows)

    overview_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    overview_png = outdir / f"{OUTPUT_STEM}.png"
    kl_pdf = outdir / f"{OUTPUT_STEM}_kl_vs_n.pdf"
    kl_png = outdir / f"{OUTPUT_STEM}_kl_vs_n.png"
    qholdout_pdf = outdir / f"{OUTPUT_STEM}_qholdout_vs_n.pdf"
    qholdout_png = outdir / f"{OUTPUT_STEM}_qholdout_vs_n.png"
    q80_pdf = outdir / f"{OUTPUT_STEM}_q80_vs_n.pdf"
    q80_png = outdir / f"{OUTPUT_STEM}_q80_vs_n.png"

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), constrained_layout=True)
    ax_kl, ax_r = axes
    _draw_panel(
        ax=ax_kl,
        n_values=n_values,
        model_keys=model_keys,
        means=kl_mean,
        stds=kl_std,
        ylabel=r"Forward KL $D_{KL}(p^* \parallel q)$",
        show_seed_points=bool(int(args.show_seed_points)),
        seed_values=seed_values,
        seed_data=kl_seed,
    )
    _draw_panel(
        ax=ax_r,
        n_values=n_values,
        model_keys=model_keys,
        means=r_q10000_mean,
        stds=r_q10000_std,
        ylabel=r"Recovery $R(10000)$",
        show_seed_points=bool(int(args.show_seed_points)),
        seed_values=seed_values,
        seed_data=r_q10000_seed,
    )
    ax_kl.set_ylim(bottom=0.0)
    ax_r.set_ylim(0.0, 1.02)
    _add_legend(ax_kl, model_keys, loc="upper right")
    fig.text(
        0.5,
        0.01,
            (
            rf"$\beta={beta:g}$, mean$\pm$std over {seed_values.size} seeds; "
            rf"holdout={holdout_mode}, $\sigma={sigma:g}$, $K={K}$"
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.savefig(overview_pdf)
    fig.savefig(overview_png, dpi=int(args.dpi))
    plt.close(fig)

    _plot_metric(
        out_pdf=kl_pdf,
        out_png=kl_png,
        metric_seed=kl_seed,
        metric_mean=kl_mean,
        metric_std=kl_std,
        model_keys=model_keys,
        n_values=n_values,
        ylabel=r"Forward KL $D_{KL}(p^* \parallel q)$",
        show_seed_points=bool(int(args.show_seed_points)),
        dpi=int(args.dpi),
        ylim=(0.0, max(0.05, float(np.nanmax(kl_seed) * 1.10))),
        legend_loc="upper right",
    )

    _plot_metric(
        out_pdf=qholdout_pdf,
        out_png=qholdout_png,
        metric_seed=q_holdout_seed,
        metric_mean=q_holdout_mean,
        metric_std=q_holdout_std,
        model_keys=model_keys,
        n_values=n_values,
        ylabel=r"Holdout mass $q(H)$",
        show_seed_points=bool(int(args.show_seed_points)),
        dpi=int(args.dpi),
        ylim=(0.0, max(0.02, float(np.nanmax(q_holdout_seed) * 1.15))),
        legend_loc="upper right",
    )
    _plot_metric(
        out_pdf=q80_pdf,
        out_png=q80_png,
        metric_seed=q80_seed,
        metric_mean=q80_mean,
        metric_std=q80_std,
        model_keys=model_keys,
        n_values=n_values,
        ylabel=r"$Q_{80}$ (lower better)",
        show_seed_points=bool(int(args.show_seed_points)),
        dpi=int(args.dpi),
        yscale="log",
        legend_loc="upper left",
    )

    run_config = _build_run_config(
        args=args,
        n_values=n_values,
        seed_values=seed_values,
        outdir=outdir,
        data_npz=data_npz,
        metrics_csv=metrics_csv,
        summary_csv=summary_csv,
    )
    _write_run_config(run_config_json, run_config)
    _write_readme(
        readme_md,
        beta=beta,
        n_values=n_values,
        seed_values=seed_values,
        data_npz=data_npz,
        metrics_csv=metrics_csv,
        summary_csv=summary_csv,
        holdout_mode=holdout_mode,
        holdout_seed=holdout_seed,
        train_m=train_m,
        sigma=sigma,
        K=K,
        iqp_steps=iqp_steps,
        artr_epochs=artr_epochs,
        artr_d_model=int(args.artr_d_model),
        artr_heads=int(args.artr_heads),
        artr_layers=int(args.artr_layers),
        artr_ff=int(args.artr_ff),
        maxent_steps=maxent_steps,
    )
    write_training_protocol(
        outdir,
        experiment_name="Experiment 6 fixed-beta n-sweep",
        note="This scaling experiment uses the shared 10-seed / 600-budget analysis standard.",
        source_relpath="experiments/analysis/plot_experiment6_fixed_beta_nsweep_all_baselines.py",
        metrics_note="The primary paper-facing output for this study is exact forward KL versus n.",
    )

    print(f"[saved] {overview_pdf}")
    print(f"[saved] {overview_png}")
    print(f"[saved] {kl_pdf}")
    print(f"[saved] {kl_png}")
    print(f"[saved] {qholdout_pdf}")
    print(f"[saved] {qholdout_png}")
    print(f"[saved] {q80_pdf}")
    print(f"[saved] {q80_png}")
    print(f"[saved] {data_npz}")
    print(f"[saved] {metrics_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {run_config_json}")
    print(f"[saved] {readme_md}")
    print(f"[saved] {outdir / 'TRAINING_PROTOCOL.md'}")


if __name__ == "__main__":
    run()
