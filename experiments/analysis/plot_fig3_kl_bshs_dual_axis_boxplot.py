#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute a Fig3-style BSHS vs forward-KL dual-axis boxplot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.plot_fig6_beta_sweep_recovery_grid_multiseed import (  # noqa: E402
    BENCHMARK_MATCHED_INSTANCE_SEED_IDS,
    HAS_PENNYLANE,
    HAS_TORCH,
    _train_classical_boltzmann,
    _train_maxent_parity,
    _train_transformer_autoregressive,
    benchmark_protocol_metadata,
    build_parity_matrix,
    build_target_distribution_paper,
    empirical_dist,
    expected_unique_fraction,
    make_bits_table,
    sample_alphas,
    sample_indices,
    select_holdout_smart,
    topk_mask_by_scores,
    train_iqp_qcbm,
)
from paper_benchmark_ledger import record_benchmark_run  # noqa: E402
from experiments.analysis.training_protocol import STANDARD_SEED_IDS_CSV, write_training_protocol  # noqa: E402
from experiments.final_scripts.plot_tv_bshs_seedmean_scatter import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
    _render_dual_axis_boxplot,
    apply_final_style,
)


def _score_levels(scores: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    s_int = scores.astype(np.int64)
    return np.sort(np.unique(s_int[support_mask]))


def _mass_by_level(probs: np.ndarray, scores: np.ndarray, levels: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    s_int = scores.astype(np.int64)
    out = np.zeros(levels.shape[0], dtype=np.float64)
    for i, lv in enumerate(levels):
        mask = support_mask & (s_int == int(lv))
        out[i] = float(np.sum(probs[mask]))
    return out


def _normalized(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    s = float(np.sum(v))
    if s <= eps:
        return np.zeros_like(v, dtype=np.float64)
    return v / s


def _bucket_hit_curve_component(q: np.ndarray, level_mask: np.ndarray, q_eval: int) -> float:
    y = expected_unique_fraction(q, level_mask, np.array([int(q_eval)], dtype=np.int64))
    return float(y[0]) if y.size else 0.0


def compute_support_bucket_metrics(
    p_star: np.ndarray,
    q: np.ndarray,
    support_mask: np.ndarray,
    scores: np.ndarray,
    q_eval: int,
) -> Dict[str, float]:
    levels = _score_levels(scores, support_mask)
    target_mass = _mass_by_level(p_star, scores, levels, support_mask)
    model_mass = _mass_by_level(q, scores, levels, support_mask)

    target_share = _normalized(target_mass)
    model_share = _normalized(model_mass)
    tv_score = 0.5 * float(np.sum(np.abs(target_share - model_share)))

    s_int = scores.astype(np.int64)
    bucket_hit = np.zeros(levels.shape[0], dtype=np.float64)
    for i, lv in enumerate(levels):
        level_mask = support_mask & (s_int == int(lv))
        bucket_hit[i] = _bucket_hit_curve_component(q, level_mask, q_eval=q_eval)

    bshs = float(np.sum(target_share * bucket_hit))
    composite = float(bshs * (1.0 - tv_score))
    return {"TV_score": tv_score, "BSHS": bshs, "Composite": composite}


def _kl_pstar_to_q(p_star: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    pv = np.asarray(p_star, dtype=np.float64)
    qv = np.asarray(q, dtype=np.float64)
    pv = np.clip(pv, eps, 1.0)
    qv = np.clip(qv, eps, 1.0)
    pv = pv / float(np.sum(pv))
    qv = qv / float(np.sum(qv))
    return float(np.sum(pv * np.log(pv / qv)))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_summary(path: Path, rows: List[Dict[str, object]], config: Dict[str, object]) -> None:
    df = pd.DataFrame(rows)
    df_mean = (
        df.groupby(["model_key", "model_label", "beta"], as_index=False)
        .agg(
            seeds_n=("seed", "nunique"),
            KL_pstar_to_q_mean=("KL_pstar_to_q", "mean"),
            KL_pstar_to_q_std=("KL_pstar_to_q", "std"),
            BSHS_mean=("BSHS", "mean"),
            BSHS_std=("BSHS", "std"),
        )
        .sort_values(["model_key", "beta"], ascending=[True, True])
        .reset_index(drop=True)
    )
    best = df_mean.sort_values(["KL_pstar_to_q_mean", "BSHS_mean"], ascending=[True, False]).iloc[0].to_dict()
    payload = {
        "config": config,
        "best_model_by_mean_forward_kl": best,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_run_config(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def run() -> None:
    ap = argparse.ArgumentParser(description="Recompute a Fig3-style BSHS vs forward-KL boxplot.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "analysis" / "fig3_kl_bshs_seedmean_scatter_10seeds_all600"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument(
        "--seeds",
        type=str,
        default=STANDARD_SEED_IDS_CSV,
        help="Comma-separated matched-instance seed IDs. The active analysis standard uses 10 seeds: 101..110.",
    )
    ap.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--q-eval", type=int, default=1000)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--artr-epochs", type=int, default=600)
    ap.add_argument("--artr-d-model", type=int, default=64)
    ap.add_argument("--artr-heads", type=int, default=4)
    ap.add_argument("--artr-layers", type=int, default=2)
    ap.add_argument("--artr-ff", type=int, default=128)
    ap.add_argument("--artr-lr", type=float, default=1e-3)
    ap.add_argument("--artr-batch-size", type=int, default=256)
    ap.add_argument("--maxent-steps", type=int, default=600)
    ap.add_argument("--maxent-lr", type=float, default=5e-2)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--recompute", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required.")
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")

    apply_final_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    beta_tag = f"{float(args.beta):.2f}".replace(".", "p")
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    points_csv = outdir / f"kl_bshs_points_multiseed_beta_q{int(args.q_eval)}_beta{beta_tag}_newseeds{len(seeds)}.csv"
    summary_json = outdir / f"kl_bshs_summary_multiseed_beta_q{int(args.q_eval)}_beta{beta_tag}_newseeds{len(seeds)}.json"
    out_stem = f"fig3_kl_bshs_seedmean_scatter_beta_{beta_tag}_dual_axis_boxplot"
    out_pdf = outdir / f"{out_stem}.pdf"
    out_png = outdir / f"{out_stem}.png"
    run_config_json = outdir / "RUN_CONFIG.json"

    if points_csv.exists() and not bool(int(args.recompute)):
        df = pd.read_csv(points_csv)
    else:
        bits_table = make_bits_table(int(args.n))
        rows: List[Dict[str, object]] = []

        for run_idx, seed in enumerate(seeds, start=1):
            print(f"[run {run_idx}/{len(seeds)}] beta={float(args.beta):g} seed={seed}")
            p_star, support, scores = build_target_distribution_paper(int(args.n), float(args.beta))
            good_mask = topk_mask_by_scores(scores, support, frac=float(args.good_frac))
            candidate_mask = support.astype(bool) if str(args.holdout_mode) == "global" else good_mask
            holdout_mask = select_holdout_smart(
                p_star=p_star,
                good_mask=candidate_mask,
                bits_table=bits_table,
                m_train=int(args.holdout_m_train),
                holdout_k=int(args.holdout_k),
                pool_size=int(args.holdout_pool),
                seed=int(seed) + 111,
            )

            p_train = p_star.copy()
            p_train[holdout_mask] = 0.0
            p_train /= float(np.sum(p_train))
            idxs_train = sample_indices(p_train, int(args.train_m), seed=int(seed) + 7)
            emp = empirical_dist(idxs_train, p_star.size)
            alphas = sample_alphas(int(args.n), float(args.sigma), int(args.K), seed=int(seed) + 222)
            P = build_parity_matrix(alphas, bits_table)
            z_data = P @ emp

            q_by_key = {
                "iqp_parity_mse": train_iqp_qcbm(
                    n=int(args.n),
                    layers=int(args.layers),
                    steps=int(args.iqp_steps),
                    lr=float(args.iqp_lr),
                    P=P,
                    z_data=z_data,
                    seed_init=int(seed) + 10000 + 7 * int(args.K),
                    eval_every=int(args.iqp_eval_every),
                ),
                "classical_nnn_fields_parity": _train_classical_boltzmann(
                    n=int(args.n),
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
                    n=int(args.n),
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
                    n=int(args.n),
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

            support_mask = p_star > 0.0
            for model_key in MODEL_ORDER:
                q = np.asarray(q_by_key[model_key], dtype=np.float64)
                sb = compute_support_bucket_metrics(
                    p_star=p_star,
                    q=q,
                    support_mask=support_mask,
                    scores=scores,
                    q_eval=int(args.q_eval),
                )
                rows.append(
                    {
                        "model_key": model_key,
                        "model_label": MODEL_LABELS.get(model_key, model_key),
                        "KL_pstar_to_q": float(_kl_pstar_to_q(p_star=p_star, q=q)),
                        "BSHS": float(sb["BSHS"]),
                        "TV_score": float(sb["TV_score"]),
                        "Composite": float(sb["Composite"]),
                        "beta": float(args.beta),
                        "seed": int(seed),
                        "q_eval": int(args.q_eval),
                        "holdout_mode": str(args.holdout_mode),
                        "train_m": int(args.train_m),
                        "sigma": float(args.sigma),
                        "K": int(args.K),
                    }
                )

        _write_csv(points_csv, rows)
        _write_summary(
            summary_json,
            rows,
            config={
                "n": int(args.n),
                "beta": float(args.beta),
                "seeds": seeds,
                "holdout_mode": str(args.holdout_mode),
                "train_m": int(args.train_m),
                "sigma": float(args.sigma),
                "K": int(args.K),
                "q_eval": int(args.q_eval),
                "seed_count": int(len(seeds)),
                "holdout_k": int(args.holdout_k),
                "holdout_pool": int(args.holdout_pool),
                "holdout_m_train": int(args.holdout_m_train),
                "holdout_seed_policy": "matched_seed_plus_111",
                "iqp_steps": int(args.iqp_steps),
                "artr_epochs": int(args.artr_epochs),
                "maxent_steps": int(args.maxent_steps),
                "kl_variant": "forward_kl_pstar_to_q_nats",
                "matches_active_standard_seed_schedule": bool(
                    seeds == [int(x) for x in BENCHMARK_MATCHED_INSTANCE_SEED_IDS]
                ),
                "benchmark_protocol": benchmark_protocol_metadata(
                    betas=[float(args.beta)],
                    K=int(args.K),
                    holdout_policy="matched_seed_plus_111",
                ),
            },
        )
        df = pd.DataFrame(rows)

    _render_dual_axis_boxplot(
        df=df,
        out_stem=out_stem,
        outdir=outdir,
        dpi=int(args.dpi),
        right_metric_col="KL_pstar_to_q",
        right_metric_axis_label=r"$D_{\mathrm{KL}}(p^*\|q)$",
        right_metric_legend_label="KL",
    )

    run_config_payload = {
        "script": "experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py",
        "selected_output": str(out_pdf.relative_to(ROOT)),
        "output_files": [str(out_pdf.relative_to(ROOT)), str(out_png.relative_to(ROOT))],
        "points_csv": str(points_csv.relative_to(ROOT)),
        "summary_json": str(summary_json.relative_to(ROOT)),
        "beta": float(args.beta),
        "n": int(args.n),
        "seeds": [int(x) for x in seeds],
        "q_eval": int(args.q_eval),
        "train_m": int(args.train_m),
        "sigma": float(args.sigma),
        "K": int(args.K),
        "holdout_mode": str(args.holdout_mode),
        "holdout_k": int(args.holdout_k),
        "holdout_pool": int(args.holdout_pool),
        "holdout_m_train": int(args.holdout_m_train),
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
        "right_metric": {
            "name": "forward_kl",
            "formula": "D_KL(p* || q)",
            "units": "nats",
        },
        "models": list(MODEL_ORDER),
    }
    _write_run_config(run_config_json, run_config_payload)
    write_training_protocol(
        outdir,
        experiment_name="Fig3 fixed-beta KL-BSHS benchmark",
        note="This run uses the shared 10-seed / 600-budget analysis standard.",
        source_relpath="experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py",
        metrics_note="The reported right-axis metric is exact forward KL D_KL(p* || q).",
    )

    if seeds == [int(x) for x in BENCHMARK_MATCHED_INSTANCE_SEED_IDS]:
        record_benchmark_run(
            experiment_id="fig3_fixed_beta_kl_bshs_10seed",
            title="Fig4 fixed-beta KL-BSHS benchmark at beta = 0.9",
            run_config_path=run_config_json,
            output_paths=[out_pdf, out_png],
            metrics_paths=[points_csv, summary_json],
            notes=[
                "Fixed-beta 10-seed active-standard artifact for beta = 0.9.",
                "Includes the strong Transformer baseline used in the paper-side disclosure.",
            ],
        )


if __name__ == "__main__":
    run()
