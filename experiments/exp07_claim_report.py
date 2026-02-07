#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 7: Claim validation suite (paper_even + paper_nonparity).

Purpose:
- Run paired sweeps across families/betas/seeds.
- Produce quantitative evidence for the 5 paper claims.
- Emit publication-ready summary tables/plots.

Outputs in <outdir>:
- raw_results.csv
- claim_summary.json
- claim1_qH_vs_Q80.pdf
- claim2_fit_vs_Q80.pdf
- claim3_visibility_vs_qH_spec.pdf
- claim4_family_q80_distribution.pdf
- claim5_budget_law_regression.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy.stats import spearmanr, wilcoxon
except Exception:  # pragma: no cover
    spearmanr = None
    wilcoxon = None

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _q80_cap(x: float, cap: float) -> float:
    x = float(x)
    if not np.isfinite(x):
        return float(cap)
    return float(min(x, cap))


def _safe_log10(x: float) -> float:
    if (not np.isfinite(x)) or x <= 0:
        return float("nan")
    return float(np.log10(x))


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _bootstrap_ci(vals: Sequence[float], n_boot: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    n = arr.size
    for _ in range(max(1, int(n_boot))):
        sample = arr[rng.integers(0, n, size=n)]
        boots.append(float(np.median(sample)))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    med = float(np.median(arr))
    return med, lo, hi


def _spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(xa.size)}
    if spearmanr is None:
        # Fallback: Pearson on ranks
        rx = np.argsort(np.argsort(xa)).astype(np.float64)
        ry = np.argsort(np.argsort(ya)).astype(np.float64)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        return {"rho": rho, "p": float("nan"), "n": int(xa.size)}
    r = spearmanr(xa, ya)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(xa.size)}


def _paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    aa = np.array(a, dtype=np.float64)
    bb = np.array(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 3:
        return {"n": int(aa.size), "stat": float("nan"), "p": float("nan")}
    if wilcoxon is None:
        return {"n": int(aa.size), "stat": float("nan"), "p": float("nan")}
    w = wilcoxon(aa, bb, zero_method="wilcox", alternative="two-sided", method="auto")
    return {"n": int(aa.size), "stat": float(w.statistic), "p": float(w.pvalue)}


def _build_target(family: str, n: int, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if family == "paper_even":
        return hv.build_target_distribution_paper(n, beta)
    if family == "paper_nonparity":
        return hv.build_target_distribution_paper_nonparity(n, beta)
    raise ValueError(f"Unsupported family: {family}")


def _run_suite(args: argparse.Namespace, outdir: str) -> List[Dict]:
    families = _parse_strs(args.families)
    betas = _parse_floats(args.betas)
    seeds = _parse_ints(args.seeds)
    sigmas = _parse_floats(args.sigmas)
    Ks = _parse_ints(args.Ks)

    bits_table = hv.make_bits_table(int(args.n))

    rows: List[Dict] = []

    for family in families:
        for beta in betas:
            p_star, support, scores = _build_target(family, int(args.n), float(beta))
            good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))

            for seed in seeds:
                holdout = hv.select_holdout_smart(
                    p_star=p_star,
                    good_mask=good_mask,
                    bits_table=bits_table,
                    m_train=int(args.train_m),
                    holdout_k=int(args.holdout_k),
                    pool_size=int(args.holdout_pool),
                    seed=int(seed) + 111,
                )

                cfg = hv.Config(
                    n=int(args.n),
                    beta=float(beta),
                    train_m=int(args.train_m),
                    holdout_k=int(args.holdout_k),
                    holdout_pool=int(args.holdout_pool),
                    seed=int(seed),
                    good_frac=float(args.good_frac),
                    sigmas=sigmas,
                    Ks=Ks,
                    Qmax=int(args.Qmax),
                    Q80_thr=float(args.Q80_thr),
                    Q80_search_max=int(args.Q80_search_max),
                    target_family=family,
                    adversarial=False,
                    use_iqp=bool(args.use_iqp),
                    use_classical=bool(args.use_classical),
                    iqp_steps=int(args.iqp_steps),
                    iqp_lr=float(args.iqp_lr),
                    iqp_eval_every=int(args.iqp_eval_every),
                    iqp_layers=int(args.iqp_layers),
                    iqp_loss=str(args.iqp_loss),
                    outdir=outdir,
                )

                res = hv.run_sweep(cfg, p_star, holdout, good_mask, bits_table)
                pH = float(p_star[holdout].sum())
                for r in res:
                    row = dict(r)
                    row.update(
                        {
                            "family": family,
                            "beta": float(beta),
                            "seed": int(seed),
                            "holdout_size": int(np.sum(holdout)),
                            "p_star_holdout": pH,
                        }
                    )
                    rows.append(row)

    return rows


def _analyze(rows: List[Dict], outdir: str, q80_cap: float, n_boot: int, alpha: float) -> Dict[str, object]:
    # Flat arrays for global correlations
    qh_iqp = [_safe_log10(float(r.get("qH_iqp", float("nan")))) for r in rows]
    q80_iqp = [_safe_log10(_q80_cap(float(r.get("Q80_iqp", float("nan"))), q80_cap)) for r in rows]
    loss_iqp = [_safe_log10(float(r.get("train_loss_iqp", float("nan")))) for r in rows]

    qh_spec = [_safe_log10(float(r.get("qH_spec", float("nan")))) for r in rows]
    q80_spec = [_safe_log10(_q80_cap(float(r.get("Q80_spec", float("nan"))), q80_cap)) for r in rows]
    vis = [float(r.get("Vis", float("nan"))) for r in rows]

    qh_class = [_safe_log10(float(r.get("qH_class", float("nan")))) for r in rows]
    q80_class = [_safe_log10(_q80_cap(float(r.get("Q80_class", float("nan"))), q80_cap)) for r in rows]

    claim2_fit_vs_discovery = {
        "iqp_loss_vs_Q80_log": _spearman(loss_iqp, q80_iqp),
        "spec_moment_mse_vs_Q80_spec_log": _spearman(
            [_safe_log10(float(r.get("moment_mse_spec", float("nan")))) for r in rows],
            q80_spec,
        ),
    }

    claim5_budget_law = {
        "iqp_log_qH_vs_log_Q80": _spearman(qh_iqp, q80_iqp),
        "class_log_qH_vs_log_Q80": _spearman(qh_class, q80_class),
        "spec_log_qH_vs_log_Q80": _spearman(qh_spec, q80_spec),
    }

    claim3_mechanism = {
        "vis_vs_qH_spec": _spearman(vis, [float(r.get("qH_spec", float("nan"))) for r in rows]),
        "vis_vs_Q80_spec_log": _spearman(vis, q80_spec),
    }

    # Paired family-wise deltas for robustness (claim 4)
    family_stats: Dict[str, Dict[str, object]] = {}
    for fam in sorted(set(str(r["family"]) for r in rows)):
        fam_rows = [r for r in rows if str(r["family"]) == fam]
        d_iqp_minus_class = []
        d_iqp_minus_spec = []
        vals_iqp = []
        vals_class = []
        vals_spec = []
        for r in fam_rows:
            qi = _q80_cap(float(r.get("Q80_iqp", float("nan"))), q80_cap)
            qc = _q80_cap(float(r.get("Q80_class", float("nan"))), q80_cap)
            qs = _q80_cap(float(r.get("Q80_spec", float("nan"))), q80_cap)
            li = _safe_log10(qi)
            lc = _safe_log10(qc)
            ls = _safe_log10(qs)
            vals_iqp.append(li)
            vals_class.append(lc)
            vals_spec.append(ls)
            if np.isfinite(li) and np.isfinite(lc):
                d_iqp_minus_class.append(li - lc)
            if np.isfinite(li) and np.isfinite(ls):
                d_iqp_minus_spec.append(li - ls)

        med_cls, lo_cls, hi_cls = _bootstrap_ci(d_iqp_minus_class, n_boot=n_boot, alpha=alpha, seed=123)
        med_sp, lo_sp, hi_sp = _bootstrap_ci(d_iqp_minus_spec, n_boot=n_boot, alpha=alpha, seed=456)

        family_stats[fam] = {
            "n_rows": len(fam_rows),
            "wilcoxon_iqp_vs_class_logQ80": _paired_wilcoxon(vals_iqp, vals_class),
            "wilcoxon_iqp_vs_spec_logQ80": _paired_wilcoxon(vals_iqp, vals_spec),
            "delta_logQ80_iqp_minus_class_median": med_cls,
            "delta_logQ80_iqp_minus_class_ci": [lo_cls, hi_cls],
            "delta_logQ80_iqp_minus_spec_median": med_sp,
            "delta_logQ80_iqp_minus_spec_ci": [lo_sp, hi_sp],
            "median_qH_ratio_iqp": float(np.nanmedian([float(r.get("qH_ratio_iqp", np.nan)) for r in fam_rows])),
            "median_qH_ratio_spec": float(np.nanmedian([float(r.get("qH_ratio_spec", np.nan)) for r in fam_rows])),
            "median_qH_ratio_class": float(np.nanmedian([float(r.get("qH_ratio_class", np.nan)) for r in fam_rows])),
        }

    # Plots
    hv.set_style(base=8)

    # Claim 1/5: qH vs Q80
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.8), constrained_layout=True)
    for fam, color in [("paper_even", "#D62728"), ("paper_nonparity", "#1F77B4")]:
        fam_rows = [r for r in rows if str(r["family"]) == fam]
        x = np.array([_safe_log10(float(r.get("qH_iqp", np.nan))) for r in fam_rows], dtype=np.float64)
        y = np.array([_safe_log10(_q80_cap(float(r.get("Q80_iqp", np.nan)), q80_cap)) for r in fam_rows], dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if np.any(m):
            ax.scatter(x[m], y[m], s=20, alpha=0.75, color=color, label=f"{fam} (IQP)")
    ax.set_xlabel(r"$\log_{10} q(H)$")
    ax.set_ylabel(r"$\log_{10} Q_{80}$")
    ax.legend(frameon=False, loc="best")
    fig.savefig(os.path.join(outdir, "claim1_qH_vs_Q80.pdf"))
    plt.close(fig)

    # Claim 2: fit vs discovery
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.8), constrained_layout=True)
    x = np.array(loss_iqp, dtype=np.float64)
    y = np.array(q80_iqp, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.any(m):
        ax.scatter(x[m], y[m], s=20, alpha=0.75, color="#444444")
    ax.set_xlabel(r"$\log_{10}$ IQP train loss")
    ax.set_ylabel(r"$\log_{10} Q_{80}$")
    fig.savefig(os.path.join(outdir, "claim2_fit_vs_Q80.pdf"))
    plt.close(fig)

    # Claim 3: visibility mechanism
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.8), constrained_layout=True)
    x = np.array(vis, dtype=np.float64)
    y = np.array([float(r.get("qH_spec", np.nan)) for r in rows], dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.any(m):
        ax.scatter(x[m], y[m], s=20, alpha=0.75, color="#2ca02c")
    ax.set_xlabel("Visibility Vis_B(H)")
    ax.set_ylabel(r"$q_{spec}(H)$")
    fig.savefig(os.path.join(outdir, "claim3_visibility_vs_qH_spec.pdf"))
    plt.close(fig)

    # Claim 4: family robustness distribution
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.8), constrained_layout=True)
    fams = ["paper_even", "paper_nonparity"]
    vals = []
    for fam in fams:
        fam_rows = [r for r in rows if str(r["family"]) == fam]
        vals.append([
            _safe_log10(_q80_cap(float(r.get("Q80_iqp", np.nan)), q80_cap))
            for r in fam_rows
            if np.isfinite(_safe_log10(_q80_cap(float(r.get("Q80_iqp", np.nan)), q80_cap)))
        ])
    ax.boxplot(vals, tick_labels=fams, showfliers=False)
    ax.set_ylabel(r"$\log_{10} Q_{80}$ (IQP)")
    fig.savefig(os.path.join(outdir, "claim4_family_q80_distribution.pdf"))
    plt.close(fig)

    # Claim 5: budget law regression (all models)
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.8), constrained_layout=True)
    for model, color, xk, yk in [
        ("IQP", "#D62728", "qH_iqp", "Q80_iqp"),
        ("Ising", "#1F77B4", "qH_class", "Q80_class"),
        ("Spec", "#555555", "qH_spec", "Q80_spec"),
    ]:
        x = np.array([_safe_log10(float(r.get(xk, np.nan))) for r in rows], dtype=np.float64)
        y = np.array([_safe_log10(_q80_cap(float(r.get(yk, np.nan)), q80_cap)) for r in rows], dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if np.any(m):
            ax.scatter(x[m], y[m], s=14, alpha=0.5, color=color, label=model)
    ax.set_xlabel(r"$\log_{10} q(H)$")
    ax.set_ylabel(r"$\log_{10} Q_{80}$")
    ax.legend(frameon=False, loc="best")
    fig.savefig(os.path.join(outdir, "claim5_budget_law_regression.pdf"))
    plt.close(fig)

    return {
        "n_rows": len(rows),
        "q80_cap": q80_cap,
        "claim2_fit_vs_discovery": claim2_fit_vs_discovery,
        "claim3_mechanism": claim3_mechanism,
        "claim4_family_stats": family_stats,
        "claim5_budget_law": claim5_budget_law,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="run+analyze", choices=["run+analyze", "analyze"])
    ap.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "exp07_claim_report"))

    ap.add_argument("--families", type=str, default="paper_even,paper_nonparity")
    ap.add_argument("--betas", type=str, default="0.4,0.8,1.2")
    ap.add_argument("--seeds", type=str, default="0,1,2")

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)

    ap.add_argument("--sigmas", type=str, default="0.5,1.0,2.0,3.0")
    ap.add_argument("--Ks", type=str, default="128,256,512")

    ap.add_argument("--Qmax", type=int, default=10000)
    ap.add_argument("--Q80-thr", type=float, default=0.8)
    ap.add_argument("--Q80-search-max", type=int, default=200000)

    ap.add_argument("--use-iqp", type=int, default=1)
    ap.add_argument("--use-classical", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=300)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--iqp-layers", type=int, default=1)
    ap.add_argument("--iqp-loss", type=str, default="parity_mse", choices=["parity_mse", "prob_mse", "xent"])

    ap.add_argument("--q80-cap", type=float, default=200000.0)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)

    args = ap.parse_args()

    outdir = _ensure_outdir(args.outdir)
    raw_csv = os.path.join(outdir, "raw_results.csv")

    if args.mode == "run+analyze":
        rows = _run_suite(args, outdir)
        _write_csv(raw_csv, rows)
    else:
        if not os.path.exists(raw_csv):
            raise FileNotFoundError(f"Missing {raw_csv}; run with --mode run+analyze first.")
        with open(raw_csv, "r", encoding="utf-8") as f:
            rows = []
            for r in csv.DictReader(f):
                parsed: Dict[str, object] = {}
                for k, v in r.items():
                    if v is None:
                        parsed[k] = float("nan")
                        continue
                    s = str(v).strip()

                    if k in ("family", "label", "iqp_loss_mode"):
                        parsed[k] = s
                        continue

                    if s == "":
                        parsed[k] = float("nan")
                        continue
                    sl = s.lower()
                    if sl == "nan":
                        parsed[k] = float("nan")
                        continue
                    if sl == "inf":
                        parsed[k] = float("inf")
                        continue
                    if sl == "-inf":
                        parsed[k] = float("-inf")
                        continue

                    try:
                        parsed[k] = float(s)
                    except ValueError:
                        parsed[k] = s

                rows.append(parsed)

    summary = _analyze(rows, outdir=outdir, q80_cap=float(args.q80_cap), n_boot=int(args.n_bootstrap), alpha=float(args.alpha))

    with open(os.path.join(outdir, "claim_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Claim report written to {outdir}")


if __name__ == "__main__":
    main()
