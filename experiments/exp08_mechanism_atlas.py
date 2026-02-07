#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 8: Mechanistic discoverability atlas.

Goal:
- Compare parity_mse vs prob_mse/xent under strictly matched conditions.
- Quantify mass flow to holdout states and mechanistic visibility/alignment links.
- Produce hero + mechanism figures (figA..figJ), CSV artifacts, summary JSON, and report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    pd = None
    _PANDAS_IMPORT_ERROR = e

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


# -----------------------------------------------------------------------------
# Parsing / IO helpers
# -----------------------------------------------------------------------------


def _parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _safe_log10(x: float) -> float:
    if (not np.isfinite(x)) or x <= 0:
        return float("nan")
    return float(np.log10(x))


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------


def _bootstrap_median_ci(vals: Sequence[float], n_boot: int, alpha: float, seed: int) -> Dict[str, float]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.empty(max(1, int(n_boot)), dtype=np.float64)
    for i in range(boots.size):
        sample = arr[rng.integers(0, n, size=n)]
        boots[i] = float(np.median(sample))
    return {
        "median": float(np.median(arr)),
        "lo": float(np.quantile(boots, alpha / 2.0)),
        "hi": float(np.quantile(boots, 1.0 - alpha / 2.0)),
        "n": int(n),
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    m = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[m]
    ya = ya[m]
    if xa.size < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(xa.size)}
    if spearmanr is None:
        rx = np.argsort(np.argsort(xa)).astype(np.float64)
        ry = np.argsort(np.argsort(ya)).astype(np.float64)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        return {"rho": rho, "p": float("nan"), "n": int(xa.size)}
    s = spearmanr(xa, ya)
    return {"rho": float(s.statistic), "p": float(s.pvalue), "n": int(xa.size)}


def _wilcoxon_zero_center(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size < 3:
        return {"stat": float("nan"), "p": float("nan"), "n": int(arr.size)}
    if wilcoxon is None:
        return {"stat": float("nan"), "p": float("nan"), "n": int(arr.size)}
    try:
        w = wilcoxon(arr, zero_method="wilcox", alternative="two-sided", method="auto")
        return {"stat": float(w.statistic), "p": float(w.pvalue), "n": int(arr.size)}
    except Exception:
        return {"stat": float("nan"), "p": float("nan"), "n": int(arr.size)}


# -----------------------------------------------------------------------------
# Core metric helpers
# -----------------------------------------------------------------------------


def _build_target(family: str, n: int, beta: float):
    f = str(family).lower()
    if f == "paper_even":
        return hv.build_target_distribution_paper(n, beta)
    if f == "paper_nonparity":
        return hv.build_target_distribution_paper_nonparity(n, beta)
    if f == "paper":
        return hv.build_target_distribution_paper(n, beta)
    raise ValueError(f"Unsupported family: {family}")


def _fit_metrics(q: np.ndarray, P: np.ndarray, z: np.ndarray, emp: np.ndarray) -> Dict[str, float]:
    q_clip = np.clip(q, 1e-12, 1.0)
    return {
        "fit_parity_mse": float(hv.moment_mse(P, q, z)),
        "fit_prob_mse": float(np.mean((q - emp) ** 2)),
        "fit_xent": float(-np.sum(emp * np.log(q_clip))),
    }


def _alignment_metrics(P: np.ndarray, q: np.ndarray, hat1H: np.ndarray) -> Dict[str, float]:
    z_model = P @ q
    vis = float(np.dot(z_model, hat1H))
    nz = float(np.linalg.norm(z_model))
    nh = float(np.linalg.norm(hat1H))
    if nz <= 0 or nh <= 0:
        cos = float("nan")
    else:
        cos = float(np.dot(z_model, hat1H) / (nz * nh))
    return {"vis_model": vis, "cos_align": cos, "z_model": z_model}


def _pair_mass_flow(q_parity: np.ndarray, q_base: np.ndarray, holdout_mask: np.ndarray) -> Dict[str, float]:
    delta = q_parity - q_base
    dH = delta[holdout_mask]
    flow_h_plus = float(np.sum(np.clip(dH, 0.0, None)))
    net_h = float(np.sum(dH))
    pos_total = float(np.sum(np.clip(delta, 0.0, None)))
    eta = float(flow_h_plus / pos_total) if pos_total > 0 else float("nan")
    return {
        "flow_H_plus": flow_h_plus,
        "net_H": net_h,
        "eta_H_plus": eta,
    }


# -----------------------------------------------------------------------------
# Training chain for figI (representative tuple)
# -----------------------------------------------------------------------------


def _train_iqp_with_chain(
    *,
    n: int,
    layers: int,
    steps: int,
    lr: float,
    P: np.ndarray,
    z_data: np.ndarray,
    seed_init: int,
    eval_every: int,
    loss_mode: str,
    emp: np.ndarray,
    holdout_mask: np.ndarray,
    qH_unif: float,
    H_size: int,
    Q80_thr: float,
    Q80_search_max: int,
    hat1H: np.ndarray,
) -> Tuple[np.ndarray, float, List[Dict]]:
    if not getattr(hv, "HAS_PENNYLANE", False):
        raise RuntimeError("Pennylane required for IQP training chain.")

    qml = hv.qml
    np_pl = hv.np

    pairs = hv.get_iqp_pairs_nn_nnn(n)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        hv.iqp_circuit_zz_only(W, range(n), pairs, layers=layers)
        return qml.probs(wires=range(n))

    loss_mode = str(loss_mode).lower()
    P_t = np_pl.array(P, requires_grad=False)
    z_t = np_pl.array(z_data, requires_grad=False)
    emp_t = np_pl.array(emp, requires_grad=False)
    emp_t = emp_t / np_pl.sum(emp_t)

    rng = np.random.default_rng(seed_init)
    num_params = len(pairs) * layers
    W = np_pl.array(0.01 * rng.standard_normal(num_params), requires_grad=True)

    opt = qml.AdamOptimizer(lr)

    def loss_fn(w):
        q = circuit(w)
        if loss_mode == "parity_mse":
            return np_pl.mean((z_t - P_t @ q) ** 2)
        if loss_mode == "prob_mse":
            return np_pl.mean((q - emp_t) ** 2)
        q_clip = np_pl.clip(q, 1e-12, 1.0)
        return -np_pl.sum(emp_t * np_pl.log(q_clip))

    chain_rows: List[Dict] = []
    last_loss = float("nan")

    eval_steps = set([1, steps])
    for t in range(eval_every, steps + 1, eval_every):
        eval_steps.add(int(t))

    for t in range(1, steps + 1):
        W, loss_val = opt.step_and_cost(loss_fn, W)
        last_loss = float(loss_val)
        if t in eval_steps:
            q_now = np.array(circuit(W), dtype=np.float64)
            q_now = np.clip(q_now, 0.0, None)
            q_now = q_now / max(1e-15, float(q_now.sum()))
            m = hv.compute_metrics_for_q(q_now, holdout_mask, qH_unif, H_size, Q80_thr, Q80_search_max)
            align = _alignment_metrics(P, q_now, hat1H)
            chain_rows.append(
                {
                    "step": int(t),
                    "train_loss": float(last_loss),
                    "qH": float(m["qH"]),
                    "qH_ratio": float(m["qH_ratio"]),
                    "Q80": float(m["Q80"]),
                    "Q80_pred": float(m["Q80_pred"]),
                    "vis_model": float(align["vis_model"]),
                    "cos_align": float(align["cos_align"]),
                }
            )

    q_final = np.array(circuit(W), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, None)
    q_final = q_final / max(1e-15, float(q_final.sum()))
    return q_final, float(last_loss), chain_rows


# -----------------------------------------------------------------------------
# Counterfactual visibility intervention (figJ)
# -----------------------------------------------------------------------------


def _select_counterfactual_holdouts(
    *,
    p_star: np.ndarray,
    good_mask: np.ndarray,
    n: int,
    holdout_k: int,
    seed: int,
    P_ref: np.ndarray,
    z_ref: np.ndarray,
    trials: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    cand = np.where(good_mask)[0]
    if cand.size < holdout_k:
        raise RuntimeError("Not enough candidates for counterfactual holdouts.")

    q_lin_ref = hv.linear_band_reconstruction(P_ref, z_ref, n)

    stats = []
    for _ in range(max(200, int(trials))):
        idx = rng.choice(cand, size=holdout_k, replace=False)
        mask = np.zeros_like(good_mask, dtype=bool)
        mask[idx] = True
        vis = float(hv.visibility_score(P_ref, z_ref, mask, n))
        pH = float(p_star[mask].sum())
        qlinH = float(q_lin_ref[mask].sum())
        q_unif_H = float(holdout_k / (2 ** n))
        ratio = float(qlinH / q_unif_H) if q_unif_H > 0 else float("nan")
        stats.append((mask, vis, pH, ratio))

    stats_sorted = sorted(stats, key=lambda t: t[1])
    head = stats_sorted[: min(40, len(stats_sorted))]
    tail = stats_sorted[-min(40, len(stats_sorted)) :]

    best = None
    for lo in head:
        for hi in tail:
            p_diff = abs(lo[2] - hi[2])
            vis_gap = hi[1] - lo[1]
            key = (p_diff, -vis_gap)
            if best is None or key < best[0]:
                best = (key, lo, hi)

    if best is None:
        lo = stats_sorted[0]
        hi = stats_sorted[-1]
    else:
        lo = best[1]
        hi = best[2]

    lo_info = {"vis_ref": float(lo[1]), "pH": float(lo[2]), "q_lin_ratio": float(lo[3])}
    hi_info = {"vis_ref": float(hi[1]), "pH": float(hi[2]), "q_lin_ratio": float(hi[3])}
    return lo[0], hi[0], lo_info, hi_info


# -----------------------------------------------------------------------------
# Run mode
# -----------------------------------------------------------------------------


def _run_suite(args: argparse.Namespace, outdir: str) -> Dict[str, str]:
    families = _parse_strs(args.families)
    betas = _parse_floats(args.betas)
    seeds = _parse_ints(args.seeds)
    sigmas = _parse_floats(args.sigmas)
    Ks = _parse_ints(args.Ks)
    losses = _parse_strs(args.iqp_losses)

    valid_losses = {"parity_mse", "prob_mse", "xent"}
    bad = [x for x in losses if x not in valid_losses]
    if bad:
        raise ValueError(f"Unsupported losses: {bad}")
    if "parity_mse" not in losses:
        raise ValueError("iqp-losses must include parity_mse for paired comparisons.")

    rep_family = args.rep_family if args.rep_family in families else families[0]
    rep_beta = float(args.rep_beta) if float(args.rep_beta) in betas else float(betas[0])
    rep_seed = int(args.rep_seed) if int(args.rep_seed) in seeds else int(seeds[0])
    rep_sigma = float(args.rep_sigma) if float(args.rep_sigma) in sigmas else float(sigmas[0])
    rep_K = int(args.rep_K) if int(args.rep_K) in Ks else int(Ks[0])
    rep_tuple = (rep_family, rep_beta, rep_seed, rep_sigma, rep_K)

    print(
        f"[RepTuple] family={rep_tuple[0]} beta={rep_tuple[1]} seed={rep_tuple[2]} "
        f"sigma={rep_tuple[3]} K={rep_tuple[4]}"
    )

    bits_table = hv.make_bits_table(int(args.n))

    rows: List[Dict] = []
    pairs: List[Dict] = []
    profile_rows: List[Dict] = []
    contrib_rows: List[Dict] = []
    chain_rows: List[Dict] = []
    counter_rows: List[Dict] = []

    maxK = int(max(Ks))

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

                H_size = int(np.sum(holdout))
                N = p_star.size

                p_train = p_star.copy()
                if H_size > 0:
                    p_train[holdout] = 0.0
                    p_train = p_train / p_train.sum()
                idxs_train = hv.sample_indices(p_train, int(args.train_m), seed=int(seed) + 7)
                emp = hv.empirical_dist(idxs_train, N)

                q_unif = np.ones(N, dtype=np.float64) / N
                qH_unif = float(q_unif[holdout].sum()) if H_size > 0 else 1.0
                p_star_holdout = float(p_star[holdout].sum())

                for si, sigma in enumerate(sigmas):
                    alphas_sup = hv.sample_alphas(int(args.n), float(sigma), maxK, seed=int(seed) + 222 + 13 * si)
                    P_sup = hv.build_parity_matrix(alphas_sup, bits_table)

                    for K in Ks:
                        P = P_sup[: int(K)]
                        z = P @ emp
                        hat1H = hv.indicator_walsh_coeffs(P, holdout, int(args.n))

                        q_spec, q_lin = hv.reconstruct_bandlimited(P, z, int(args.n), return_q_lin=True)
                        assert q_lin is not None
                        spec_metrics = hv.compute_metrics_for_q(
                            q_spec,
                            holdout,
                            qH_unif,
                            H_size,
                            float(args.Q80_thr),
                            int(args.Q80_search_max),
                        )

                        q_by_loss: Dict[str, np.ndarray] = {}
                        row_by_loss: Dict[str, Dict] = {}

                        seed_init = int(seed) + 10000 + 97 * si + 7 * int(K)

                        for loss in losses:
                            is_rep = (family, float(beta), int(seed), float(sigma), int(K)) == rep_tuple

                            if bool(args.make_training_chain) and is_rep:
                                q_iqp, train_loss, chain = _train_iqp_with_chain(
                                    n=int(args.n),
                                    layers=int(args.iqp_layers),
                                    steps=int(args.iqp_steps),
                                    lr=float(args.iqp_lr),
                                    P=P,
                                    z_data=z,
                                    seed_init=seed_init,
                                    eval_every=int(args.iqp_eval_every),
                                    loss_mode=str(loss),
                                    emp=emp,
                                    holdout_mask=holdout,
                                    qH_unif=qH_unif,
                                    H_size=H_size,
                                    Q80_thr=float(args.Q80_thr),
                                    Q80_search_max=int(args.Q80_search_max),
                                    hat1H=hat1H,
                                )
                                for ch in chain:
                                    ch_row = dict(ch)
                                    ch_row.update(
                                        {
                                            "family": family,
                                            "beta": float(beta),
                                            "seed": int(seed),
                                            "sigma": float(sigma),
                                            "K": int(K),
                                            "loss": str(loss),
                                        }
                                    )
                                    chain_rows.append(ch_row)
                            else:
                                q_iqp, train_loss, _ = hv.train_iqp_qcbm(
                                    n=int(args.n),
                                    layers=int(args.iqp_layers),
                                    steps=int(args.iqp_steps),
                                    lr=float(args.iqp_lr),
                                    P=P,
                                    z_data=z,
                                    seed_init=seed_init,
                                    eval_every=int(args.iqp_eval_every),
                                    return_hist=False,
                                    loss_mode=str(loss),
                                    xent_emp=emp,
                                )

                            q_by_loss[str(loss)] = q_iqp

                            m = hv.compute_metrics_for_q(
                                q_iqp,
                                holdout,
                                qH_unif,
                                H_size,
                                float(args.Q80_thr),
                                int(args.Q80_search_max),
                            )
                            fits = _fit_metrics(q_iqp, P, z, emp)
                            align = _alignment_metrics(P, q_iqp, hat1H)

                            row = {
                                "family": family,
                                "beta": float(beta),
                                "seed": int(seed),
                                "sigma": float(sigma),
                                "K": int(K),
                                "loss": str(loss),
                                "holdout_size": H_size,
                                "p_star_holdout": p_star_holdout,
                                "qH": float(m["qH"]),
                                "qH_ratio": float(m["qH_ratio"]),
                                "R_Q1000": float(m["R_Q1000"]),
                                "R_Q10000": float(m["R_Q10000"]),
                                "Q80": float(m["Q80"]),
                                "Q80_pred": float(m["Q80_pred"]),
                                "Q80_lb": float(m["Q80_lb"]),
                                "train_loss": float(train_loss),
                                "vis_model": float(align["vis_model"]),
                                "cos_align": float(align["cos_align"]),
                                "fit_parity_mse": float(fits["fit_parity_mse"]),
                                "fit_prob_mse": float(fits["fit_prob_mse"]),
                                "fit_xent": float(fits["fit_xent"]),
                                "q_lin_H": float(np.sum(q_lin[holdout])) if H_size > 0 else 0.0,
                                "q_spec_H": float(spec_metrics["qH"]),
                                "Q80_spec": float(spec_metrics["Q80"]),
                                "vis_data": float(hv.visibility_score(P, z, holdout, int(args.n))),
                            }

                            rows.append(row)
                            row_by_loss[str(loss)] = row

                            if is_rep:
                                z_model = align["z_model"]
                                c = z_model * hat1H
                                order = np.argsort(-np.abs(c))
                                c_sorted = c[order]
                                c_cum = np.cumsum(c_sorted)
                                for rk in range(c_sorted.size):
                                    contrib_rows.append(
                                        {
                                            "family": family,
                                            "beta": float(beta),
                                            "seed": int(seed),
                                            "sigma": float(sigma),
                                            "K": int(K),
                                            "loss": str(loss),
                                            "rank": int(rk + 1),
                                            "c_signed": float(c_sorted[rk]),
                                            "cum_signed": float(c_cum[rk]),
                                        }
                                    )

                        # Build pairs against each base loss
                        for base_loss in ("prob_mse", "xent"):
                            if "parity_mse" not in q_by_loss or base_loss not in q_by_loss:
                                continue

                            q_par = q_by_loss["parity_mse"]
                            q_base = q_by_loss[base_loss]
                            r_par = row_by_loss["parity_mse"]
                            r_base = row_by_loss[base_loss]

                            q80_par = float(r_par["Q80"])
                            q80_base = float(r_base["Q80"])
                            finite_pair = bool(np.isfinite(q80_par) and np.isfinite(q80_base) and q80_base > 0)
                            q80_ratio = float(q80_par / q80_base) if finite_pair else float("nan")
                            dlog = float(_safe_log10(q80_par) - _safe_log10(q80_base)) if finite_pair else float("nan")

                            flow = _pair_mass_flow(q_par, q_base, holdout)

                            pair = {
                                "family": family,
                                "beta": float(beta),
                                "seed": int(seed),
                                "sigma": float(sigma),
                                "K": int(K),
                                "base_loss": base_loss,
                                "delta_qH": float(r_par["qH"] - r_base["qH"]),
                                "delta_qH_ratio": float(r_par["qH_ratio"] - r_base["qH_ratio"]),
                                "Q80_parity": q80_par,
                                "Q80_base": q80_base,
                                "Q80_ratio": q80_ratio,
                                "delta_logQ80": dlog,
                                "finite_q80_pair": int(finite_pair),
                                "parity_q80_inf": int(not np.isfinite(q80_par)),
                                "base_q80_inf": int(not np.isfinite(q80_base)),
                                "flow_H_plus": float(flow["flow_H_plus"]),
                                "net_H": float(flow["net_H"]),
                                "eta_H_plus": float(flow["eta_H_plus"]),
                                "qH_parity": float(r_par["qH"]),
                                "qH_base": float(r_base["qH"]),
                                "qH_ratio_parity": float(r_par["qH_ratio"]),
                                "qH_ratio_base": float(r_base["qH_ratio"]),
                                "vis_parity": float(r_par["vis_model"]),
                                "vis_base": float(r_base["vis_model"]),
                                "cos_align_parity": float(r_par["cos_align"]),
                                "cos_align_base": float(r_base["cos_align"]),
                            }
                            pairs.append(pair)

                            # Representative holdout profiles for figD/figE
                            is_rep = (family, float(beta), int(seed), float(sigma), int(K)) == rep_tuple
                            if is_rep:
                                idxH = np.where(holdout)[0]
                                sort_idx = idxH[np.argsort(-q_par[idxH])]
                                cum_par = np.cumsum(q_par[sort_idx])
                                cum_base = np.cumsum(q_base[sort_idx])
                                for rnk, idx in enumerate(sort_idx, start=1):
                                    profile_rows.append(
                                        {
                                            "family": family,
                                            "beta": float(beta),
                                            "seed": int(seed),
                                            "sigma": float(sigma),
                                            "K": int(K),
                                            "base_loss": base_loss,
                                            "rank": int(rnk),
                                            "state_index": int(idx),
                                            "bitstring": hv.bits_str(bits_table[int(idx)]),
                                            "q_parity": float(q_par[int(idx)]),
                                            "q_base": float(q_base[int(idx)]),
                                            "delta": float(q_par[int(idx)] - q_base[int(idx)]),
                                            "cum_parity": float(cum_par[rnk - 1]),
                                            "cum_base": float(cum_base[rnk - 1]),
                                        }
                                    )

    # Counterfactual intervention (figJ)
    if bool(args.make_counterfactual):
        cf_family = args.counter_family if args.counter_family in families else families[0]
        cf_beta = float(args.counter_beta) if float(args.counter_beta) in betas else float(betas[0])
        cf_seed = int(args.counter_seed)

        p_star, support, scores = _build_target(cf_family, int(args.n), cf_beta)
        good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))

        # Reference band for visibility discrimination
        K_ref = int(args.counter_K)
        sigma_ref = float(args.counter_sigma)
        alphas_ref = hv.sample_alphas(int(args.n), sigma_ref, K_ref, seed=cf_seed + 991)
        P_ref = hv.build_parity_matrix(alphas_ref, bits_table)
        z_ref = P_ref @ p_star

        lo_mask, hi_mask, lo_info, hi_info = _select_counterfactual_holdouts(
            p_star=p_star,
            good_mask=good_mask,
            n=int(args.n),
            holdout_k=int(args.holdout_k),
            seed=cf_seed + 777,
            P_ref=P_ref,
            z_ref=z_ref,
            trials=int(args.counter_trials),
        )

        for group, hmask, info, offset in [
            ("low_visibility", lo_mask, lo_info, 0),
            ("high_visibility", hi_mask, hi_info, 5000),
        ]:
            N = p_star.size
            H_size = int(np.sum(hmask))
            p_train = p_star.copy()
            p_train[hmask] = 0.0
            p_train = p_train / p_train.sum()
            idxs_train = hv.sample_indices(p_train, int(args.train_m), seed=cf_seed + 7 + offset)
            emp = hv.empirical_dist(idxs_train, N)
            q_unif = np.ones(N, dtype=np.float64) / N
            qH_unif = float(q_unif[hmask].sum()) if H_size > 0 else 1.0

            alphas = hv.sample_alphas(int(args.n), float(args.counter_sigma), int(args.counter_K), seed=cf_seed + 333 + offset)
            P = hv.build_parity_matrix(alphas, bits_table)
            z = P @ emp

            q_iqp, train_loss, _ = hv.train_iqp_qcbm(
                n=int(args.n),
                layers=int(args.iqp_layers),
                steps=int(args.iqp_steps),
                lr=float(args.iqp_lr),
                P=P,
                z_data=z,
                seed_init=cf_seed + 10000 + offset,
                eval_every=int(args.iqp_eval_every),
                return_hist=False,
                loss_mode="parity_mse",
                xent_emp=emp,
            )

            m = hv.compute_metrics_for_q(
                q_iqp,
                hmask,
                qH_unif,
                H_size,
                float(args.Q80_thr),
                int(args.Q80_search_max),
            )
            counter_rows.append(
                {
                    "group": group,
                    "family": cf_family,
                    "beta": cf_beta,
                    "seed": int(cf_seed),
                    "sigma": float(args.counter_sigma),
                    "K": int(args.counter_K),
                    "holdout_size": int(H_size),
                    "p_star_holdout": float(info["pH"]),
                    "vis_ref": float(info["vis_ref"]),
                    "q_lin_ratio": float(info["q_lin_ratio"]),
                    "qH_model": float(m["qH"]),
                    "qH_ratio_model": float(m["qH_ratio"]),
                    "Q80_model": float(m["Q80"]),
                    "Q80_pred_model": float(m["Q80_pred"]),
                    "train_loss": float(train_loss),
                }
            )

    paths = {
        "rows": os.path.join(outdir, "atlas_rows.csv"),
        "pairs": os.path.join(outdir, "atlas_pairs.csv"),
        "profile": os.path.join(outdir, "atlas_holdout_profile.csv"),
        "contrib": os.path.join(outdir, "atlas_contrib_spectrum.csv"),
        "chain": os.path.join(outdir, "atlas_training_chain.csv"),
        "counter": os.path.join(outdir, "atlas_counterfactual.csv"),
    }

    _write_csv(paths["rows"], rows)
    _write_csv(paths["pairs"], pairs)
    _write_csv(paths["profile"], profile_rows)
    _write_csv(paths["contrib"], contrib_rows)
    _write_csv(paths["chain"], chain_rows)
    _write_csv(paths["counter"], counter_rows)

    print(f"[Saved] {paths['rows']}")
    print(f"[Saved] {paths['pairs']}")
    if profile_rows:
        print(f"[Saved] {paths['profile']}")
    if contrib_rows:
        print(f"[Saved] {paths['contrib']}")
    if chain_rows:
        print(f"[Saved] {paths['chain']}")
    if counter_rows:
        print(f"[Saved] {paths['counter']}")

    return paths


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _plot_figA(df_pairs: "pd.DataFrame", outdir: str) -> None:
    cats = []
    vals = []
    for fam in sorted(df_pairs["family"].unique()):
        for base in ["prob_mse", "xent"]:
            d = df_pairs[(df_pairs["family"] == fam) & (df_pairs["base_loss"] == base)]["delta_qH"].to_numpy()
            cats.append(f"{fam}\nvs {base}")
            vals.append(d[np.isfinite(d)])

    fig, ax = plt.subplots(figsize=hv.fig_size("full", 3.0), constrained_layout=True)
    bp = ax.boxplot(vals, tick_labels=cats, showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#f3f3f3", edgecolor="#444444")
    for i, arr in enumerate(vals, start=1):
        if arr.size > 0:
            jitter = np.random.default_rng(0).normal(0, 0.05, size=arr.size)
            ax.scatter(np.full(arr.size, i) + jitter, arr, s=8, alpha=0.5, color="#1f77b4")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylabel(r"$\Delta q(H)$ (parity - base)")
    ax.set_title("Paired holdout-mass shift")
    fig.savefig(os.path.join(outdir, "figA_paired_delta_qH_violin.pdf"))
    plt.close(fig)


def _plot_figB(df_pairs: "pd.DataFrame", outdir: str) -> None:
    cats = []
    vals = []
    inf_text = []
    for fam in sorted(df_pairs["family"].unique()):
        for base in ["prob_mse", "xent"]:
            dd = df_pairs[(df_pairs["family"] == fam) & (df_pairs["base_loss"] == base)]
            ratio = dd["Q80_ratio"].to_numpy(dtype=float)
            finite = ratio[np.isfinite(ratio) & (ratio > 0)]
            inf_count = int(np.sum(~np.isfinite(dd["Q80_parity"].to_numpy(dtype=float)) | ~np.isfinite(dd["Q80_base"].to_numpy(dtype=float))))
            cats.append(f"{fam}\nvs {base}")
            vals.append(finite)
            inf_text.append(inf_count)

    fig, ax = plt.subplots(figsize=hv.fig_size("full", 3.0), constrained_layout=True)
    bp = ax.boxplot(vals, tick_labels=cats, showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#fff2e6", edgecolor="#a34f00")
    for i, arr in enumerate(vals, start=1):
        if arr.size > 0:
            jitter = np.random.default_rng(1).normal(0, 0.05, size=arr.size)
            ax.scatter(np.full(arr.size, i) + jitter, arr, s=8, alpha=0.5, color="#d62728")
        ax.text(i, 0.92 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0, f"inf={inf_text[i-1]}", ha="center", va="top", fontsize=7)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel(r"$Q80_{parity}/Q80_{base}$")
    ax.set_title("Paired discoverability ratio (finite pairs)")
    fig.savefig(os.path.join(outdir, "figB_paired_Q80_ratio_violin.pdf"))
    plt.close(fig)


def _plot_figC(df_pairs: "pd.DataFrame", outdir: str, max_arrows: int = 240) -> None:
    fig, ax = plt.subplots(figsize=hv.fig_size("full", 3.1), constrained_layout=True)
    rng = np.random.default_rng(123)

    for base, color in [("prob_mse", "#1f77b4"), ("xent", "#ff7f0e")]:
        d = df_pairs[df_pairs["base_loss"] == base]
        if len(d) == 0:
            continue
        if len(d) > max_arrows:
            d = d.sample(n=max_arrows, random_state=123)
        xb = d["qH_base"].to_numpy(dtype=float)
        yb = d["Q80_base"].to_numpy(dtype=float)
        xp = d["qH_parity"].to_numpy(dtype=float)
        yp = d["Q80_parity"].to_numpy(dtype=float)
        m = np.isfinite(xb) & np.isfinite(yb) & np.isfinite(xp) & np.isfinite(yp) & (xb > 0) & (xp > 0) & (yb > 0) & (yp > 0)
        xb, yb, xp, yp = xb[m], yb[m], xp[m], yp[m]
        for i in range(xb.size):
            ax.annotate("", xy=(xp[i], yp[i]), xytext=(xb[i], yb[i]), arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.35))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$q(H)$")
    ax.set_ylabel(r"$Q80$")
    ax.set_title("Base-loss to parity trajectory in (qH, Q80)")
    fig.savefig(os.path.join(outdir, "figC_qH_Q80_arrow_phaseplot.pdf"))
    plt.close(fig)


def _plot_figD(df_prof: "pd.DataFrame", outdir: str) -> None:
    bases = [b for b in ["prob_mse", "xent"] if b in set(df_prof["base_loss"].unique())]
    if not bases:
        return
    fig, axes = plt.subplots(1, len(bases), figsize=hv.fig_size("full", 3.0), constrained_layout=True)
    if len(bases) == 1:
        axes = [axes]
    for ax, base in zip(axes, bases):
        d = df_prof[df_prof["base_loss"] == base].sort_values("rank")
        x = d["rank"].to_numpy(dtype=int)
        ax.plot(x, d["q_parity"].to_numpy(dtype=float), color="#d62728", label="parity_mse")
        ax.plot(x, d["q_base"].to_numpy(dtype=float), color="#1f77b4", label=base)
        ax.set_title(base)
        ax.set_xlabel("Holdout rank")
        ax.set_ylabel("q(x) on H")
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(outdir, "figD_holdout_rank_mass_profile.pdf"))
    plt.close(fig)


def _plot_figE(df_prof: "pd.DataFrame", outdir: str) -> None:
    bases = [b for b in ["prob_mse", "xent"] if b in set(df_prof["base_loss"].unique())]
    if not bases:
        return
    fig, axes = plt.subplots(1, len(bases), figsize=hv.fig_size("full", 3.0), constrained_layout=True)
    if len(bases) == 1:
        axes = [axes]
    for ax, base in zip(axes, bases):
        d = df_prof[df_prof["base_loss"] == base].sort_values("rank")
        x = d["rank"].to_numpy(dtype=int)
        y = d["delta"].to_numpy(dtype=float)
        colors = np.where(y >= 0, "#2ca02c", "#d62728")
        ax.bar(x, y, color=colors, width=0.9)
        ax.axhline(0.0, color="#444444", lw=1.0)
        ax.set_title(base)
        ax.set_xlabel("Holdout rank")
        ax.set_ylabel(r"$\Delta_x = q_{parity}(x)-q_{base}(x)$")
    fig.savefig(os.path.join(outdir, "figE_holdout_delta_barcode.pdf"))
    plt.close(fig)


def _plot_figF(df_contrib: "pd.DataFrame", outdir: str) -> None:
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.9), constrained_layout=True)
    for loss, color in [("parity_mse", "#d62728"), ("prob_mse", "#1f77b4"), ("xent", "#ff7f0e")]:
        d = df_contrib[df_contrib["loss"] == loss].sort_values("rank")
        if len(d) == 0:
            continue
        ax.plot(d["rank"].to_numpy(dtype=int), d["c_signed"].to_numpy(dtype=float), label=loss, color=color)
    ax.axhline(0.0, color="#666666", lw=1.0)
    ax.set_xlabel("Feature rank (|c_k| sorted)")
    ax.set_ylabel("Signed contribution c_k")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(outdir, "figF_signed_contribution_spectrum.pdf"))
    plt.close(fig)


def _plot_figG(df_contrib: "pd.DataFrame", outdir: str) -> None:
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.9), constrained_layout=True)
    for loss, color in [("parity_mse", "#d62728"), ("prob_mse", "#1f77b4"), ("xent", "#ff7f0e")]:
        d = df_contrib[df_contrib["loss"] == loss].sort_values("rank")
        if len(d) == 0:
            continue
        ax.plot(d["rank"].to_numpy(dtype=int), d["cum_signed"].to_numpy(dtype=float), label=loss, color=color)
    ax.axhline(0.0, color="#666666", lw=1.0)
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Cumulative signed contribution")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(outdir, "figG_cumulative_visibility_curve.pdf"))
    plt.close(fig)


def _plot_figH(df_rows: "pd.DataFrame", outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=hv.fig_size("full", 3.0), constrained_layout=True)
    for loss, color in [("parity_mse", "#d62728"), ("prob_mse", "#1f77b4"), ("xent", "#ff7f0e")]:
        d = df_rows[df_rows["loss"] == loss]
        x = d["cos_align"].to_numpy(dtype=float)
        y1 = d["qH"].to_numpy(dtype=float)
        y2 = d["Q80"].to_numpy(dtype=float)
        m1 = np.isfinite(x) & np.isfinite(y1)
        m2 = np.isfinite(x) & np.isfinite(y2) & (y2 > 0)
        axes[0].scatter(x[m1], y1[m1], s=10, alpha=0.4, color=color, label=loss)
        axes[1].scatter(x[m2], np.log10(y2[m2]), s=10, alpha=0.4, color=color, label=loss)
    axes[0].set_xlabel("cos(z, hat1H)")
    axes[0].set_ylabel("q(H)")
    axes[1].set_xlabel("cos(z, hat1H)")
    axes[1].set_ylabel("log10(Q80)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(outdir, "figH_alignment_vs_qH_and_Q80.pdf"))
    plt.close(fig)


def _plot_figI(df_chain: "pd.DataFrame", outdir: str) -> None:
    if len(df_chain) == 0:
        return
    fig, axes = plt.subplots(2, 2, figsize=hv.fig_size("full", 4.6), constrained_layout=True)
    panels = [
        ("train_loss", "Train loss"),
        ("vis_model", "Visibility"),
        ("qH", "q(H)"),
        ("Q80_pred", "Q80_pred"),
    ]
    for loss, color in [("parity_mse", "#d62728"), ("prob_mse", "#1f77b4"), ("xent", "#ff7f0e")]:
        d = df_chain[df_chain["loss"] == loss].sort_values("step")
        if len(d) == 0:
            continue
        x = d["step"].to_numpy(dtype=int)
        for ax, (col, lab) in zip(axes.ravel(), panels):
            y = d[col].to_numpy(dtype=float)
            m = np.isfinite(y) & (y > 0 if col == "Q80_pred" else np.ones_like(y, dtype=bool))
            if np.any(m):
                ax.plot(x[m], y[m], label=loss, color=color)
            ax.set_xlabel("Step")
            ax.set_ylabel(lab)
            if col == "Q80_pred":
                ax.set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(outdir, "figI_training_dynamics_chain.pdf"))
    plt.close(fig)


def _plot_figJ(df_counter: "pd.DataFrame", outdir: str) -> None:
    if len(df_counter) == 0:
        return
    d = df_counter.sort_values("group")
    groups = d["group"].tolist()
    x = np.arange(len(groups))

    fig, axes = plt.subplots(1, 3, figsize=hv.fig_size("full", 2.7), constrained_layout=True)

    axes[0].bar(x, d["vis_ref"].to_numpy(dtype=float), color=["#1f77b4", "#d62728"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, rotation=15)
    axes[0].set_ylabel("Ref visibility")

    axes[1].bar(x, d["qH_ratio_model"].to_numpy(dtype=float), color=["#1f77b4", "#d62728"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, rotation=15)
    axes[1].set_ylabel("q(H)/q_unif(H)")

    q80 = d["Q80_model"].to_numpy(dtype=float)
    q80_plot = np.where(np.isfinite(q80), q80, np.nan)
    axes[2].bar(x, q80_plot, color=["#1f77b4", "#d62728"])
    axes[2].set_yscale("log")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(groups, rotation=15)
    axes[2].set_ylabel("Q80")

    fig.savefig(os.path.join(outdir, "figJ_counterfactual_visibility_intervention.pdf"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# Analyze mode
# -----------------------------------------------------------------------------


def _analyze_and_plot(paths: Dict[str, str], outdir: str, args: argparse.Namespace) -> Dict[str, object]:
    if pd is None:
        raise RuntimeError(f"pandas is required for analyze mode: {_PANDAS_IMPORT_ERROR}")

    rows_path = paths["rows"]
    pairs_path = paths["pairs"]
    if not os.path.exists(rows_path) or not os.path.exists(pairs_path):
        raise FileNotFoundError("Missing atlas_rows.csv or atlas_pairs.csv. Run with --mode run+analyze first.")

    df_rows = pd.read_csv(rows_path)
    df_pairs = pd.read_csv(pairs_path)
    df_prof = pd.read_csv(paths["profile"]) if os.path.exists(paths["profile"]) and os.path.getsize(paths["profile"]) > 0 else pd.DataFrame()
    df_contrib = pd.read_csv(paths["contrib"]) if os.path.exists(paths["contrib"]) and os.path.getsize(paths["contrib"]) > 0 else pd.DataFrame()
    df_chain = pd.read_csv(paths["chain"]) if os.path.exists(paths["chain"]) and os.path.getsize(paths["chain"]) > 0 else pd.DataFrame()
    df_counter = pd.read_csv(paths["counter"]) if os.path.exists(paths["counter"]) and os.path.getsize(paths["counter"]) > 0 else pd.DataFrame()

    hv.set_style(base=8)

    _plot_figA(df_pairs, outdir)
    _plot_figB(df_pairs, outdir)
    _plot_figC(df_pairs, outdir)
    if len(df_prof) > 0:
        _plot_figD(df_prof, outdir)
        _plot_figE(df_prof, outdir)
    else:
        # still create empty placeholders for required filenames
        for name in ["figD_holdout_rank_mass_profile.pdf", "figE_holdout_delta_barcode.pdf"]:
            fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
            ax.text(0.5, 0.5, "No profile data available", ha="center", va="center")
            ax.axis("off")
            fig.savefig(os.path.join(outdir, name))
            plt.close(fig)

    if len(df_contrib) > 0:
        _plot_figF(df_contrib, outdir)
        _plot_figG(df_contrib, outdir)
    else:
        for name in ["figF_signed_contribution_spectrum.pdf", "figG_cumulative_visibility_curve.pdf"]:
            fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
            ax.text(0.5, 0.5, "No contribution data available", ha="center", va="center")
            ax.axis("off")
            fig.savefig(os.path.join(outdir, name))
            plt.close(fig)

    _plot_figH(df_rows, outdir)
    _plot_figI(df_chain, outdir)
    _plot_figJ(df_counter, outdir)

    # Aggregated summary
    summary: Dict[str, object] = {
        "n_rows": int(len(df_rows)),
        "n_pairs": int(len(df_pairs)),
        "loss_counts": {str(k): int(v) for k, v in df_rows["loss"].value_counts().to_dict().items()},
        "q80_inf_counts": {str(k): int(v) for k, v in df_rows.assign(is_inf=~np.isfinite(df_rows["Q80"])) .groupby("loss")["is_inf"].sum().to_dict().items()},
        "pair_stats": {},
        "correlations": {},
    }

    # Pair stats by family/base
    pair_stats: Dict[str, object] = {}
    for fam in sorted(df_pairs["family"].unique()):
        for base in ["prob_mse", "xent"]:
            d = df_pairs[(df_pairs["family"] == fam) & (df_pairs["base_loss"] == base)]
            if len(d) == 0:
                continue
            key = f"{fam}__{base}"
            vals_dqH = d["delta_qH"].to_numpy(dtype=float)
            vals_dqHr = d["delta_qH_ratio"].to_numpy(dtype=float)
            vals_ratio = d["Q80_ratio"].to_numpy(dtype=float)
            vals_eta = d["eta_H_plus"].to_numpy(dtype=float)

            finite_ratio = vals_ratio[np.isfinite(vals_ratio) & (vals_ratio > 0)]

            pair_stats[key] = {
                "n": int(len(d)),
                "finite_q80_ratio_n": int(finite_ratio.size),
                "delta_qH": _bootstrap_median_ci(vals_dqH, int(args.n_bootstrap), float(args.alpha), seed=101),
                "delta_qH_ratio": _bootstrap_median_ci(vals_dqHr, int(args.n_bootstrap), float(args.alpha), seed=202),
                "Q80_ratio": _bootstrap_median_ci(finite_ratio, int(args.n_bootstrap), float(args.alpha), seed=303),
                "eta_H_plus": _bootstrap_median_ci(vals_eta, int(args.n_bootstrap), float(args.alpha), seed=404),
                "wilcoxon_delta_qH_zero": _wilcoxon_zero_center(vals_dqH),
            }

    summary["pair_stats"] = pair_stats

    # Requested correlations
    summary["correlations"] = {
        "cos_align_vs_qH_all": _spearman(df_rows["cos_align"].to_numpy(dtype=float), df_rows["qH"].to_numpy(dtype=float)),
        "cos_align_vs_logQ80_all": _spearman(
            df_rows["cos_align"].to_numpy(dtype=float),
            np.array([_safe_log10(v) for v in df_rows["Q80"].to_numpy(dtype=float)], dtype=np.float64),
        ),
        "by_loss": {},
    }

    by_loss: Dict[str, object] = {}
    for loss in sorted(df_rows["loss"].unique()):
        d = df_rows[df_rows["loss"] == loss]
        by_loss[str(loss)] = {
            "cos_align_vs_qH": _spearman(d["cos_align"].to_numpy(dtype=float), d["qH"].to_numpy(dtype=float)),
            "cos_align_vs_logQ80": _spearman(
                d["cos_align"].to_numpy(dtype=float),
                np.array([_safe_log10(v) for v in d["Q80"].to_numpy(dtype=float)], dtype=np.float64),
            ),
        }
    summary["correlations"]["by_loss"] = by_loss

    summary_path = os.path.join(outdir, "atlas_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {summary_path}")

    return summary


# -----------------------------------------------------------------------------
# Report writer
# -----------------------------------------------------------------------------


def _write_report(summary: Dict[str, object], outdir: str) -> str:
    report_path = str(ROOT / "docs" / "exp08_mechanism_report.md")

    cor = summary.get("correlations", {})
    pair_stats = summary.get("pair_stats", {})

    lines = []
    lines.append("# EXP08 Mechanism Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("Mechanistic ablation of IQP losses: parity_mse vs prob_mse/xent under matched protocol.")
    lines.append("")
    lines.append("## Main Outcome")
    lines.append("Parity-moment training is compared against standard losses via holdout mass flow and Q80.")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append(f"- atlas_rows: {summary.get('n_rows')}")
    lines.append(f"- atlas_pairs: {summary.get('n_pairs')}")
    lines.append("")

    lines.append("## Pairwise Effects (Median + 95% bootstrap CI)")
    for key in sorted(pair_stats.keys()):
        st = pair_stats[key]
        dq = st["delta_qH"]
        dqhr = st["delta_qH_ratio"]
        qr = st["Q80_ratio"]
        eta = st["eta_H_plus"]
        lines.append(f"- {key} (n={st['n']}, finite Q80 ratio n={st['finite_q80_ratio_n']}):")
        lines.append(f"  delta_qH: median={dq['median']:.6g}, CI=[{dq['lo']:.6g}, {dq['hi']:.6g}]")
        lines.append(f"  delta_qH_ratio: median={dqhr['median']:.6g}, CI=[{dqhr['lo']:.6g}, {dqhr['hi']:.6g}]")
        lines.append(f"  Q80_ratio: median={qr['median']:.6g}, CI=[{qr['lo']:.6g}, {qr['hi']:.6g}]")
        lines.append(f"  eta_H_plus: median={eta['median']:.6g}, CI=[{eta['lo']:.6g}, {eta['hi']:.6g}]")
        wz = st["wilcoxon_delta_qH_zero"]
        lines.append(f"  Wilcoxon(delta_qH=0): p={wz['p']}")
    lines.append("")

    lines.append("## Mechanistic Correlations")
    all_qh = cor.get("cos_align_vs_qH_all", {})
    all_q80 = cor.get("cos_align_vs_logQ80_all", {})
    lines.append(f"- cos_align vs qH (all losses): rho={all_qh.get('rho')}, p={all_qh.get('p')}, n={all_qh.get('n')}")
    lines.append(f"- cos_align vs logQ80 (all losses): rho={all_q80.get('rho')}, p={all_q80.get('p')}, n={all_q80.get('n')}")
    by_loss = cor.get("by_loss", {})
    for loss in sorted(by_loss.keys()):
        lq = by_loss[loss].get("cos_align_vs_qH", {})
        l80 = by_loss[loss].get("cos_align_vs_logQ80", {})
        lines.append(f"- {loss}: rho(cos,qH)={lq.get('rho')}, rho(cos,logQ80)={l80.get('rho')}")
    lines.append("")

    lines.append("## Mass-flow Interpretation")
    lines.append("Positive delta_qH and eta_H_plus indicate directed mass transfer toward unseen holdout states.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("- Controlled synthetic benchmark; no broad OOD/real-world generalization claim.")
    lines.append("- Q80=inf cases are reported explicitly and affect finite-ratio analyses.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Output directory: `{outdir}`")
    lines.append("- Required files: figA..figJ, atlas_rows.csv, atlas_pairs.csv, atlas_summary.json")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Saved] {report_path}")
    return report_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _paths(outdir: str) -> Dict[str, str]:
    return {
        "rows": os.path.join(outdir, "atlas_rows.csv"),
        "pairs": os.path.join(outdir, "atlas_pairs.csv"),
        "profile": os.path.join(outdir, "atlas_holdout_profile.csv"),
        "contrib": os.path.join(outdir, "atlas_contrib_spectrum.csv"),
        "chain": os.path.join(outdir, "atlas_training_chain.csv"),
        "counter": os.path.join(outdir, "atlas_counterfactual.csv"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Mechanistic discoverability atlas (exp08)")
    ap.add_argument("--mode", type=str, default="run+analyze", choices=["run", "analyze", "run+analyze"])
    ap.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "exp08_mechanism_atlas"))

    # Required in explanation
    ap.add_argument("--families", type=str, default="paper_even,paper_nonparity")
    ap.add_argument("--betas", type=str, default="0.4,0.8,1.2")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--sigmas", type=str, default="0.5,1.0,2.0,3.0")
    ap.add_argument("--Ks", type=str, default="128,256,512")
    ap.add_argument("--iqp-losses", type=str, default="parity_mse,prob_mse,xent")

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)

    ap.add_argument("--iqp-steps", type=int, default=300)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=25)
    ap.add_argument("--iqp-layers", type=int, default=1)

    ap.add_argument("--Q80-thr", type=float, default=0.8)
    ap.add_argument("--Q80-search-max", type=int, default=200000)

    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)

    # Representative tuple for detailed mechanism figures
    ap.add_argument("--rep-family", type=str, default="paper_even")
    ap.add_argument("--rep-beta", type=float, default=0.8)
    ap.add_argument("--rep-seed", type=int, default=0)
    ap.add_argument("--rep-sigma", type=float, default=1.0)
    ap.add_argument("--rep-K", type=int, default=256)

    # Counterfactual intervention
    ap.add_argument("--make-counterfactual", type=int, default=1)
    ap.add_argument("--counter-family", type=str, default="paper_even")
    ap.add_argument("--counter-beta", type=float, default=0.8)
    ap.add_argument("--counter-seed", type=int, default=0)
    ap.add_argument("--counter-sigma", type=float, default=1.0)
    ap.add_argument("--counter-K", type=int, default=512)
    ap.add_argument("--counter-trials", type=int, default=1500)

    ap.add_argument("--make-training-chain", type=int, default=1)

    args = ap.parse_args()

    outdir = _ensure_outdir(args.outdir)
    hv.set_style(base=8)

    cfg_path = os.path.join(outdir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.mode in ("run", "run+analyze"):
        _run_suite(args, outdir)

    summary = None
    if args.mode in ("analyze", "run+analyze"):
        summary = _analyze_and_plot(_paths(outdir), outdir, args)

    if summary is not None:
        _write_report(summary, outdir)

    print(f"Done. Results in {outdir}")


if __name__ == "__main__":
    main()
