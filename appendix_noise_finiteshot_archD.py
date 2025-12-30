#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix Figures (PRA-ready): Finite-shot + Noise Robustness (Arch D)
====================================================================

What this script does (evaluation-only; no retraining)
-----------------------------------------------------
It consumes the artifacts produced by your main holdout script:

  <indir>/
    holdout_strings.txt
    runs/run_archD_seed42.json
    runs/run_archD_seedXX.json
    ...

Each run JSON must contain:
  "holdout_probs": list of q_theta(x) for x in H (order aligned with ascending holdout indices)

It then generates two clean, appendix-ready PDFs:

  figG_archD_noise_recovery_curves.pdf
  figG_archD_finite_shot_Q80_estimation.pdf

Key presentation requirements satisfied
---------------------------------------
- No in-plot explanatory text boxes (no gray text overlay).
- Legends placed where they do not obscure data.
- Constrained layout + conservative font sizes for RevTeX.
- Stable axis limits so nothing gets clipped.

Noise model (simple, interpretable)
-----------------------------------
Effective mixing-to-uniform on the *output distribution*:
    q_eta(x) = (1-eta) q(x) + eta * 2^{-n}.
This is not "hardware-accurate", but is a standard appendix sanity check:
recovery/Q80 should degrade smoothly rather than change qualitatively.

Finite-shot model (for estimating q_theta(H))
---------------------------------------------
Given S shots and event "sample lands in H", we estimate:
    qhat(H) = (#hits in H) / S,
with #hits ~ Binomial(S, q(H)).
We propagate shot noise through Eq.(2) predictor Q80^pred ~ |H| ln 5 / q(H)
and plot the *relative* finite-shot error:
    Q80^pred(qhat)/Q80^pred(q) = q(H)/qhat(H).

Usage example
-------------
python appendix_finiteshot_noise_pretty.py \
  --indir ablation_holdout \
  --arch D \
  --noise-etas 0 0.1 0.2 0.3 \
  --shot-list 200 500 1000 2000 5000 \
  --mc-reps 4000 \
  --Qmax 10000 \
  --fig-target full
"""

import os
import glob
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# 0) Journal-aware sizes + style (RevTeX-friendly)
# ------------------------------------------------------------------------------

COL_W  = 3.37
FULL_W = 6.95

COLORS = {
    "black": "#111111",
    "gray":  "#666666",
    "light": "#DDDDDD",
    "hero":  "#D62728",
}

def fig_size(fig_target: str, h_col: float, h_full: float) -> Tuple[float, float]:
    return (COL_W, h_col) if fig_target == "col" else (FULL_W, h_full)

def set_style(fig_target: str = "full") -> None:
    # A touch smaller labels than the main script → more robust (no clipping) in RevTeX.
    base = 8 if fig_target == "col" else 11
    label = base + 1 if fig_target == "col" else 14
    title = base + 1 if fig_target == "col" else 14
    lw = 1.6 if fig_target == "col" else 2.5

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": base,
        "axes.labelsize": label,
        "axes.titlesize": title,
        "legend.fontsize": base if fig_target == "col" else 11,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.edgecolor": "white",
        "legend.borderpad": 0.35,
        "legend.handlelength": 2.2,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.0,
        "lines.linewidth": lw,
        "axes.linewidth": 0.8,
        "xtick.direction": "out" if fig_target == "col" else "in",
        "ytick.direction": "out" if fig_target == "col" else "in",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ------------------------------------------------------------------------------
# 1) Load artifacts
# ------------------------------------------------------------------------------

def read_holdout_strings(path: str) -> Dict[str, Any]:
    """
    Parse holdout_strings.txt written by your main script.

    Non-comment format:
      idx  bitstring  score  p_star

    We sort by idx ascending to align with run JSON, which stores holdout_probs
    in ascending index order (because holdout_idxs comes from np.where).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            bitstr = parts[1].strip()
            pstar = float(parts[3])
            rows.append((idx, bitstr, pstar))

    if not rows:
        raise RuntimeError(f"Could not parse any holdout entries from {path}")

    rows.sort(key=lambda t: t[0])
    n = len(rows[0][1])

    idxs = np.array([r[0] for r in rows], dtype=int)
    p_star = np.array([r[2] for r in rows], dtype=np.float64)
    return {"n": n, "H": len(rows), "idxs": idxs, "p_star": p_star}

def find_run_files(indir: str, arch: str, seeds: Optional[List[int]] = None) -> List[str]:
    runs_dir = os.path.join(indir, "runs")
    if seeds is None or len(seeds) == 0:
        return sorted(glob.glob(os.path.join(runs_dir, f"run_arch{arch}_seed*.json")))
    files = [os.path.join(runs_dir, f"run_arch{arch}_seed{s}.json") for s in seeds]
    return [f for f in files if os.path.exists(f)]

def load_runs(indir: str, arch: str, seeds: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    files = find_run_files(indir, arch, seeds)
    if not files:
        raise FileNotFoundError(
            f"No run files for arch={arch} found in {os.path.join(indir,'runs')}.\n"
            f"Expected e.g. run_arch{arch}_seed42.json.\n"
            f"Generate them with your training script first."
        )
    runs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            r = json.load(f)
        if "holdout_probs" not in r:
            raise RuntimeError(f"Missing 'holdout_probs' in {fp}")
        runs.append(r)
    return runs


# ------------------------------------------------------------------------------
# 2) Recovery math
# ------------------------------------------------------------------------------

def expected_unique_from_probs(p_vec: np.ndarray, Q_vals: np.ndarray) -> np.ndarray:
    if p_vec.size == 0:
        return np.zeros_like(Q_vals, dtype=np.float64)
    p = np.clip(p_vec.astype(np.float64), 0.0, 1.0)[:, None]
    Q = Q_vals[None, :].astype(np.int64)
    return np.sum(1.0 - np.power(1.0 - p, Q), axis=0)

def expected_recovery_curve(p_vec: np.ndarray, Q_grid: np.ndarray) -> np.ndarray:
    H = int(p_vec.size)
    if H == 0:
        return np.zeros_like(Q_grid, dtype=np.float64)
    return expected_unique_from_probs(p_vec, Q_grid) / H

def recovery_fraction(p_vec: np.ndarray, Q: int) -> float:
    H = int(p_vec.size)
    if H == 0:
        return 0.0
    return float(expected_unique_from_probs(p_vec, np.array([int(Q)]))[0] / H)

def min_Q_for_recovery(p_vec: np.ndarray, target_frac: float, Q_cap: int = 50_000_000) -> float:
    H = int(p_vec.size)
    if H == 0:
        return float("nan")
    p_vec = np.array(p_vec, dtype=np.float64)
    max_reachable = float(np.count_nonzero(p_vec > 0.0) / H)
    if max_reachable + 1e-12 < target_frac:
        return float("inf")
    hi = 1
    while hi < Q_cap and recovery_fraction(p_vec, hi) < target_frac:
        hi *= 2
    if recovery_fraction(p_vec, hi) < target_frac:
        return float("inf")
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if recovery_fraction(p_vec, mid) >= target_frac:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ------------------------------------------------------------------------------
# 3) Noise model: mixing-to-uniform on output distribution
# ------------------------------------------------------------------------------

def apply_depolarizing_mix_holdout(p_holdout: np.ndarray, n: int, eta: float) -> np.ndarray:
    """
    q_eta(x) = (1-eta) q(x) + eta * 2^{-n}, restricted to x in H.
    """
    eta = float(eta)
    eta = min(max(eta, 0.0), 1.0)
    u = 1.0 / (2.0 ** n)
    return (1.0 - eta) * np.array(p_holdout, dtype=np.float64) + eta * u


# ------------------------------------------------------------------------------
# 4) Figure 1: Recovery curves under noise
# ------------------------------------------------------------------------------

def plot_noise_recovery_curves(
    *,
    outdir: str,
    fig_target: str,
    Qmax: int,
    p_star_holdout: np.ndarray,
    runs_holdout_probs: List[np.ndarray],
    n: int,
    noise_etas: List[float],
) -> str:
    Q_grid = np.unique(np.logspace(0, math.log10(Qmax), 260).astype(int))
    Q_grid = Q_grid[Q_grid >= 1]

    target_curve = expected_recovery_curve(p_star_holdout, Q_grid)
    target_Q80 = min_Q_for_recovery(p_star_holdout, 0.80)
    t_txt = str(int(target_Q80)) if np.isfinite(target_Q80) else "∞"

    fig, ax = plt.subplots(
        figsize=fig_size(fig_target, 2.9, 3.8),
        constrained_layout=True
    )

    ax.plot(
        Q_grid, target_curve,
        color=COLORS["black"],
        lw=(1.7 if fig_target == "col" else 2.3),
        ls=":",
        label=fr"Target $p^*$ (Q$_{{80}}$={t_txt})",
        zorder=5
    )

    line_styles = ["-", "--", "-.", (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2))]
    for j, eta in enumerate(noise_etas):
        curves = []
        q80s = []
        for pH in runs_holdout_probs:
            pH_noisy = apply_depolarizing_mix_holdout(pH, n=n, eta=eta)
            curves.append(expected_recovery_curve(pH_noisy, Q_grid))
            q80s.append(min_Q_for_recovery(pH_noisy, 0.80))

        C = np.array(curves, dtype=np.float64)
        m = np.mean(C, axis=0)
        s = np.std(C, axis=0, ddof=1) / math.sqrt(C.shape[0]) if C.shape[0] > 1 else np.zeros_like(m)

        q80_arr = np.array([x for x in q80s if np.isfinite(x)], dtype=np.float64)
        q80_mean = float(np.mean(q80_arr)) if q80_arr.size else float("inf")
        q80_txt = str(int(q80_mean)) if np.isfinite(q80_mean) else "∞"

        is_clean = (abs(eta) < 1e-12)
        col = COLORS["hero"] if is_clean else COLORS["black"]
        alpha = 1.0 if is_clean else 0.85
        lw = (3.2 if fig_target == "col" else 4.0) if is_clean else (2.0 if fig_target == "col" else 2.6)
        ls = line_styles[min(j, len(line_styles) - 1)]

        ax.plot(
            Q_grid, m,
            color=col, lw=lw, ls=ls, alpha=alpha,
            label=fr"Arch D, $\eta$={eta:g} (Q$_{{80}}$={q80_txt})",
            zorder=10 if is_clean else 8
        )
        if np.any(s > 0):
            ax.fill_between(
                Q_grid,
                np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                color=col, alpha=0.10 if is_clean else 0.06,
                linewidth=0, zorder=1
            )

    ax.set_xscale("log")
    ax.set_xlim(Q_grid.min(), Q_grid.max())
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Sampling Budget $Q$ (log)")
    ax.set_ylabel(r"Holdout Recovery  $U_H(Q)/|H|$")

    # Legend placed in the empty lower-right region (no overlap with curves)
    leg = ax.legend(loc="upper left")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.96)

    outpath = os.path.join(outdir, "figG_archD_noise_recovery_curves.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ------------------------------------------------------------------------------
# 5) Figure 2: Finite-shot uncertainty of Eq.(2) predictor (NO in-plot gray text)
# ------------------------------------------------------------------------------

def plot_finite_shot_Q80_estimation(
    *,
    outdir: str,
    fig_target: str,
    p_holdout_mean: np.ndarray,
    n: int,
    noise_etas: List[float],
    shot_list: List[int],
    mc_reps: int,
    rng_seed: int,
) -> str:
    """
    For each eta:
      - compute q(H) from noisy holdout slice
      - draw Binomial(shots, q(H)) to model finite-shot estimation of q(H)
      - plot ratio Q80^pred(qhat)/Q80^pred(q) = q(H)/qhat(H)
      - show median and 16–84% interval (paper-friendly)
    """
    H = int(p_holdout_mean.size)
    if H == 0:
        raise RuntimeError("Empty holdout; cannot plot finite-shot predictor uncertainty.")

    rng = np.random.default_rng(int(rng_seed))

    fig, ax = plt.subplots(
        figsize=fig_size(fig_target, 2.9, 3.8),
        constrained_layout=True
    )

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    line_styles = ["-", "--", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]
    colors = [COLORS["hero"], COLORS["black"], COLORS["gray"], "#999999", "#BBBBBB"]

    all_lo, all_hi = [], []

    for j, eta in enumerate(noise_etas):
        pH_noisy = apply_depolarizing_mix_holdout(p_holdout_mean, n=n, eta=eta)
        qH = float(np.sum(pH_noisy))
        if qH <= 0.0:
            continue

        med_list, lo_list, hi_list = [], [], []
        for shots in shot_list:
            shots = int(shots)
            counts = rng.binomial(n=shots, p=min(max(qH, 0.0), 1.0), size=mc_reps)
            qhat = counts.astype(np.float64) / max(1, shots)

            ratio = np.full_like(qhat, np.inf, dtype=np.float64)
            nz = qhat > 0
            ratio[nz] = qH / qhat[nz]

            finite = np.isfinite(ratio)
            rfin = ratio[finite]
            med = float(np.median(rfin))
            lo = float(np.quantile(rfin, 0.16))
            hi = float(np.quantile(rfin, 0.84))

            med_list.append(med)
            lo_list.append(lo)
            hi_list.append(hi)

        x = np.array(shot_list, dtype=np.float64)
        y = np.array(med_list, dtype=np.float64)
        yerr = np.vstack([y - np.array(lo_list), np.array(hi_list) - y])

        col = colors[min(j, len(colors) - 1)]
        mk = markers[min(j, len(markers) - 1)]
        ls = line_styles[min(j, len(line_styles) - 1)]

        ax.errorbar(
            x, y, yerr=yerr,
            marker=mk, linestyle=ls,
            markersize=6 if fig_target == "full" else 5,
            lw=2.0 if fig_target == "full" else 1.6,
            elinewidth=1.3,
            capsize=3.0,
            color=col,
            label=fr"$\eta$={eta:g}",
            zorder=5
        )

        all_lo.extend(lo_list)
        all_hi.extend(hi_list)

    # Reference line
    ax.axhline(1.0, color=COLORS["gray"], lw=1.4, ls="--", alpha=0.75, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"Shots used to estimate $q_\theta(H)$")
    ax.set_ylabel(r"$Q_{80}^{\mathrm{pred}}(\hat q)/Q_{80}^{\mathrm{pred}}(q)$")

    # Robust y-limits (no clipping, no huge empty space)
    finite_lo = np.array([v for v in all_lo if np.isfinite(v)], dtype=np.float64)
    finite_hi = np.array([v for v in all_hi if np.isfinite(v)], dtype=np.float64)
    if finite_lo.size and finite_hi.size:
        ymin = float(np.min(finite_lo)) * 0.92
        ymax = float(np.max(finite_hi)) * 1.10
        ymin = max(0.65, min(ymin, 0.95))
        ymax = min(2.00, max(ymax, 1.10))
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0.8, 1.4)

    # Legend: inside, upper-right (clean region)
    leg = ax.legend(loc="upper right")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.96)

    outpath = os.path.join(outdir, "figG_archD_finite_shot_Q80_estimation.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ------------------------------------------------------------------------------
# 6) Main
# ------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="ablation_holdout",
                    help="Directory produced by the main holdout ablation script.")
    ap.add_argument("--arch", type=str, default="D",
                    help="Architecture letter (default: D).")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Optional explicit list of seeds to load (default: auto-discover).")

    ap.add_argument("--noise-etas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3],
                    help="Noise strengths eta in [0,1] for q_eta=(1-eta)q + eta*unif.")
    ap.add_argument("--shot-list", type=int, nargs="+", default=[200, 500, 1000, 2000, 5000],
                    help="Shot counts for estimating q(H).")
    ap.add_argument("--mc-reps", type=int, default=4000,
                    help="Monte Carlo reps for shot-estimation uncertainty.")
    ap.add_argument("--Qmax", type=int, default=10000,
                    help="Max Q for recovery curves.")
    ap.add_argument("--fig-target", type=str, default="full", choices=["col", "full"],
                    help="Figure size preset.")
    ap.add_argument("--rng-seed", type=int, default=1234,
                    help="RNG seed for Monte Carlo.")
    args = ap.parse_args()

    set_style(args.fig_target)

    holdout_path = os.path.join(args.indir, "holdout_strings.txt")
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(
            f"Missing {holdout_path}. Run your main holdout ablation script first."
        )

    hold = read_holdout_strings(holdout_path)
    n = int(hold["n"])
    H = int(hold["H"])
    p_star_holdout = hold["p_star"]

    runs = load_runs(args.indir, args.arch, seeds=args.seeds)
    runs_holdout_probs = [np.array(r["holdout_probs"], dtype=np.float64) for r in runs]

    for k, pH in enumerate(runs_holdout_probs):
        if pH.size != H:
            raise RuntimeError(
                f"Holdout size mismatch: holdout_strings.txt has H={H}, "
                f"but run {k} has len(holdout_probs)={pH.size}."
            )

    p_holdout_mean = np.mean(np.stack(runs_holdout_probs, axis=0), axis=0)

    print(f"[Info] Loaded {len(runs)} run(s) for Arch {args.arch} | n={n} | |H|={H}")
    print(f"[Info] noise_etas={args.noise_etas}")
    print(f"[Info] shot_list={args.shot_list} | mc_reps={args.mc_reps}")

    out1 = plot_noise_recovery_curves(
        outdir=args.indir,
        fig_target=args.fig_target,
        Qmax=args.Qmax,
        p_star_holdout=p_star_holdout,
        runs_holdout_probs=runs_holdout_probs,
        n=n,
        noise_etas=list(args.noise_etas),
    )
    print(f"[Saved] {out1}")

    out2 = plot_finite_shot_Q80_estimation(
        outdir=args.indir,
        fig_target=args.fig_target,
        p_holdout_mean=p_holdout_mean,
        n=n,
        noise_etas=list(args.noise_etas),
        shot_list=list(args.shot_list),
        mc_reps=int(args.mc_reps),
        rng_seed=int(args.rng_seed),
    )
    print(f"[Saved] {out2}")

    print("Done.")

if __name__ == "__main__":
    main()
