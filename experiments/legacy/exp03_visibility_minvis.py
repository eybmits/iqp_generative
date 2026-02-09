#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adversarial visibility ablation with H_inv chosen by minimal |Vis|.

Standalone script (no PennyLane) that constructs two holdout sets of equal size and
equal target mass (by sampling from a single integer score level), then evaluates
recovery under the constraint-implied completion q~ (linear band + positivity + normalization).

Holdouts:
  - H_vis: maximal +visibility Vis_B(H)
  - H_inv: minimal absolute visibility |Vis_B(H)| (greedy subset-sum + local swap refinement)

Mechanism story:
  - Lemma 1 (linear level): q_lin(H) = |H|/2^n + Vis_B(H)
  - If |Vis_B(H_inv)| ≈ 0, then q_lin(H_inv) ≈ |H|/2^n  ⇒ collapse to uniform baseline.

IMPORTANT STYLE RULE FOR THIS FIGURE:
  - Red is reserved for IQP-QCBM curves elsewhere in the paper.
  - Here, "visible/invisible" are HOLDOUT TYPES (not different models), so they are plotted in blues.

Outputs
-------
  <outdir>/6a_adversarial_curves_minvis.pdf
  <outdir>/holdout_strings_visible_minvis.txt
  <outdir>/holdout_strings_invisible_minvis.txt
  <outdir>/minvis_summary.json

Usage
-----
  # Paper-like n=12 (auto sigma-grid will kick in unless you set --sigma-grid explicitly):
  python experiments/exp03_visibility_minvis.py --outdir outputs/exp03_visibility_minvis --n 12 --beta 0.9 \
      --holdout-k 20 --train-m 1000 --K 512 --score-level 7

  # Force a specific sigma:
  python experiments/exp03_visibility_minvis.py --outdir outputs/exp03_visibility_minvis --n 12 --sigma-grid 1.0

Optional:
  - choose sigma from a grid to make cancellation easier:
      --sigma-grid 0.5,1,2,3

Dependencies
------------
  pip install numpy matplotlib
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Style (paper look; NO RED here)
# ----------------------------------------------------------------------------

COL_W = 3.37
FIG_H = 2.6

# Two distinct blues for visible/invisible (legend-readable)
COLORS = {
    "target":  "#222222",  # almost black
    "vis":     "#1f77b4",  # matplotlib blue (visible)
    "inv":     "#0b3d91",  # darker blue (invisible)
    "uniform": "#888888",  # gray
}

# Linestyles chosen to be clearly distinguishable in legend
LS = {
    "target":  "-",                 # solid black
    "vis":     (0, (6, 2)),         # long-dash
    "inv":     (0, (2, 2)),         # short-dash
    "uniform": (0, (1, 2)),         # dotted-ish
}


def set_style(base: int = 8) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": base,
        "axes.labelsize": base + 1,
        "axes.titlesize": base + 1,
        "legend.fontsize": base - 1,
        "legend.frameon": False,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "lines.linewidth": 1.6,          # slightly thicker -> legend readability
        "lines.markersize": 4.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


ROOT = Path(__file__).resolve().parents[2]


def ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# Bitstring utilities + score / target distribution
# (Matches Experiment_1 / build_target_distribution_paper)
# ----------------------------------------------------------------------------

def int2bits(k: int, n: int) -> np.ndarray:
    return np.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=np.int8)


def bits_str(bits: np.ndarray) -> str:
    return "".join("1" if int(b) else "0" for b in bits)


def parity_even(bits: np.ndarray) -> bool:
    return (int(np.sum(bits)) % 2) == 0


def longest_zero_run_between_ones(bits: np.ndarray) -> int:
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2:
        return 0
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    return max(gaps) if gaps else 0


def make_bits_table(n: int) -> np.ndarray:
    N = 2 ** n
    return np.array([int2bits(i, n) for i in range(N)], dtype=np.int8)


def build_target_distribution(n: int, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target p*: even parity support, exponential tilt by score (paper target)."""
    N = 2 ** n
    scores = np.full(N, -100.0, dtype=np.float64)
    support = np.zeros(N, dtype=bool)

    for k in range(N):
        b = int2bits(k, n)
        if parity_even(b):
            support[k] = True
            scores[k] = 1.0 + float(longest_zero_run_between_ones(b))

    logits = np.full(N, -np.inf, dtype=np.float64)
    logits[support] = beta * scores[support]

    m = float(np.max(logits[support]))
    unnorm = np.zeros(N, dtype=np.float64)
    unnorm[support] = np.exp(logits[support] - m)
    p_star = unnorm / float(np.sum(unnorm))
    return p_star.astype(np.float64), support, scores.astype(np.float64)


def topk_mask(scores: np.ndarray, support: np.ndarray, frac: float = 0.05) -> np.ndarray:
    valid = np.where(support)[0]
    k = max(1, int(np.floor(frac * valid.size)))
    local_order = np.argsort(-scores[valid])
    top_indices = valid[local_order[:k]]
    mask = np.zeros_like(support, dtype=bool)
    mask[top_indices] = True
    return mask


# ----------------------------------------------------------------------------
# Walsh/parity features
# ----------------------------------------------------------------------------

def p_sigma(sigma: float) -> float:
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma ** 2))) if sigma > 0 else 0.5


def sample_alphas(n: int, sigma: float, K: int, seed: int) -> np.ndarray:
    """Sample parity masks alpha_k ~ Bernoulli(p(sigma)), excluding the all-zero mask."""
    rng = np.random.default_rng(seed)
    p = p_sigma(sigma)
    alphas = rng.binomial(1, p, size=(K, n)).astype(np.int8)

    zero = np.where(np.sum(alphas, axis=1) == 0)[0]
    while zero.size > 0:
        alphas[zero] = rng.binomial(1, p, size=(zero.size, n)).astype(np.int8)
        zero = np.where(np.sum(alphas, axis=1) == 0)[0]

    return alphas


def build_parity_matrix(alphas: np.ndarray, bits_table: np.ndarray) -> np.ndarray:
    """P[k,x] = (-1)^{alpha_k · x} ∈ {+1, -1}."""
    A = alphas.astype(np.int16)
    X = bits_table.astype(np.int16).T
    par = (A @ X) & 1
    return np.where(par == 0, 1.0, -1.0).astype(np.float64)


# ----------------------------------------------------------------------------
# Mechanism objects: q_lin, completion, recovery
# ----------------------------------------------------------------------------

def linear_band_reconstruction(P: np.ndarray, z: np.ndarray, n: int) -> np.ndarray:
    N = 2 ** n
    return (1.0 / N) * (1.0 + (P.T @ z))


def completion_by_axioms(q_lin: np.ndarray) -> np.ndarray:
    q = np.clip(q_lin, 0.0, None)
    s = float(np.sum(q))
    if s <= 0:
        return np.ones_like(q, dtype=np.float64) / q.size
    return (q / s).astype(np.float64)


def sample_indices(probs: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = probs / float(np.sum(probs))
    return rng.choice(len(p), size=m, replace=True, p=p)


def empirical_dist(idxs: np.ndarray, N: int) -> np.ndarray:
    c = np.bincount(idxs, minlength=N)
    return (c / max(1, int(np.sum(c)))).astype(np.float64)


def expected_unique_fraction(probs: np.ndarray, mask: np.ndarray, Q_vals: np.ndarray) -> np.ndarray:
    Q_vals = np.array(Q_vals, dtype=int)
    H = int(np.sum(mask))
    if H == 0:
        return np.zeros_like(Q_vals, dtype=np.float64)
    pS = probs[mask].astype(np.float64)[:, None]
    return np.sum(1.0 - np.power(1.0 - pS, Q_vals[None, :]), axis=0) / H


def compute_Q80(probs: np.ndarray, mask: np.ndarray, thr: float = 0.8, Qmax: int = 200000) -> float:
    H = int(np.sum(mask))
    if H == 0:
        return float("nan")

    def frac(Q: int) -> float:
        return float(expected_unique_fraction(probs, mask, np.array([Q]))[0])

    if frac(1) >= thr:
        return 1.0

    lo, hi = 1, 1
    while hi < Qmax and frac(hi) < thr:
        hi *= 2
    if hi >= Qmax and frac(Qmax) < thr:
        return float("inf")
    hi = min(hi, Qmax)

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if frac(mid) >= thr:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ----------------------------------------------------------------------------
# Holdout construction
# ----------------------------------------------------------------------------

def greedy_min_abs_sum(
    cand_idxs: np.ndarray,
    cand_vals: np.ndarray,
    k: int,
    improve_steps: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    if cand_idxs.size < k:
        raise ValueError("Not enough candidates for greedy selection")

    remaining = list(range(cand_idxs.size))
    selected = []
    S = 0.0

    for _ in range(k):
        best_j = None
        best_abs = None
        for j in remaining:
            a = abs(S + float(cand_vals[j]))
            if (best_abs is None) or (a < best_abs):
                best_abs = a
                best_j = j
        assert best_j is not None
        selected.append(best_j)
        S += float(cand_vals[best_j])
        remaining.remove(best_j)

    selected_set = set(selected)
    remaining_set = set(remaining)

    if improve_steps > 0 and len(remaining) > 0:
        rng = np.random.default_rng(seed)
        sel_list = list(selected_set)
        rem_list = list(remaining_set)

        for _ in range(improve_steps):
            a = int(rng.choice(sel_list))
            b = int(rng.choice(rem_list))
            S_new = S - float(cand_vals[a]) + float(cand_vals[b])
            if abs(S_new) < abs(S):
                S = S_new
                selected_set.remove(a)
                selected_set.add(b)
                remaining_set.remove(b)
                remaining_set.add(a)
                sel_list = list(selected_set)
                rem_list = list(remaining_set)

    sel_pos = np.array(list(selected_set), dtype=int)
    if sel_pos.size != k:
        sel_pos = np.array(selected, dtype=int)

    sel_idxs = cand_idxs[sel_pos]
    S_final = float(np.sum(cand_vals[sel_pos]))
    return sel_idxs.astype(int), S_final


def save_holdout_list(
    holdout_idxs: np.ndarray,
    bits_table: np.ndarray,
    p_star: np.ndarray,
    scores: np.ndarray,
    outpath: Path,
    header: str,
) -> None:
    idxs = np.array(holdout_idxs, dtype=int)
    order = idxs[np.argsort(-p_star[idxs])]
    with outpath.open("w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        f.write(f"# k={idxs.size}\n")
        f.write(f"# {'Index':<8} {'Bitstring':<18} {'Score':<8} {'Prob p*(x)':<16}\n")
        f.write("-" * 60 + "\n")
        for i in order:
            b = bits_table[int(i)]
            f.write(
                f"{int(i):<8d} {bits_str(b):<18s} {float(scores[int(i)]):<8.1f} {float(p_star[int(i)]):.6e}\n"
            )


# ----------------------------------------------------------------------------
# Plotting (Visible/Invisible in two blues + legend-readable dashes)
# ----------------------------------------------------------------------------

def plot_adversarial_curves(
    p_star: np.ndarray,
    holdout_vis_mask: np.ndarray,
    holdout_inv_mask: np.ndarray,
    q_vis: np.ndarray,
    q_inv: np.ndarray,
    outpath: Path,
    Qmax: int = 10000,
) -> None:
    Q = np.unique(np.concatenate([
        np.unique(np.logspace(0, 3.5, 110).astype(int)),
        np.linspace(1000, Qmax, 120).astype(int),
    ]))
    Q = Q[Q <= Qmax]

    # Target curve: we use H_vis mask; by construction p*(H_vis)=p*(H_inv), so this matches either.
    y_star = expected_unique_fraction(p_star, holdout_vis_mask, Q)
    y_vis = expected_unique_fraction(q_vis, holdout_vis_mask, Q)
    y_inv = expected_unique_fraction(q_inv, holdout_inv_mask, Q)

    u = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_u = expected_unique_fraction(u, holdout_vis_mask, Q)

    fig, ax = plt.subplots(figsize=(COL_W, FIG_H), constrained_layout=True)

    ax.plot(Q, y_star, color=COLORS["target"], linestyle=LS["target"],
            linewidth=1.9, label=r"Target $p^*$", zorder=4)

    ax.plot(Q, y_vis, color=COLORS["vis"], linestyle=LS["vis"],
            linewidth=2.2, label=r"Visible $\mathcal{H}_{\mathrm{vis}}$", zorder=6)

    ax.plot(Q, y_inv, color=COLORS["inv"], linestyle=LS["inv"],
            linewidth=2.0, label=r"Invisible $\mathcal{H}_{\mathrm{inv}}$", zorder=5)

    ax.plot(Q, y_u, color=COLORS["uniform"], linestyle=LS["uniform"],
            linewidth=1.6, label="Uniform", zorder=3)

    ax.axhline(1.0, color=COLORS["uniform"], linestyle=":", alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"Recovery $R(Q)$")

    # Place legend where it doesn't sit on curves (paper-like)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78), frameon=False)

    fig.savefig(outpath)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "exp03_visibility_minvis"))

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)

    ap.add_argument("--train-m", type=int, default=1000)
    ap.add_argument("--holdout-k", type=int, default=20)

    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument(
        "--sigma-grid",
        type=str,
        default="",
        help=(
            "Optional comma-separated grid. If set, choose sigma minimizing |Vis(H_inv)|.\n"
            "NOTE: For n=12, if you do NOT pass --sigma-grid, the script will auto-use 0.5,1,2,3 "
            "to reproduce the paper-style separation (since sigma=1 can make cancellation impossible)."
        ),
    )
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--score-level", type=int, default=7)

    ap.add_argument(
        "--improve-steps",
        type=int,
        default=2000,
        help="Local swap steps to further reduce |Vis| for H_inv (default: 2000).",
    )

    ap.add_argument("--Qmax", type=int, default=10000)

    args = ap.parse_args()

    set_style(base=8)
    outdir = ensure_outdir(args.outdir)

    n = int(args.n)
    N = 2 ** n

    bits_table = make_bits_table(n)
    p_star, support, scores = build_target_distribution(n, float(args.beta))
    good_mask = topk_mask(scores, support, frac=float(args.good_frac))

    s_int = scores.astype(int)
    requested_level = int(args.score_level)

    def pool_for_level(level: int) -> np.ndarray:
        return np.where(good_mask & (s_int == int(level)))[0]

    cand = pool_for_level(requested_level)
    if cand.size < 2 * int(args.holdout_k):
        levels = sorted(set(s_int[good_mask].tolist()))
        viable = [lvl for lvl in levels if pool_for_level(lvl).size >= 2 * int(args.holdout_k)]
        if not viable:
            raise SystemExit(
                f"No score level inside G has >= 2*holdout_k={2*int(args.holdout_k)} candidates. "
                "Try smaller --holdout-k or larger n."
            )
        chosen_level = min(viable, key=lambda lv: abs(int(lv) - requested_level))
        cand = pool_for_level(chosen_level)
        print(f"[Pool] score-level {requested_level} too small; using level {chosen_level} (|cand|={cand.size}).")
        score_level = int(chosen_level)
    else:
        score_level = requested_level
        print(f"[Pool] using score-level {score_level} (|cand|={cand.size}).")

    # ----------------------------
    # Sigma selection (FIX for n=12)
    # ----------------------------
    sigma_grid: List[float]
    if str(args.sigma_grid).strip():
        sigma_grid = [float(x.strip()) for x in str(args.sigma_grid).split(",") if x.strip()]
    else:
        # For n=12, sigma=1.0 can yield same-sign cand_vals (no cancellation possible for min-|Vis|).
        # Auto-try the paper grid unless user explicitly sets --sigma-grid.
        if int(n) == 12:
            sigma_grid = [0.5, 1.0, 2.0, 3.0]
            print(f"[Sigma] n=12 and --sigma-grid not set -> auto grid {sigma_grid} (override with --sigma-grid).")
        else:
            sigma_grid = [float(args.sigma)]

    best = None
    for gi, sigma in enumerate(sigma_grid):
        alphas = sample_alphas(n, float(sigma), int(args.K), seed=int(args.seed) + 999 + 17 * gi)
        P = build_parity_matrix(alphas, bits_table)

        # reference moments on the true target (for holdout construction only)
        z_ref = P @ p_star
        r_ref = P.T @ z_ref

        cand_vals = r_ref[cand]

        H_inv, _ = greedy_min_abs_sum(
            cand_idxs=cand,
            cand_vals=cand_vals,
            k=int(args.holdout_k),
            improve_steps=int(args.improve_steps),
            seed=int(args.seed) + 12345 + 31 * gi,
        )

        inv_set = set(int(i) for i in H_inv.tolist())
        remaining = np.array([int(i) for i in cand.tolist() if int(i) not in inv_set], dtype=int)
        if remaining.size >= int(args.holdout_k):
            rem_vals = r_ref[remaining]
            order_vis = np.argsort(-rem_vals)
            H_vis = remaining[order_vis[: int(args.holdout_k)]]
        else:
            order_vis = np.argsort(-cand_vals)
            H_vis = cand[order_vis[: int(args.holdout_k)]]

        Vis_vis_ref = float(np.sum(r_ref[H_vis]) / N)
        Vis_inv_ref = float(np.sum(r_ref[H_inv]) / N)

        obj = abs(Vis_inv_ref)
        tie = abs(Vis_vis_ref)

        if (best is None) or (obj < best["obj"] - 1e-18) or (abs(obj - best["obj"]) <= 1e-18 and tie > best["tie"]):
            best = {
                "sigma": float(sigma),
                "P": P,
                "r_ref": r_ref,
                "H_vis": H_vis,
                "H_inv": H_inv,
                "Vis_vis_ref": Vis_vis_ref,
                "Vis_inv_ref": Vis_inv_ref,
                "obj": obj,
                "tie": tie,
            }

    assert best is not None

    sigma_used = float(best["sigma"])
    P = best["P"]
    H_vis = np.array(best["H_vis"], dtype=int)
    H_inv = np.array(best["H_inv"], dtype=int)
    Vis_vis_ref = float(best["Vis_vis_ref"])
    Vis_inv_ref = float(best["Vis_inv_ref"])

    if len(sigma_grid) > 1:
        print(f"[Sigma] chose sigma={sigma_used} from grid {sigma_grid} to minimize |Vis(H_inv)|={abs(Vis_inv_ref):.3e}")

    pH_vis = float(np.sum(p_star[H_vis]))
    pH_inv = float(np.sum(p_star[H_inv]))

    print(f"[Check] score-level={score_level} => p*(x) constant within pool")
    print(f"[Check] p*(H_vis)={pH_vis:.6e}, p*(H_inv)={pH_inv:.6e} (should match)")
    print(f"[Vis ref] Vis_B(H_vis)={Vis_vis_ref:+.6e}")
    print(f"[Vis ref] Vis_B(H_inv)={Vis_inv_ref:+.6e} (min-|Vis| construction)")

    def completion_for_holdout(holdout_idxs: np.ndarray, seed_offset: int):
        mask = np.zeros(N, dtype=bool)
        mask[holdout_idxs] = True

        # Training distribution = target with holdout removed + renormalized
        p_train = p_star.copy()
        p_train[mask] = 0.0
        p_train /= float(np.sum(p_train))

        idxs_train = sample_indices(p_train, int(args.train_m), seed=int(args.seed) + 7 + seed_offset)
        emp = empirical_dist(idxs_train, N)

        z_train = P @ emp
        r_train = P.T @ z_train
        Vis_train = float(np.sum(r_train[holdout_idxs]) / N)

        q_lin = linear_band_reconstruction(P, z_train, n)
        q_tilde = completion_by_axioms(q_lin)

        qH = float(np.sum(q_tilde[mask]))
        qH_unif = float(np.sum((np.ones(N) / N)[mask]))
        return q_tilde, mask, Vis_train, qH, qH_unif

    q_vis, mask_vis, Vis_vis_train, qH_vis, qH_unif = completion_for_holdout(H_vis, seed_offset=0)
    q_inv, mask_inv, Vis_inv_train, qH_inv, _ = completion_for_holdout(H_inv, seed_offset=123)

    print(f"[Vis train] Vis_B(H_vis)={Vis_vis_train:+.6e}")
    print(f"[Vis train] Vis_B(H_inv)={Vis_inv_train:+.6e}")
    print(f"[Mass] q~(H_vis)={qH_vis:.6e} vs uniform={qH_unif:.6e} (ratio={qH_vis/qH_unif:.2f}x)")
    print(f"[Mass] q~(H_inv)={qH_inv:.6e} vs uniform={qH_unif:.6e} (ratio={qH_inv/qH_unif:.2f}x)")

    Q80_vis = compute_Q80(q_vis, mask_vis, thr=0.8, Qmax=200000)
    Q80_inv = compute_Q80(q_inv, mask_inv, thr=0.8, Qmax=200000)
    Q80_unif = compute_Q80(np.ones(N) / N, mask_vis, thr=0.8, Qmax=200000)
    print(f"[Q80] visible={Q80_vis:.0f} | invisible={Q80_inv:.0f} | uniform={Q80_unif:.0f}")

    save_holdout_list(
        H_vis, bits_table, p_star, scores,
        outdir / "holdout_strings_visible_minvis.txt",
        header="Visible holdout (max +Vis)"
    )
    save_holdout_list(
        H_inv, bits_table, p_star, scores,
        outdir / "holdout_strings_invisible_minvis.txt",
        header="Invisible holdout (min |Vis|)"
    )

    summary = {
        "n": n,
        "N": N,
        "beta": float(args.beta),
        "sigma": float(sigma_used),
        "K": int(args.K),
        "train_m": int(args.train_m),
        "holdout_k": int(args.holdout_k),
        "good_frac": float(args.good_frac),
        "score_level": int(score_level),
        "seed": int(args.seed),
        "improve_steps": int(args.improve_steps),
        "pH_vis": pH_vis,
        "pH_inv": pH_inv,
        "Vis_ref": {"vis": Vis_vis_ref, "inv": Vis_inv_ref},
        "Vis_train": {"vis": Vis_vis_train, "inv": Vis_inv_train},
        "qH_tilde": {"vis": qH_vis, "inv": qH_inv, "unif": qH_unif},
        "qH_ratio": {"vis": qH_vis / qH_unif, "inv": qH_inv / qH_unif},
        "Q80": {"vis": Q80_vis, "inv": Q80_inv, "unif": Q80_unif},
        "H_vis": H_vis.astype(int).tolist(),
        "H_inv": H_inv.astype(int).tolist(),
    }
    with (outdir / "minvis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    outplot = outdir / "6a_adversarial_curves_minvis.pdf"
    plot_adversarial_curves(
        p_star=p_star,
        holdout_vis_mask=mask_vis,
        holdout_inv_mask=mask_inv,
        q_vis=q_vis,
        q_inv=q_inv,
        outpath=outplot,
        Qmax=int(args.Qmax),
    )

    print(f"[Saved] {outplot}")
    print(f"Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
