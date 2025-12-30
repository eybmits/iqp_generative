#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaxCut target-family extension for the "Sampling the Unseen" holdout-recovery pipeline.

This script is intended as a *publishable, standalone* reference implementation of:
  - Target distribution p*_β(x) ∝ exp(β s(x)) restricted to a symmetry sector (default: even parity),
  - Holdout masking p_train(x) ∝ p*_β(x) 1[x ∉ H],
  - Parity-feature MMD training with random Walsh (parity) features,
  - Fit + discovery metrics: D_KL, q(G), diversity U_G(Q), holdout recovery R(Q), Q80, and q(H).

Here we replace the original "longest-zero-run" score with a concrete combinatorial objective:
  - s(x) := MaxCut_G(x) = number of edges cut by assignment x ∈ {0,1}^n.

The rest of the protocol is unchanged.

Dependencies (CPU-only is fine):
  - numpy
  - torch
  - matplotlib (optional, for plots)

Example:
  python maxcut_holdout_iqp_qcbm.py --n 16 --beta 0.6 --L 1 --topology D --target_graph random_regular --d 3 \
      --seeds 3 --outdir results_maxcut

Notes:
  - The ansatz is the IQP-QCBM template U(θ) = H^{⊗n} ∏_{ℓ=1}^L (U_diag^{(ℓ)}(θ_ℓ) H^{⊗n}),
    where each U_diag is a product of commuting Z-type interactions (ZZ / ZZZZ, optionally Z fields).
  - We exploit this structure to simulate by alternating:
      (i) an n-qubit Hadamard transform (FWHT) and
      (ii) elementwise multiplication by a diagonal phase vector.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------
# Matplotlib paper style (matched to the figures in "Sampling the Unseen")
# -----------------------------

def _set_mpl_style_like_sampling_unseen(*, usetex: bool = False) -> None:
    """Set matplotlib rcParams to closely match the paper figure style.

    Notes:
      - We default to mathtext (no LaTeX dependency). If you have LaTeX installed and
        want 1:1 font matching to a LaTeX manuscript, pass usetex=True.
      - We keep everything "paper-safe": vector PDF output, thin axes, small fonts,
        and the same red/black/gray palette used throughout the manuscript.
    """
    if plt is None:  # pragma: no cover
        return

    import matplotlib as mpl

    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        # Try to match LaTeX/Computer-Modern look even when usetex=False.
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "mathtext.fontset": "cm",
        # Layout / strokes
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 3.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        # Save as vector-friendly
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "text.usetex": bool(usetex),
    })


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance TVD(p,q) = 1/2 ||p-q||_1."""
    q2 = q.astype(np.float64, copy=False)
    q2 = q2 / q2.sum()
    return float(0.5 * np.sum(np.abs(p - q2)))


def _paper_figure_maxcut_extension(
    *,
    cfg: "TrainConfig",
    outdir: Path,
    scores: np.ndarray,
    p_star: np.ndarray,
    holdout: np.ndarray,
    q_model: np.ndarray,
    q_recon: np.ndarray,
    rep_metrics: Dict[str, float],
    usetex: bool = False,
) -> None:
    """Create a single, paper-ready figure (PDF + PNG) for the MaxCut extension.

    The figure is deliberately formatted to match the style of the plots in
    "Sampling the Unseen" (same palette, typography, and panel labeling).

    Panel (a): score spectrum (probability mass aggregated by MaxCut score).
    Panel (b): holdout recovery curves R(Q) (log-Q axis) with Q80 inset.
    """
    if plt is None:  # pragma: no cover
        return

    _set_mpl_style_like_sampling_unseen(usetex=usetex)

    # ---------- Panel (a): score spectrum ----------
    max_score = int(scores.max())
    xs = np.arange(max_score + 1)

    # Aggregate probability mass by score value.
    # (np.bincount expects nonnegative ints.)
    scores_int = scores.astype(np.int64)
    spec_p = np.bincount(scores_int, weights=p_star, minlength=max_score + 1)
    spec_q = np.bincount(scores_int, weights=q_model, minlength=max_score + 1)

    tvd_val = _tvd(p_star, q_model)

    # ---------- Panel (b): holdout recovery curves ----------
    # Pick a log-spaced range that safely covers typical Q80 for n=16.
    # (For uniform, Q80 ≈ 2^n ln 5 ≈ 1.05e5 at n=16.)
    Qs = np.unique(np.round(np.logspace(0, 6, 220)).astype(int))  # 1..1e6
    R_target = np.array([recovery_fraction(p_star, holdout, int(Q)) for Q in Qs], dtype=np.float64)
    R_model = np.array([recovery_fraction(q_model, holdout, int(Q)) for Q in Qs], dtype=np.float64)
    R_recon = np.array([recovery_fraction(q_recon, holdout, int(Q)) for Q in Qs], dtype=np.float64)

    # Uniform baseline is independent of H.
    p_unif = 1.0 / float(1 << cfg.n)
    R_unif = 1.0 - np.exp(Qs.astype(np.float64) * np.log1p(-p_unif))

    Q80_target = find_Q80(p_star, holdout, target=cfg.Q80_target, Q_max=cfg.Q80_max)
    Q80_model = int(rep_metrics.get("Q80_measured", find_Q80(q_model, holdout, target=cfg.Q80_target, Q_max=cfg.Q80_max)))
    Q80_recon = find_Q80(q_recon, holdout, target=cfg.Q80_target, Q_max=cfg.Q80_max)
    # Closed form for uniform baseline Q80.
    Q80_unif = int(math.ceil(math.log(1.0 - cfg.Q80_target) / math.log1p(-p_unif)))

    # ---------- Plot layout ----------
    # Match the paper's "small multipanel" style: two compact panels side-by-side.
    fig = plt.figure(figsize=(6.8, 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Panel labels (a), (b)
    ax1.text(-0.12, 1.02, "a", transform=ax1.transAxes, ha="left", va="bottom")
    ax2.text(-0.12, 1.02, "b", transform=ax2.transAxes, ha="left", va="bottom")

    # ---- (a) Score spectrum bars ----
    ax1.bar(
        xs,
        spec_p,
        width=0.85,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
        label=r"Target $p^*$",
        zorder=1,
    )
    ax1.bar(
        xs,
        spec_q,
        width=0.55,
        color="tab:red",
        edgecolor="black",
        linewidth=0.8,
        label=r"Model $q_{\theta}$",
        zorder=2,
    )
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Probability Mass")
    ax1.set_xlim(-0.5, max_score + 0.5)
    ax1.set_ylim(0.0, max(float(spec_p.max()), float(spec_q.max())) * 1.10)

    ax1.text(
        0.02,
        0.96,
        f"TVD = {tvd_val:.3f}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", linewidth=0.6, alpha=0.9),
    )
    ax1.legend(loc="upper right", handlelength=1.6)

    # ---- (b) Recovery curves ----
    ax2.plot(Qs, R_target, color="black", label=r"Target $p^*$")
    ax2.plot(Qs, R_model, color="tab:red", label=r"IQP-QCBM $q_{\theta}$")
    ax2.plot(Qs, R_recon, color="black", linestyle="-.", label=r"Spectral recon $\tilde q$")
    ax2.plot(Qs, R_unif, color="0.4", linestyle="--", label="Uniform")

    ax2.set_xscale("log")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xlabel(r"Sampling Budget $Q$ (log)")
    ax2.set_ylabel(r"Recovery $R(Q)$")
    ax2.legend(loc="lower right", handlelength=2.2)

    inset = "\n".join([
        r"$Q_{80}$ (80% recovery):",
        f"Target:  {Q80_target}",
        f"Model:   {Q80_model}",
        f"Recon:   {Q80_recon}",
        f"Uniform: {Q80_unif}",
    ])
    ax2.text(
        0.02,
        0.98,
        inset,
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", linewidth=0.6, alpha=0.9),
    )

    # Export
    figpath_pdf = outdir / "paper_maxcut_extension.pdf"
    figpath_png = outdir / "paper_maxcut_extension.png"
    fig.savefig(figpath_pdf)
    fig.savefig(figpath_png, dpi=300)
    plt.close(fig)


# -----------------------------
# Bit / Hamming utilities
# -----------------------------

def popcount_uint32(x: np.ndarray) -> np.ndarray:
    """
    Vectorized popcount for uint32 arrays (Hacker's Delight).
    Returns uint32 counts in [0,32] with same shape as x.
    """
    x = x.astype(np.uint32, copy=False)
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    x = (x + (x >> 4)) & np.uint32(0x0F0F0F0F)
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & np.uint32(0x0000003F)


def parity_uint32(x: np.ndarray) -> np.ndarray:
    """
    Vectorized parity (popcount mod 2) for uint32 arrays.
    Returns uint32 in {0,1}.
    """
    x = x.astype(np.uint32, copy=False)
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x & np.uint32(0xF)
    # 0x6996 is a 16-bit lookup table for parity of 0..15
    return (np.uint32(0x6996) >> x) & np.uint32(1)


def int_to_bitstring(x: int, n: int) -> str:
    return format(x, f"0{n}b")


# -----------------------------
# Graph utilities (target MaxCut)
# -----------------------------

def random_d_regular_graph(n: int, d: int, rng: np.random.Generator, max_tries: int = 10_000) -> List[Tuple[int, int]]:
    """
    Simple rejection-sampling configuration-model generator for a simple undirected d-regular graph.
    n must satisfy n*d even and (typically) n > d.

    Returns list of edges (i,j) with i<j.
    """
    if (n * d) % 2 != 0:
        raise ValueError("Need n*d even for d-regular graph.")

    if d >= n:
        raise ValueError("Need d < n.")

    stubs = np.repeat(np.arange(n, dtype=np.int32), d)

    for _ in range(max_tries):
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue  # self-loop

        edges = np.sort(pairs, axis=1)
        # Check duplicates
        edges_view = edges[:, 0].astype(np.int64) * n + edges[:, 1].astype(np.int64)
        if len(np.unique(edges_view)) != len(edges_view):
            continue

        # Looks like a simple graph
        return [(int(i), int(j)) for i, j in edges]

    raise RuntimeError("Failed to sample a simple d-regular graph; try increasing max_tries or changing seed.")


def grid_graph_2d(m: int, n: int) -> List[Tuple[int, int]]:
    """m x n 2D grid graph with vertices indexed row-major, edges undirected."""
    edges: List[Tuple[int, int]] = []
    def vid(r: int, c: int) -> int:
        return r * n + c
    for r in range(m):
        for c in range(n):
            v = vid(r, c)
            if r + 1 < m:
                edges.append((v, vid(r + 1, c)))
            if c + 1 < n:
                edges.append((v, vid(r, c + 1)))
    return [(min(i, j), max(i, j)) for i, j in edges]


def maxcut_scores_all_states(n: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Compute MaxCut score for every x ∈ {0,1}^n (as integer 0..2^n-1).
    score(x) = # edges (i,j) with x_i XOR x_j = 1.

    Returns int16 array of length 2^n.
    """
    N = 1 << n
    states = np.arange(N, dtype=np.uint32)
    score = np.zeros(N, dtype=np.int16)

    for i, j in edges:
        bi = (states >> np.uint32(i)) & np.uint32(1)
        bj = (states >> np.uint32(j)) & np.uint32(1)
        score += (bi ^ bj).astype(np.int16)

    return score


# -----------------------------
# Target distribution + holdout protocol
# -----------------------------

def even_parity_mask(n: int) -> np.ndarray:
    """Boolean mask over states x∈{0,1}^n indicating even Hamming weight."""
    N = 1 << n
    states = np.arange(N, dtype=np.uint32)
    # parity of popcount
    par = parity_uint32(states)
    return (par == 0)


def make_target_distribution(
    scores: np.ndarray,
    beta: float,
    sector_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    p*_β(x) ∝ exp(beta * score(x)) on the chosen sector, else 0.
    Returns float64 array of length 2^n normalized to sum to 1.
    """
    scores = scores.astype(np.float64)
    if sector_mask is None:
        sector_mask = np.ones_like(scores, dtype=bool)

    logits = np.full_like(scores, -np.inf, dtype=np.float64)
    logits[sector_mask] = beta * scores[sector_mask]

    # Stable softmax over sector
    max_logit = np.max(logits[sector_mask])
    unnorm = np.zeros_like(scores, dtype=np.float64)
    unnorm[sector_mask] = np.exp(logits[sector_mask] - max_logit)
    Z = unnorm.sum()
    if not np.isfinite(Z) or Z <= 0:
        raise RuntimeError("Target distribution normalization failed.")
    return unnorm / Z


def top_percent_set(
    scores: np.ndarray,
    sector_mask: np.ndarray,
    top_frac: float = 0.05,
) -> np.ndarray:
    """
    Return integer array of states in the top `top_frac` fraction of the sector by score.
    Ties broken deterministically by smaller integer state index first.
    """
    candidates = np.where(sector_mask)[0].astype(np.int64)
    sc = scores[candidates]
    # Sort by (-score, state)
    order = np.lexsort((candidates, -sc))
    sorted_states = candidates[order]
    k = max(1, int(math.ceil(top_frac * len(sorted_states))))
    return sorted_states[:k]


def select_holdout_set(
    G_states: np.ndarray,
    p_star: np.ndarray,
    n: int,
    m_train: int,
    holdout_size: int = 20,
    topM: int = 400,
) -> np.ndarray:
    """
    Greedy heuristic matching Appendix B:
      1) Restrict to G; apply probability floor τ from a schedule starting at ~1/m until at least |H| remain.
      2) Keep top M candidates by p*.
      3) Initialize H with highest-p* candidate.
      4) Add candidates by farthest-point sampling in Hamming distance; ties by larger p*.
    """
    if holdout_size > len(G_states):
        raise ValueError("Holdout size larger than |G|.")

    # 1) probability floor schedule (start at 1/m, then relax)
    tau0 = 1.0 / float(m_train)
    tau_schedule = [tau0, 0.5 * tau0, 0.2 * tau0, 0.1 * tau0, 0.05 * tau0, 0.02 * tau0, 0.01 * tau0, 0.005 * tau0, 0.002 * tau0, 0.001 * tau0, 0.0]

    candidates = None
    for tau in tau_schedule:
        cand = G_states[p_star[G_states] >= tau]
        if len(cand) >= holdout_size:
            candidates = cand
            break
    if candidates is None:
        candidates = G_states.copy()

    # 2) keep top M by p*
    cand_p = p_star[candidates]
    order = np.lexsort((candidates, -cand_p))  # (-p, state)
    candidates = candidates[order]
    if len(candidates) > topM:
        candidates = candidates[:topM]

    # 3) init with highest probability
    H: List[int] = [int(candidates[0])]
    remaining = candidates[1:].astype(np.uint32)

    # 4) farthest-point sampling
    while len(H) < holdout_size:
        chosen = np.array(H, dtype=np.uint32)  # shape (h,)
        # Distances to chosen: popcount(xor)
        xor = remaining[:, None] ^ chosen[None, :]
        dists = popcount_uint32(xor)  # (r,h)
        min_dist = dists.min(axis=1).astype(np.int32)

        best_min = int(min_dist.max())
        idxs = np.where(min_dist == best_min)[0]
        if len(idxs) == 1:
            pick_idx = int(idxs[0])
        else:
            # tie-break by larger p*, then smaller state index
            tied_states = remaining[idxs].astype(np.int64)
            tied_p = p_star[tied_states]
            # primary: -p, secondary: state
            torder = np.lexsort((tied_states, -tied_p))
            pick_idx = int(idxs[torder[0]])

        pick_state = int(remaining[pick_idx])
        H.append(pick_state)
        remaining = np.delete(remaining, pick_idx, axis=0)

    return np.array(H, dtype=np.int64)


def masked_training_distribution(p_star: np.ndarray, holdout: np.ndarray) -> np.ndarray:
    """p_train ∝ p_star with holdout states removed (probability set to 0)."""
    p_train = p_star.copy()
    p_train[holdout] = 0.0
    Z = p_train.sum()
    if Z <= 0:
        raise RuntimeError("p_train normalization failed (all mass removed?)")
    return p_train / Z


# -----------------------------
# Random parity features + moments
# -----------------------------

def bernoulli_masks(n: int, K: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample K random parity masks α ∈ {0,1}^n with α_i ~ Bernoulli(p(σ)),
    where p(σ)=1/2*(1-exp(-1/(2σ^2))).

    Returns uint32 array of length K, each entry encodes a bitmask.
    """
    p = 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma * sigma)))
    bits = rng.binomial(1, p, size=(K, n)).astype(np.uint32)  # (K,n)
    powers = (np.uint32(1) << np.arange(n, dtype=np.uint32))[None, :]  # (1,n)
    masks = (bits * powers).sum(axis=1).astype(np.uint32)
    return masks


def parity_features_from_samples(samples: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Compute empirical moments z_k = E[(-1)^{α_k·x}] from integer samples x and mask integers α_k.
    samples: shape (m,), uint32
    masks: shape (K,), uint32
    Returns float32 array shape (K,).
    """
    samples = samples.astype(np.uint32, copy=False)
    masks = masks.astype(np.uint32, copy=False)

    # v_ij = samples_i & masks_j, then parity(popcount(v_ij)) gives α·x mod 2
    v = np.bitwise_and(samples[:, None], masks[None, :])  # (m,K) uint32
    par = parity_uint32(v).astype(np.float32)  # (m,K) in {0,1}
    phi = 1.0 - 2.0 * par  # in {+1,-1}
    return phi.mean(axis=0).astype(np.float32)


# -----------------------------
# IQP-QCBM ansatz (torch simulation)
# -----------------------------

@dataclass(frozen=True)
class AnsatzTerms:
    """
    Stores the diagonal Z-type terms for one IQP layer.
    Each term is represented as a tuple of qubit indices; length 1 => Z_i, length 2 => Z_i Z_j, length 4 => Z_i Z_j Z_k Z_l.
    """
    terms: Tuple[Tuple[int, ...], ...]

    @property
    def num_terms(self) -> int:
        return len(self.terms)


def build_ansatz_terms(n: int, topology: str, include_z_fields: bool = False) -> AnsatzTerms:
    """
    Implements the five topologies A–E (per-layer interaction sets).

    A: ring NN ZZ  -> n terms
    B: A + ring ZZZZ -> 2n terms
    C: ring NN + NNN ZZ -> 2n terms
    D: C + ring ZZZZ -> 3n terms
    E: all-to-all ZZ -> n*(n-1)/2 terms

    If include_z_fields is True, adds n single-qubit Z_i terms (odd-weight) to each layer.
    """
    topo = topology.upper()
    terms: List[Tuple[int, ...]] = []

    def add_ring_zz(step: int) -> None:
        for i in range(n):
            j = (i + step) % n
            a, b = (i, j) if i < j else (j, i)
            terms.append((a, b))

    def add_ring_zzzz() -> None:
        # ring-local 4-body terms on consecutive quadruples (i,i+1,i+2,i+3) mod n
        for i in range(n):
            quad = tuple(sorted(((i + k) % n) for k in range(4)))
            terms.append(quad)

    if topo == "A":
        add_ring_zz(step=1)
    elif topo == "B":
        add_ring_zz(step=1)
        add_ring_zzzz()
    elif topo == "C":
        add_ring_zz(step=1)
        add_ring_zz(step=2)
    elif topo == "D":
        add_ring_zz(step=1)
        add_ring_zz(step=2)
        add_ring_zzzz()
    elif topo == "E":
        for i in range(n):
            for j in range(i + 1, n):
                terms.append((i, j))
    else:
        raise ValueError(f"Unknown topology '{topology}'. Use A,B,C,D,E.")

    if include_z_fields:
        for i in range(n):
            terms.append((i,))

    # Deduplicate terms (can happen for small n)
    terms = sorted(set(terms))
    return AnsatzTerms(terms=tuple(terms))


def precompute_z_eigenvalues(n: int) -> np.ndarray:
    """
    Precompute z_i(x) = (-1)^{x_i} for all i and all basis states x.
    Returns array shape (n, 2^n) float32.
    """
    N = 1 << n
    states = np.arange(N, dtype=np.uint32)
    bits = ((states[None, :] >> np.arange(n, dtype=np.uint32)[:, None]) & np.uint32(1)).astype(np.float32)  # (n,N)
    z = 1.0 - 2.0 * bits  # +1 for bit=0, -1 for bit=1
    return z.astype(np.float32)


def build_eigenvalue_matrix(n: int, terms: AnsatzTerms, z_cache: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build eigenvalue matrix E_terms where each row corresponds to a diagonal term's eigenvalue (+/-1)
    over all basis states x. Shape: (num_terms, 2^n).
    """
    if z_cache is None:
        z_cache = precompute_z_eigenvalues(n)
    N = 1 << n
    ev = np.empty((terms.num_terms, N), dtype=np.float32)
    for t_idx, qubits in enumerate(terms.terms):
        prod = np.ones(N, dtype=np.float32)
        for q in qubits:
            prod *= z_cache[q]
        ev[t_idx] = prod
    return ev


def fwht_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalized fast Walsh–Hadamard transform (FWHT) for 1D tensor of length 2^n.
    Works for real or complex tensors and is differentiable.
    """
    y = x
    N = y.shape[0]
    h = 1
    while h < N:
        y = y.reshape(-1, 2 * h)
        a = y[:, :h]
        b = y[:, h:]
        y = torch.cat((a + b, a - b), dim=1)
        y = y.reshape(N)
        h *= 2
    return y


def hadamard_all_qubits(state: torch.Tensor) -> torch.Tensor:
    """Apply H^{⊗n} to a statevector via FWHT with normalization 1/sqrt(N)."""
    N = state.shape[0]
    return fwht_torch(state) / math.sqrt(N)


def iqp_statevector(
    theta: torch.Tensor,          # (L, num_terms), real
    eigenvalues: torch.Tensor,     # (num_terms, N), real, +/-1
) -> torch.Tensor:
    """
    Compute |ψ(θ)⟩ = U(θ)|0^n⟩ for IQP-QCBM template using alternating H^{⊗n} and diagonal phases.

    theta: (L, T) parameters
    eigenvalues: (T, N) eigenvalue matrix for diagonal terms
    Returns complex statevector of length N.
    """
    device = theta.device
    dtype_c = torch.complex64 if theta.dtype == torch.float32 else torch.complex128

    T, N = eigenvalues.shape
    L = theta.shape[0]
    # |0...0>
    state = torch.zeros(N, dtype=dtype_c, device=device)
    state[0] = 1.0 + 0.0j

    # initial H^{⊗n}
    state = hadamard_all_qubits(state)

    # layers
    for l in range(L):
        # diag_arg(x) = sum_t theta[l,t] * eigenvalue_t(x)
        diag_arg = torch.matmul(theta[l], eigenvalues)  # (N,) real
        phase = torch.exp((-0.5j) * diag_arg.to(dtype_c))
        state = state * phase
        state = hadamard_all_qubits(state)

    return state


# -----------------------------
# Metrics
# -----------------------------

def clip_and_renorm(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q2 = np.maximum(q, eps)
    q2 = q2 / q2.sum()
    return q2


def dkl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """DKL(p||q) with clipping/renormalization of q for numerical stability."""
    q2 = clip_and_renorm(q, eps=eps)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q2[mask]))))


def expected_unique(r: np.ndarray, states: np.ndarray, Q: int) -> float:
    """
    U_Ω(r)(Q) = sum_{x in Ω} [1 - (1 - r(x))^Q].
    Computed stably via exp(Q * log1p(-r)).
    """
    probs = r[states]
    # clip for numerical stability: log1p(-p) requires p<=1
    probs = np.clip(probs, 0.0, 1.0)
    return float(np.sum(1.0 - np.exp(Q * np.log1p(-probs))))


def recovery_fraction(q: np.ndarray, holdout: np.ndarray, Q: int) -> float:
    return expected_unique(q, holdout, Q) / float(len(holdout))


def find_Q80(q: np.ndarray, holdout: np.ndarray, target: float = 0.8, Q_max: int = 2_000_000) -> int:
    """Binary search minimal integer Q such that R(Q) >= target. Returns Q_max if never reaches target."""
    low, high = 1, 1
    while high < Q_max and recovery_fraction(q, holdout, high) < target:
        high *= 2
    high = min(high, Q_max)
    if recovery_fraction(q, holdout, high) < target:
        return Q_max

    # binary search in [low, high]
    low = high // 2
    while low + 1 < high:
        mid = (low + high) // 2
        if recovery_fraction(q, holdout, mid) >= target:
            high = mid
        else:
            low = mid
    return high


# -----------------------------
# Circuit-free reconstruction baseline (optional)
# -----------------------------

def reconstruction_from_moments(n: int, masks: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Circuit-free partial Walsh inverse:
      q~(x) = 2^{-n} ( 1 + sum_k z_k * phi_{alpha_k}(x) )
    Implemented efficiently via FWHT on the coefficient vector a(alpha).
    Then project to a valid distribution by clipping negatives and renormalizing.
    """
    N = 1 << n
    coeff = np.zeros(N, dtype=np.float64)
    coeff[0] = 1.0
    # If duplicate masks occur, we accumulate.
    for m, zk in zip(masks.astype(np.int64), z.astype(np.float64)):
        coeff[m] += float(zk)

    # FWHT in numpy (unnormalized), then divide by 2^n
    f = coeff.copy()
    h = 1
    while h < N:
        f = f.reshape(-1, 2 * h)
        a = f[:, :h]
        b = f[:, h:]
        f = np.concatenate([a + b, a - b], axis=1)
        f = f.reshape(N)
        h *= 2

    q_tilde = f / float(N)
    q_proj = np.maximum(q_tilde, 0.0)
    Z = q_proj.sum()
    if Z <= 0:
        # fallback: uniform
        q_proj = np.ones(N, dtype=np.float64) / float(N)
    else:
        q_proj /= Z
    return q_proj


# -----------------------------
# Training loop (Adam)
# -----------------------------

@dataclass
class TrainConfig:
    n: int = 16
    beta: float = 0.9
    L: int = 1
    topology: str = "D"          # A,B,C,D,E
    parity_fix: bool = False     # add Z-fields + include global parity feature
    sector: str = "even"         # "even" or "none"

    # Target MaxCut graph
    target_graph: str = "random_regular"   # random_regular | grid
    d: int = 3                     # degree for random_regular
    grid_rows: int = 4
    grid_cols: int = 4

    # Holdout protocol
    holdout_size: int = 20
    top_frac_G: float = 0.05

    # Random parity features / MMD training
    m_train: int = 1000
    K: int = 512
    sigma: float = 1.0

    # Optimizer
    lr: float = 0.05
    steps: int = 600
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Evaluation
    Q_diversity: int = 5000
    Q80_target: float = 0.8
    Q80_max: int = 2_000_000

    # Seeds
    seeds: int = 5
    seed_holdout: int = 0
    seed_base: int = 1

    # Output
    outdir: str = "results"
    make_plots: bool = True
    paper_usetex: bool = False  # enable LaTeX rendering in matplotlib (for 1:1 manuscript fonts)
    plot_seed: str = "median_dkl"  # seed used for paper plots: seed0 | median_dkl | best_dkl
    device: str = "cpu"          # cpu | cuda (if available)
    dtype: str = "float32"       # float32 | float64


def train_one_seed(
    cfg: TrainConfig,
    p_train: np.ndarray,
    p_star: np.ndarray,
    scores: np.ndarray,
    G_states: np.ndarray,
    holdout: np.ndarray,
    eigenvalues: torch.Tensor,
    terms: AnsatzTerms,
    run_seed: int,
    outdir: Path,
) -> Dict[str, float]:
    """
    Train one IQP-QCBM instance (one seed) and compute metrics.
    """
    rng = np.random.default_rng(cfg.seed_base + run_seed)

    # Sample training data
    N = len(p_train)
    train_samples = rng.choice(np.arange(N, dtype=np.int64), size=cfg.m_train, replace=True, p=p_train).astype(np.uint32)

    # Sample parity masks
    masks = bernoulli_masks(cfg.n, cfg.K, cfg.sigma, rng)

    # Parity fix: include global parity feature mask (all ones) among masks
    if cfg.parity_fix:
        parity_mask = np.uint32((1 << cfg.n) - 1)
        masks[0] = parity_mask  # overwrite first feature deterministically

    # Compute empirical training moments z(p_train)
    z_train = parity_features_from_samples(train_samples, masks)  # (K,)

    # Torch tensors
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    theta = torch.tensor(
        rng.uniform(low=-math.pi, high=math.pi, size=(cfg.L, terms.num_terms)).astype(np.float32 if cfg.dtype == "float32" else np.float64),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)  # moments in float32 for loss

    # Adam state
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)

    losses: List[float] = []

    def loss_fn(th: torch.Tensor) -> torch.Tensor:
        state = iqp_statevector(th, eigenvalues)  # (N,) complex
        q = (state.real ** 2 + state.imag ** 2)  # (N,) real
        w = fwht_torch(q)  # unnormalized Walsh moments for all frequencies
        # Gather enforced moments
        idx = torch.tensor(masks.astype(np.int64), dtype=torch.long, device=device)
        z_q = w.index_select(0, idx).to(torch.float32)
        return torch.mean((z_q - z_train_t) ** 2)

    for t in range(1, cfg.steps + 1):
        loss = loss_fn(theta)
        loss.backward()

        with torch.no_grad():
            g = theta.grad
            m.mul_(cfg.adam_beta1).add_(g, alpha=1.0 - cfg.adam_beta1)
            v.mul_(cfg.adam_beta2).addcmul_(g, g, value=1.0 - cfg.adam_beta2)

            m_hat = m / (1.0 - cfg.adam_beta1 ** t)
            v_hat = v / (1.0 - cfg.adam_beta2 ** t)

            theta.addcdiv_(m_hat, torch.sqrt(v_hat).add_(cfg.adam_eps), value=-cfg.lr)

        theta.grad.zero_()
        if t == 1 or t % 10 == 0 or t == cfg.steps:
            losses.append(float(loss.detach().cpu().item()))

    # Final distribution q_theta
    with torch.no_grad():
        state = iqp_statevector(theta, eigenvalues)
        q = (state.real ** 2 + state.imag ** 2).detach().cpu().numpy().astype(np.float64)

    # Metrics
    eps = 1e-12
    dkl_val = dkl(p_star, q, eps=eps)
    qG = float(q[G_states].sum())
    qH = float(q[holdout].sum())
    UG = expected_unique(q, G_states, cfg.Q_diversity)
    RH_1000 = recovery_fraction(q, holdout, 1000)
    Q80_measured = find_Q80(q, holdout, target=cfg.Q80_target, Q_max=cfg.Q80_max)
    Q80_pred = float(len(holdout) / max(qH, 1e-18) * math.log(5.0))

    # Optional: circuit-free reconstruction baseline
    q_recon = reconstruction_from_moments(cfg.n, masks, z_train)
    Q80_recon = find_Q80(q_recon, holdout, target=cfg.Q80_target, Q_max=cfg.Q80_max)
    qH_recon = float(q_recon[holdout].sum())

    metrics = {
        "seed": int(run_seed),
        "dkl_pstar_q": float(dkl_val),
        "qG": float(qG),
        "qH": float(qH),
        "qH_over_qunif": float(qH / (len(holdout) / float(1 << cfg.n))),
        "UG_Qdiv": float(UG),
        "RH_1000": float(RH_1000),
        "Q80_measured": int(Q80_measured),
        "Q80_pred_ln5": float(Q80_pred),
        "Q80_recon": int(Q80_recon),
        "qH_recon": float(qH_recon),
        "final_loss_logged": float(losses[-1]) if len(losses) else float("nan"),
        "num_terms": int(terms.num_terms),
    }

    # Save per-seed artifacts
    seed_dir = outdir / f"seed_{run_seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    torch.save(theta.detach().cpu(), seed_dir / "theta.pt")
    np.save(seed_dir / "masks.npy", masks.astype(np.uint32))
    np.save(seed_dir / "z_train.npy", z_train.astype(np.float32))
    np.save(seed_dir / "q.npy", q.astype(np.float64))
    np.save(seed_dir / "losses_logged.npy", np.array(losses, dtype=np.float64))
    with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# -----------------------------
# Main
# -----------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="MaxCut target-family extension for IQP-QCBM holdout recovery.")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--L", type=int, default=1)
    p.add_argument("--topology", type=str, default="D", choices=list("ABCDE"))
    p.add_argument("--parity_fix", action="store_true", help="Add Z-fields and include global parity feature (recommended for L even).")
    p.add_argument("--sector", type=str, default="even", choices=["even", "none"])

    p.add_argument("--target_graph", type=str, default="random_regular", choices=["random_regular", "grid"])
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--grid_rows", type=int, default=4)
    p.add_argument("--grid_cols", type=int, default=4)

    p.add_argument("--holdout_size", type=int, default=20)
    p.add_argument("--m_train", type=int, default=1000)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--sigma", type=float, default=1.0)

    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=600)

    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed_holdout", type=int, default=0)
    p.add_argument("--seed_base", type=int, default=1)

    p.add_argument("--Q_diversity", type=int, default=5000)
    p.add_argument("--outdir", type=str, default="results_maxcut")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--paper_usetex", action="store_true", help="Use LaTeX for matplotlib text (1:1 manuscript fonts).")
    p.add_argument("--plot_seed", type=str, default="median_dkl", choices=["seed0", "median_dkl", "best_dkl"],
                   help="Which seed run to visualize in the paper-style plot.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    args = p.parse_args()
    cfg = TrainConfig(
        n=args.n,
        beta=args.beta,
        L=args.L,
        topology=args.topology,
        parity_fix=bool(args.parity_fix),
        sector=args.sector,
        target_graph=args.target_graph,
        d=args.d,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        holdout_size=args.holdout_size,
        m_train=args.m_train,
        K=args.K,
        sigma=args.sigma,
        lr=args.lr,
        steps=args.steps,
        seeds=args.seeds,
        seed_holdout=args.seed_holdout,
        seed_base=args.seed_base,
        Q_diversity=args.Q_diversity,
        outdir=args.outdir,
        make_plots=(not args.no_plots),
        paper_usetex=bool(args.paper_usetex),
        plot_seed=str(args.plot_seed),
        device=args.device,
        dtype=args.dtype,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Build target MaxCut graph and scores
    rng_holdout = np.random.default_rng(cfg.seed_holdout)

    if cfg.target_graph == "random_regular":
        edges = random_d_regular_graph(cfg.n, cfg.d, rng_holdout)
    elif cfg.target_graph == "grid":
        if cfg.grid_rows * cfg.grid_cols != cfg.n:
            raise ValueError("For grid target_graph, need grid_rows*grid_cols == n.")
        edges = grid_graph_2d(cfg.grid_rows, cfg.grid_cols)
    else:
        raise ValueError("Unknown target_graph")

    scores = maxcut_scores_all_states(cfg.n, edges)  # (2^n,)

    # 2) Sector restriction
    if cfg.sector == "even":
        sector_mask = even_parity_mask(cfg.n)
    else:
        sector_mask = np.ones(1 << cfg.n, dtype=bool)

    # 3) Target distribution p*_β and good set G (top 5% by score within sector)
    p_star = make_target_distribution(scores, cfg.beta, sector_mask=sector_mask)
    G_states = top_percent_set(scores, sector_mask, top_frac=cfg.top_frac_G)

    # 4) Holdout selection H ⊆ G
    holdout = select_holdout_set(
        G_states=G_states,
        p_star=p_star,
        n=cfg.n,
        m_train=cfg.m_train,
        holdout_size=cfg.holdout_size,
        topM=400,
    )

    # 5) Masked training distribution p_train
    p_train = masked_training_distribution(p_star, holdout)

    # Save target + holdout metadata
    meta = {
        "config": dataclasses.asdict(cfg),
        "target_edges": edges,
        "num_edges_target_graph": len(edges),
        "G_size": int(len(G_states)),
        "H_size": int(len(holdout)),
        "p_star_G_mass": float(p_star[G_states].sum()),
        "p_star_H_mass": float(p_star[holdout].sum()),
    }
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Human-readable holdout list
    with open(outdir / "holdout.txt", "w", encoding="utf-8") as f:
        f.write("# holdout states (integer, bitstring, score, p_star)\n")
        for x in holdout:
            f.write(f"{int(x)}\t{int_to_bitstring(int(x), cfg.n)}\t{int(scores[int(x)])}\t{p_star[int(x)]:.6e}\n")

    # 6) Build ansatz terms and eigenvalues for simulation
    terms = build_ansatz_terms(cfg.n, cfg.topology, include_z_fields=cfg.parity_fix)
    z_cache = precompute_z_eigenvalues(cfg.n)
    eigen_np = build_eigenvalue_matrix(cfg.n, terms, z_cache=z_cache)  # (T,N)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    eigenvalues = torch.tensor(eigen_np, dtype=dtype, device=device)  # constant

    # 7) Train across seeds
    all_metrics: List[Dict[str, float]] = []
    for s in range(cfg.seeds):
        metrics = train_one_seed(
            cfg=cfg,
            p_train=p_train,
            p_star=p_star,
            scores=scores,
            G_states=G_states,
            holdout=holdout,
            eigenvalues=eigenvalues,
            terms=terms,
            run_seed=s,
            outdir=outdir,
        )
        all_metrics.append(metrics)
        print(f"[seed {s}] DKL={metrics['dkl_pstar_q']:.3f}  q(G)={metrics['qG']:.3f}  q(H)={metrics['qH']:.4e}  Q80={metrics['Q80_measured']}")

    # 8) Aggregate + plots
    with open(outdir / "metrics_all.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Mean/std summary
    def mean_std(key: str) -> Tuple[float, float]:
        vals = np.array([m[key] for m in all_metrics], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    summary = {}
    for key in ["dkl_pstar_q", "qG", "qH", "qH_over_qunif", "UG_Qdiv", "Q80_measured", "Q80_pred_ln5", "Q80_recon", "qH_recon"]:
        mu, sd = mean_std(key)
        summary[key + "_mean"] = mu
        summary[key + "_std"] = sd

    with open(outdir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (mean ± std) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if cfg.make_plots and plt is not None:
        # Pick a representative seed for plotting (default: median-DKL seed)
        dkl_vals = np.array([m["dkl_pstar_q"] for m in all_metrics], dtype=np.float64)
        if cfg.plot_seed == "seed0":
            rep_seed = 0
        elif cfg.plot_seed == "best_dkl":
            rep_seed = int(np.argmin(dkl_vals))
        else:  # "median_dkl"
            med = float(np.median(dkl_vals))
            rep_seed = int(np.argmin(np.abs(dkl_vals - med)))

        rep_metrics = next(m for m in all_metrics if int(m["seed"]) == rep_seed)

        seed_dir = outdir / f"seed_{rep_seed:02d}"
        q0 = np.load(seed_dir / "q.npy")
        masks0 = np.load(seed_dir / "masks.npy")
        ztrain0 = np.load(seed_dir / "z_train.npy")
        q_recon0 = reconstruction_from_moments(cfg.n, masks0, ztrain0)

        # Match the plot style used in the main manuscript.
        _set_mpl_style_like_sampling_unseen(usetex=cfg.paper_usetex)

        # Score spectrum plot
        max_score = int(scores.max())
        bins = np.arange(max_score + 2)
        def spectrum(prob: np.ndarray) -> np.ndarray:
            out = np.zeros(max_score + 1, dtype=np.float64)
            for s_val in range(max_score + 1):
                out[s_val] = prob[scores == s_val].sum()
            return out

        spec_p = spectrum(p_star)
        spec_q = spectrum(q0)
        spec_r = spectrum(q_recon0)

        xs = np.arange(max_score + 1)
        fig1 = plt.figure(figsize=(3.4, 2.2))
        ax = fig1.add_subplot(1, 1, 1)
        ax.bar(xs, spec_p, width=0.85, facecolor="white", edgecolor="black", linewidth=0.8, hatch="///", label=r"Target $p^*$")
        ax.bar(xs, spec_q, width=0.55, color="tab:red", edgecolor="black", linewidth=0.8, label=r"Model $q_{\theta}$")
        ax.set_xlabel("Score")
        ax.set_ylabel("Probability Mass")
        ax.legend(loc="upper left", handlelength=1.6)
        fig1.savefig(outdir / "score_spectrum.pdf")
        fig1.savefig(outdir / "score_spectrum.png", dpi=300)
        plt.close(fig1)

        # Recovery curves
        Qs = np.unique(np.round(np.logspace(0, 5, 120)).astype(int))  # 1..1e5
        R_target = np.array([recovery_fraction(p_star, holdout, int(Q)) for Q in Qs], dtype=np.float64)
        R_model = np.array([recovery_fraction(q0, holdout, int(Q)) for Q in Qs], dtype=np.float64)
        R_recon = np.array([recovery_fraction(q_recon0, holdout, int(Q)) for Q in Qs], dtype=np.float64)

        fig2 = plt.figure(figsize=(3.4, 2.2))
        ax = fig2.add_subplot(1, 1, 1)
        ax.plot(Qs, R_target, color="black", label=r"Target $p^*$")
        ax.plot(Qs, R_model, color="tab:red", label=r"IQP-QCBM $q_{\theta}$")
        ax.plot(Qs, R_recon, color="black", linestyle="-.", label=r"Spectral recon $\tilde q$")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"Sampling Budget $Q$ (log)")
        ax.set_ylabel(r"Recovery $R(Q)$")
        ax.legend(loc="upper left", handlelength=2.2)
        fig2.savefig(outdir / "recovery_curve.pdf")
        fig2.savefig(outdir / "recovery_curve.png", dpi=300)
        plt.close(fig2)

        # One combined, paper-ready figure (side-by-side panels) for direct inclusion.
        _paper_figure_maxcut_extension(
            cfg=cfg,
            outdir=outdir,
            scores=scores,
            p_star=p_star,
            holdout=holdout,
            q_model=q0,
            q_recon=q_recon0,
            rep_metrics=rep_metrics,
            usetex=cfg.paper_usetex,
        )

        print(f"\nSaved plots to: {outdir}")
        print(f"Representative seed for plots: {rep_seed} (plot_seed={cfg.plot_seed})")

    else:
        if cfg.make_plots and plt is None:
            print("matplotlib not available; skipping plots.")


if __name__ == "__main__":
    main()
