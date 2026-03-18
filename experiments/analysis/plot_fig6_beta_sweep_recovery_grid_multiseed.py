#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained Fig6-style beta sweep recomputation with multiseed bands."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from paper_benchmark_ledger import record_benchmark_run
from training_protocol import (
    PROTOCOL_VERSION as ACTIVE_PROTOCOL_VERSION,
    STANDARD_SEED_IDS,
    STANDARD_SEED_IDS_CSV,
    STANDARD_SEED_SCHEDULE_CSV,
    write_training_protocol,
)


HAS_PENNYLANE = False
try:
    import pennylane as qml  # type: ignore
    from pennylane import numpy as qnp  # type: ignore

    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False
    qml = None  # type: ignore[assignment]
    qnp = None  # type: ignore[assignment]

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

    class _TorchStub:
        class Tensor:
            pass

    class _NNStub:
        class Module:
            pass

    class _FStub:
        pass

    torch = _TorchStub()  # type: ignore[assignment]
    nn = _NNStub()  # type: ignore[assignment]
    F = _FStub()  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = "fig6_beta_sweep_recovery_grid_multiseed"
PNG_DPI = 300
COLOR_TARGET = "#222222"
COLOR_GRAY = "#666666"
LW_TARGET = 1.95
LW_UNIFORM = 1.35
LW_MODEL_SCALE = 1.0
LW_Q80 = 1.2
MS_Q80 = 6.0
BENCHMARK_PROTOCOL_VERSION = ACTIVE_PROTOCOL_VERSION
BENCHMARK_MATCHED_INSTANCE_SEED_IDS = STANDARD_SEED_IDS
BENCHMARK_HOLDOUT_SEED = 46
BENCHMARK_SEED_SCHEDULE_CSV = STANDARD_SEED_SCHEDULE_CSV
BENCHMARK_TRAIN_SAMPLE_OFFSET = 7
BENCHMARK_PARITY_BAND_OFFSET = 222
BENCHMARK_HOLDOUT_SELECTION_OFFSET = 111
BENCHMARK_CLASSICAL_NNN_INIT_OFFSET = 30001
BENCHMARK_CLASSICAL_DENSE_INIT_OFFSET = 30004
BENCHMARK_TRANSFORMER_INIT_OFFSET = 35501
BENCHMARK_TRANSFORMER_DATALOADER_OFFSET = BENCHMARK_TRANSFORMER_INIT_OFFSET + 11
BENCHMARK_MAXENT_INIT_OFFSET = 36001


def benchmark_seed_rows(K: int = 512) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    iqp_offset = 10000 + 7 * int(K)
    for idx, seed in enumerate(BENCHMARK_MATCHED_INSTANCE_SEED_IDS, start=1):
        rows.append(
            {
                "matched_seed_index": int(idx),
                "matched_seed_id": int(seed),
                "train_sample_seed": int(seed) + BENCHMARK_TRAIN_SAMPLE_OFFSET,
                "parity_band_seed": int(seed) + BENCHMARK_PARITY_BAND_OFFSET,
                "iqp_init_seed": int(seed) + iqp_offset,
                "classical_nnn_init_seed": int(seed) + BENCHMARK_CLASSICAL_NNN_INIT_OFFSET,
                "classical_dense_xent_init_seed": int(seed) + BENCHMARK_CLASSICAL_DENSE_INIT_OFFSET,
                "transformer_init_seed": int(seed) + BENCHMARK_TRANSFORMER_INIT_OFFSET,
                "transformer_dataloader_seed": int(seed) + BENCHMARK_TRANSFORMER_DATALOADER_OFFSET,
                "maxent_init_seed": int(seed) + BENCHMARK_MAXENT_INIT_OFFSET,
            }
        )
    return rows


def benchmark_protocol_metadata(
    *,
    betas: List[float],
    K: int,
    holdout_policy: str,
    fixed_holdout_seed: Optional[int] = None,
) -> Dict[str, object]:
    if fixed_holdout_seed is None:
        holdout_statement = (
            "The holdout mask is derived per matched seed from matched_seed + "
            f"{BENCHMARK_HOLDOUT_SELECTION_OFFSET}."
        )
        holdout_seed_value: Optional[int] = None
        holdout_selection_seed: Optional[int] = None
    else:
        holdout_statement = (
            "The holdout mask is shared across matched seeds and is derived from fixed holdout_seed + "
            f"{BENCHMARK_HOLDOUT_SELECTION_OFFSET}."
        )
        holdout_seed_value = int(fixed_holdout_seed)
        holdout_selection_seed = int(fixed_holdout_seed) + BENCHMARK_HOLDOUT_SELECTION_OFFSET
    return {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "matched_instance_definition": (
            "A matched instance is indexed by (beta, s), with beta in {0.1, 0.2, ..., 2.0} "
            f"and s in {{1, ..., {len(BENCHMARK_MATCHED_INSTANCE_SEED_IDS)}}}, "
            f"yielding {20 * len(BENCHMARK_MATCHED_INSTANCE_SEED_IDS)} matched instances in total."
        ),
        "matched_instance_seed_ids": [int(x) for x in BENCHMARK_MATCHED_INSTANCE_SEED_IDS],
        "matched_instance_count_per_beta": int(len(BENCHMARK_MATCHED_INSTANCE_SEED_IDS)),
        "matched_instance_count_total_wide_beta_sweep": int(20 * len(BENCHMARK_MATCHED_INSTANCE_SEED_IDS)),
        "seed_schedule_csv": BENCHMARK_SEED_SCHEDULE_CSV,
        "shared_randomness_statement": (
            "Within each matched instance, all models receive the same D_train; parity-based models "
            "additionally receive the same parity band Omega."
        ),
        "holdout_policy": str(holdout_policy),
        "holdout_randomness_statement": holdout_statement,
        "randomness_stack": [
            "Fix beta and build p* on the even-parity support.",
            holdout_statement,
            f"Sample D_train from p_train with matched_seed + {BENCHMARK_TRAIN_SAMPLE_OFFSET}.",
            f"Sample the parity band Omega with matched_seed + {BENCHMARK_PARITY_BAND_OFFSET}.",
            "Initialize each model with its model-specific initialization seed.",
        ],
        "fixed_holdout_seed": holdout_seed_value,
        "fixed_holdout_selection_seed": holdout_selection_seed,
        "restart_policy": {
            "num_restarts": 1,
            "varied_across_restarts": "none",
            "selection_rule": "single_run_only",
        },
        "betas_for_this_run": [float(x) for x in betas],
        "K_for_this_run": int(K),
        "seed_rows": benchmark_seed_rows(K=int(K)),
    }


def _parse_float_list(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    return np.asarray(vals, dtype=np.float64)


def _parse_int_list(s: str) -> np.ndarray:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    return np.asarray(vals, dtype=np.int64)


def _first_q_crossing(Q: np.ndarray, y: np.ndarray, thr: float) -> float:
    idx = np.where(y >= thr)[0]
    if idx.size == 0:
        return float("inf")
    i = int(idx[0])
    if i == 0:
        return float(Q[0])
    x0, x1 = float(Q[i - 1]), float(Q[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 <= y0 + 1e-12:
        return x1
    t = (float(thr) - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return x0 + t * (x1 - x0)


def _select_holdout_random(candidate_mask: np.ndarray, holdout_k: int, seed: int) -> np.ndarray:
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    holdout = np.zeros_like(candidate_mask, dtype=bool)
    cand = np.flatnonzero(candidate_mask)
    if holdout_k <= 0 or cand.size == 0:
        return holdout
    rng = np.random.default_rng(int(seed))
    choose_k = min(int(holdout_k), int(cand.size))
    chosen = rng.choice(cand, size=choose_k, replace=False)
    holdout[np.asarray(chosen, dtype=np.int64)] = True
    return holdout


def _legend_handles_compact(
    model_labels: List[str],
    style_color: List[str],
    style_ls: List[object],
    style_lw: List[float],
) -> List[Line2D]:
    handles: List[Line2D] = [
        Line2D([0], [0], color=COLOR_TARGET, lw=LW_TARGET, ls="-", label=r"Target $p^*$"),
    ]
    for lab, col, ls, lw in zip(model_labels, style_color, style_ls, style_lw):
        handles.append(
            Line2D(
                [0],
                [0],
                color=str(col),
                lw=max(1.0, float(lw) * LW_MODEL_SCALE),
                ls=ls,
                label=str(lab),
            )
        )
    handles.append(Line2D([0], [0], color=COLOR_GRAY, lw=LW_UNIFORM, ls="--", label="Uniform"))
    return handles


def int2bits(k: int, n: int) -> np.ndarray:
    return np.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=np.int8)


def parity_even(bits: np.ndarray) -> bool:
    return (int(np.sum(bits)) % 2) == 0


def make_bits_table(n: int) -> np.ndarray:
    N = 2**n
    return np.array([int2bits(i, n) for i in range(N)], dtype=np.int8)


def longest_zero_run_between_ones(bits: np.ndarray) -> int:
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2:
        return 0
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    return max(gaps) if gaps else 0


def build_target_distribution_score_tilt(
    n: int,
    beta: float,
    even_parity_only: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = 2**n
    support = np.zeros(N, dtype=bool)
    scores = np.zeros(N, dtype=np.float64)

    for k in range(N):
        bits = int2bits(k, n)
        if (not even_parity_only) or parity_even(bits):
            support[k] = True
            scores[k] = 1.0 + float(longest_zero_run_between_ones(bits))

    logits = np.full(N, -np.inf, dtype=np.float64)
    logits[support] = beta * scores[support]
    m = float(np.max(logits[support]))
    unnorm = np.zeros(N, dtype=np.float64)
    unnorm[support] = np.exp(logits[support] - m)
    z = float(np.sum(unnorm))
    if z <= 0.0:
        raise RuntimeError("Failed to normalize target distribution.")
    p_star = unnorm / z
    return p_star.astype(np.float64), support, scores.astype(np.float64)


def build_target_distribution_paper(n: int, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_target_distribution_score_tilt(n=n, beta=beta, even_parity_only=True)


def topk_mask_by_scores(scores: np.ndarray, support: np.ndarray, frac: float = 0.05) -> np.ndarray:
    valid = np.where(support)[0]
    k = max(1, int(np.floor(frac * valid.size)))
    order = np.argsort(-scores[valid])
    top_indices = valid[order[:k]]
    mask = np.zeros_like(support, dtype=bool)
    mask[top_indices] = True
    return mask


def sample_indices(probs: np.ndarray, m: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    p = np.asarray(probs, dtype=np.float64)
    p = p / float(np.sum(p))
    return rng.choice(len(p), size=int(m), replace=True, p=p)


def empirical_dist(idxs: np.ndarray, N: int) -> np.ndarray:
    counts = np.bincount(np.asarray(idxs, dtype=np.int64), minlength=int(N))
    return (counts / max(1, int(np.sum(counts)))).astype(np.float64)


def _min_hamming_to_set(bit_vec: np.ndarray, sel_bits: np.ndarray) -> int:
    if sel_bits.shape[0] == 0:
        return bit_vec.shape[0]
    d = np.sum(sel_bits != bit_vec[None, :], axis=1)
    return int(np.min(d))


def select_holdout_smart(
    p_star: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    m_train: int,
    holdout_k: int,
    pool_size: int,
    seed: int,
) -> np.ndarray:
    if holdout_k <= 0:
        return np.zeros_like(good_mask, dtype=bool)

    rng = np.random.default_rng(int(seed))
    good_idxs = np.where(good_mask)[0]
    taus = [1.0 / max(1, int(m_train)), 0.5 / max(1, int(m_train)), 0.25 / max(1, int(m_train)), 0.0]

    cand: Optional[np.ndarray] = None
    for tau in taus:
        cand = good_idxs[p_star[good_idxs] >= tau]
        if cand.size >= int(holdout_k):
            break
    if cand is None or cand.size == 0:
        raise RuntimeError("No candidates for holdout selection.")

    order = np.argsort(-p_star[cand])
    cand = cand[order]
    pool = cand[: min(int(pool_size), int(cand.size))].copy()
    rng.shuffle(pool)

    pool_sorted = pool[np.argsort(-p_star[pool])]
    selected = [int(pool_sorted[0])]
    selected_bits = bits_table[[selected[-1]]].copy()

    while len(selected) < int(holdout_k) and len(selected) < int(pool.size):
        best_idx = None
        best_d = -1
        best_p = -1.0
        for idx in pool:
            idx_int = int(idx)
            if idx_int in selected:
                continue
            d = _min_hamming_to_set(bits_table[idx_int], selected_bits)
            p = float(p_star[idx_int])
            if (d > best_d) or (d == best_d and p > best_p):
                best_d = d
                best_p = p
                best_idx = idx_int
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_bits = np.vstack([selected_bits, bits_table[best_idx]])

    holdout = np.zeros_like(good_mask, dtype=bool)
    holdout[np.asarray(selected, dtype=np.int64)] = True
    return holdout


def p_sigma(sigma: float) -> float:
    if sigma <= 0:
        return 0.5
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma**2)))


def sample_alphas(n: int, sigma: float, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    p = p_sigma(float(sigma))
    alphas = rng.binomial(1, p, size=(int(K), int(n))).astype(np.int8)

    zero = np.where(np.sum(alphas, axis=1) == 0)[0]
    while zero.size > 0:
        alphas[zero] = rng.binomial(1, p, size=(zero.size, int(n))).astype(np.int8)
        zero = np.where(np.sum(alphas, axis=1) == 0)[0]
    return alphas


def build_parity_matrix(alphas: np.ndarray, bits_table: np.ndarray) -> np.ndarray:
    A = alphas.astype(np.int16)
    X = bits_table.astype(np.int16).T
    par = (A @ X) & 1
    return np.where(par == 0, 1.0, -1.0).astype(np.float64)


def expected_unique_fraction(probs: np.ndarray, mask: np.ndarray, Q_vals: np.ndarray) -> np.ndarray:
    Q_vals = np.asarray(Q_vals, dtype=np.int64)
    H = int(np.sum(mask))
    if H == 0:
        return np.zeros_like(Q_vals, dtype=np.float64)
    pS = np.asarray(probs[mask], dtype=np.float64)[:, None]
    return np.sum(1.0 - np.power(1.0 - pS, Q_vals[None, :]), axis=0) / H


def find_Q_threshold(probs: np.ndarray, mask: np.ndarray, thr: float = 0.8, Qmax: int = 200000) -> float:
    if int(np.sum(mask)) == 0:
        return float("nan")

    def frac(Q: int) -> float:
        return float(expected_unique_fraction(probs, mask, np.array([Q], dtype=np.int64))[0])

    if frac(1) >= float(thr):
        return 1.0

    lo, hi = 1, 1
    while hi < int(Qmax) and frac(hi) < float(thr):
        hi *= 2
    if hi >= int(Qmax) and frac(int(Qmax)) < float(thr):
        return float("inf")
    hi = min(hi, int(Qmax))

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if frac(mid) >= float(thr):
            hi = mid
        else:
            lo = mid
    return float(hi)


def compute_metrics_for_q(q: np.ndarray, holdout_mask: np.ndarray, q80_thr: float, q80_search_max: int) -> Dict[str, float]:
    qH = float(np.sum(np.asarray(q, dtype=np.float64)[np.asarray(holdout_mask, dtype=bool)]))
    R10000 = float(expected_unique_fraction(q, holdout_mask, np.array([10000], dtype=np.int64))[0])
    Q80 = find_Q_threshold(q, holdout_mask, thr=float(q80_thr), Qmax=int(q80_search_max))
    return {"qH": qH, "R_Q10000": R10000, "Q80": float(Q80)}


def get_iqp_pairs_nn_nnn(n: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(int(n)):
        pairs.append(tuple(sorted((i, (i + 1) % int(n)))))
        pairs.append(tuple(sorted((i, (i + 2) % int(n)))))
    return sorted(list(set(pairs)))


def iqp_circuit_zz_only(W, wires, pairs: List[Tuple[int, int]], layers: int = 1) -> None:
    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)
    for _ in range(int(layers)):
        for i, j in pairs:
            qml.IsingZZ(W[idx], wires=[wires[i], wires[j]])
            idx += 1
        for w in wires:
            qml.Hadamard(wires=w)


def train_iqp_qcbm(
    n: int,
    layers: int,
    steps: int,
    lr: float,
    P: np.ndarray,
    z_data: np.ndarray,
    seed_init: int,
    eval_every: int = 50,
) -> np.ndarray:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP recomputation.")

    dev = qml.device("default.qubit", wires=int(n))
    pairs = get_iqp_pairs_nn_nnn(int(n))

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit_zz_only(W, range(int(n)), pairs, layers=int(layers))
        return qml.probs(wires=range(int(n)))

    P_t = qnp.array(np.asarray(P, dtype=np.float64), requires_grad=False)
    z_t = qnp.array(np.asarray(z_data, dtype=np.float64), requires_grad=False)

    num_params = len(pairs) * int(layers)
    rng = np.random.default_rng(int(seed_init))
    W = qnp.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(float(lr))

    def loss_fn(w):
        q = circuit(w)
        return qnp.mean((z_t - P_t @ q) ** 2)

    for step in range(1, int(steps) + 1):
        W, _ = opt.step_and_cost(loss_fn, W)
        if int(eval_every) > 0 and step % int(eval_every) == 0:
            pass

    q_final = np.asarray(circuit(W), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final = q_final / max(1e-15, float(np.sum(q_final)))
    return q_final.astype(np.float64)


def _all_pairs_dense(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(int(n)) for j in range(i + 1, int(n))]


class _ARTransformer(nn.Module):
    def __init__(
        self,
        n: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(3, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, int(n), d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, inp_tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(inp_tokens) + self.pos_emb[:, : inp_tokens.shape[1], :]
        t = inp_tokens.shape[1]
        causal = torch.triu(torch.ones(t, t, device=inp_tokens.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(x, mask=causal)
        logits = self.out(h).squeeze(-1)
        return logits


def _train_transformer_autoregressive(
    bits_table: np.ndarray,
    idxs_train: np.ndarray,
    n: int,
    seed: int,
    epochs: int = 250,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_ff: int = 128,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> np.ndarray:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for the Transformer baseline.")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cpu")

    x_train = torch.from_numpy(bits_table[np.asarray(idxs_train, dtype=np.int64)].astype(np.int64)).to(device)
    ds = torch.utils.data.TensorDataset(x_train)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator(device="cpu").manual_seed(int(seed) + 11),
    )

    model = _ARTransformer(
        n=int(n),
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_ff=int(dim_ff),
        dropout=0.0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    for _ in range(max(1, int(epochs))):
        model.train()
        for (xb,) in dl:
            bos = torch.full((xb.shape[0], 1), 2, device=device, dtype=torch.long)
            inp = torch.cat([bos, xb[:, :-1]], dim=1)
            logits = model(inp)
            loss = F.binary_cross_entropy_with_logits(logits, xb.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    x_all = torch.from_numpy(bits_table.astype(np.int64)).to(device)
    with torch.no_grad():
        bos = torch.full((x_all.shape[0], 1), 2, device=device, dtype=torch.long)
        inp = torch.cat([bos, x_all[:, :-1]], dim=1)
        logits = model(inp)
        probs = torch.sigmoid(logits)
        x_float = x_all.float()
        logp = x_float * torch.log(torch.clamp(probs, 1e-12, 1.0))
        logp = logp + (1.0 - x_float) * torch.log(torch.clamp(1.0 - probs, 1e-12, 1.0))
        logp = torch.sum(logp, dim=1)
        logp = logp - torch.logsumexp(logp, dim=0)
        q = torch.exp(logp).cpu().numpy().astype(np.float64)

    q = np.clip(q, 0.0, 1.0)
    q = q / max(1e-15, float(np.sum(q)))
    return q


def _train_maxent_parity(
    P: np.ndarray,
    z_data: np.ndarray,
    seed: int,
    steps: int = 2000,
    lr: float = 5e-2,
) -> np.ndarray:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for the MaxEnt parity baseline.")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cpu")

    P_t = torch.from_numpy(np.asarray(P, dtype=np.float32)).to(device)
    z_t = torch.from_numpy(np.asarray(z_data, dtype=np.float32)).to(device)
    K = int(P_t.shape[0])

    theta = nn.Parameter(torch.zeros(K, device=device))
    opt = torch.optim.Adam([theta], lr=float(lr))

    for _ in range(max(1, int(steps))):
        logits = torch.matmul(theta, P_t)
        loss = torch.logsumexp(logits, dim=0) - torch.dot(theta, z_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = torch.matmul(theta, P_t)
        logits = logits - torch.logsumexp(logits, dim=0)
        q = torch.exp(logits).cpu().numpy().astype(np.float64)

    q = np.clip(q, 0.0, 1.0)
    q = q / max(1e-15, float(np.sum(q)))
    return q


def _train_classical_boltzmann(
    n: int,
    steps: int,
    lr: float,
    seed_init: int,
    P: np.ndarray,
    z_data: np.ndarray,
    loss_mode: str,
    emp_dist: np.ndarray,
    topology: str = "nn_nnn",
    include_fields: bool = True,
) -> np.ndarray:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for classical baseline recomputation.")

    if topology == "nn_nnn":
        pairs = get_iqp_pairs_nn_nnn(int(n))
    elif topology == "dense":
        pairs = _all_pairs_dense(int(n))
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    bits = make_bits_table(int(n))
    spins = 1.0 - 2.0 * bits.astype(np.float64)
    N = spins.shape[0]

    pair_feats = np.zeros((len(pairs), N), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        pair_feats[k] = spins[:, i] * spins[:, j]

    feat_blocks = [pair_feats]
    if include_fields:
        feat_blocks.append(spins.T.copy())
    F_mat = np.concatenate(feat_blocks, axis=0)

    F_t = qnp.array(F_mat, requires_grad=False)
    P_t = qnp.array(np.asarray(P, dtype=np.float64), requires_grad=False)
    z_t = qnp.array(np.asarray(z_data, dtype=np.float64), requires_grad=False)
    emp_t = qnp.array(np.asarray(emp_dist, dtype=np.float64), requires_grad=False)
    emp_t = emp_t / qnp.sum(emp_t)

    rng = np.random.default_rng(int(seed_init))
    theta = qnp.array(0.01 * rng.standard_normal(F_mat.shape[0]), requires_grad=True)
    opt = qml.AdamOptimizer(float(lr))

    def _softmax(logits):
        m = qnp.max(logits)
        ex = qnp.exp(logits - m)
        return ex / qnp.sum(ex)

    def _q_from_theta(theta_flat):
        logits = qnp.dot(theta_flat, F_t)
        return _softmax(logits)

    loss_name = str(loss_mode).lower()
    if loss_name not in {"parity_mse", "xent"}:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    def _loss(theta_flat):
        q = _q_from_theta(theta_flat)
        if loss_name == "parity_mse":
            return qnp.mean((z_t - P_t @ q) ** 2)
        q_clip = qnp.clip(q, 1e-12, 1.0)
        return -qnp.sum(emp_t * qnp.log(q_clip))

    for _ in range(max(1, int(steps))):
        theta, _ = opt.step_and_cost(_loss, theta)

    q_final = np.asarray(_q_from_theta(theta), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final = q_final / max(1e-15, float(np.sum(q_final)))
    return q_final


def _distribution_fit_metrics(q: np.ndarray, p_star: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    qv = np.asarray(q, dtype=np.float64)
    pv = np.asarray(p_star, dtype=np.float64)
    qv = np.clip(qv, eps, 1.0)
    pv = np.clip(pv, eps, 1.0)
    qv = qv / float(np.sum(qv))
    pv = pv / float(np.sum(pv))
    tv = 0.5 * float(np.sum(np.abs(qv - pv)))
    return {"tv": tv}


def _compute_curves(
    *,
    betas: np.ndarray,
    model_order: List[str],
    Q: np.ndarray,
    seed_values: np.ndarray,
    holdout_seed: int,
    holdout_mode: str,
    n: int,
    train_m: int,
    holdout_m_train: int,
    sigma: float,
    K: int,
    layers: int,
    holdout_k: int,
    holdout_pool: int,
    good_frac: float,
    iqp_steps: int,
    iqp_lr: float,
    iqp_eval_every: int,
    q80_thr: float,
    q80_search_max: int,
    artr_epochs: int,
    artr_d_model: int,
    artr_heads: int,
    artr_layers: int,
    artr_ff: int,
    artr_lr: float,
    artr_batch_size: int,
    maxent_steps: int,
    maxent_lr: float,
    outdir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bits_table = make_bits_table(int(n))
    curves = np.full((betas.size, len(model_order), seed_values.size, Q.size), np.nan, dtype=np.float64)
    y_target = np.full((betas.size, Q.size), np.nan, dtype=np.float64)
    y_unif = np.full((betas.size, Q.size), np.nan, dtype=np.float64)
    tv_vals = np.full((betas.size, len(model_order)), np.nan, dtype=np.float64)
    metric_rows: List[Dict[str, float | int | str]] = []

    for bi, beta in enumerate(betas.tolist()):
        p_star, support, scores = build_target_distribution_paper(int(n), float(beta))
        good_mask = topk_mask_by_scores(scores, support, frac=float(good_frac))

        if holdout_mode in {"global", "random_global"}:
            holdout_candidate = support.astype(bool)
        elif holdout_mode in {"high_value", "random_high_value"}:
            holdout_candidate = good_mask
        else:
            raise ValueError(f"Unsupported holdout mode: {holdout_mode}")

        if holdout_mode.startswith("random_"):
            holdout_mask = _select_holdout_random(
                holdout_candidate,
                holdout_k=int(holdout_k),
                seed=int(holdout_seed) + 111,
            )
        else:
            holdout_mask = select_holdout_smart(
                p_star=p_star,
                good_mask=holdout_candidate,
                bits_table=bits_table,
                m_train=int(holdout_m_train),
                holdout_k=int(holdout_k),
                pool_size=int(holdout_pool),
                seed=int(holdout_seed) + 111,
            )

        y_target[bi] = expected_unique_fraction(p_star, holdout_mask, Q).astype(np.float64)
        q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
        y_unif[bi] = expected_unique_fraction(q_unif, holdout_mask, Q).astype(np.float64)

        p_train = p_star.copy()
        p_train[holdout_mask] = 0.0
        p_train /= float(np.sum(p_train))

        for si, seed in enumerate(seed_values.tolist()):
            idxs_train = sample_indices(p_train, int(train_m), seed=int(seed) + 7)
            emp = empirical_dist(idxs_train, p_star.size)

            alphas = sample_alphas(int(n), float(sigma), int(K), seed=int(seed) + 222)
            P = build_parity_matrix(alphas, bits_table)
            z_data = P @ emp

            q_by_key = {
                "iqp_parity_mse": train_iqp_qcbm(
                    n=int(n),
                    layers=int(layers),
                    steps=int(iqp_steps),
                    lr=float(iqp_lr),
                    P=P,
                    z_data=z_data,
                    seed_init=int(seed) + 10000 + 7 * int(K),
                    eval_every=int(iqp_eval_every),
                ),
                "classical_nnn_fields_parity": _train_classical_boltzmann(
                    n=int(n),
                    steps=int(iqp_steps),
                    lr=float(iqp_lr),
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
                    steps=int(iqp_steps),
                    lr=float(iqp_lr),
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
                    epochs=int(artr_epochs),
                    d_model=int(artr_d_model),
                    nhead=int(artr_heads),
                    num_layers=int(artr_layers),
                    dim_ff=int(artr_ff),
                    lr=float(artr_lr),
                    batch_size=int(artr_batch_size),
                ),
                "classical_maxent_parity": _train_maxent_parity(
                    P=P,
                    z_data=z_data,
                    seed=int(seed) + 36001,
                    steps=int(maxent_steps),
                    lr=float(maxent_lr),
                ),
            }

            for mi, model_key in enumerate(model_order):
                q_model = np.asarray(q_by_key[model_key], dtype=np.float64)
                y_curve = expected_unique_fraction(q_model, holdout_mask, Q).astype(np.float64)
                curves[bi, mi, si] = y_curve
                fit = _distribution_fit_metrics(q=q_model, p_star=p_star)
                tv = float(fit["tv"])
                if np.isnan(tv_vals[bi, mi]):
                    tv_vals[bi, mi] = 0.0
                tv_vals[bi, mi] += tv
                met = compute_metrics_for_q(q_model, holdout_mask, float(q80_thr), int(q80_search_max))
                metric_rows.append(
                    {
                        "beta": float(beta),
                        "model_key": str(model_key),
                        "seed": int(seed),
                        "holdout_seed": int(holdout_seed),
                        "holdout_mode": str(holdout_mode),
                        "holdout_m_train": int(holdout_m_train),
                        "n": int(n),
                        "train_m": int(train_m),
                        "sigma": float(sigma),
                        "K": int(K),
                        "iqp_steps": int(iqp_steps),
                        "artr_epochs": int(artr_epochs),
                        "maxent_steps": int(maxent_steps),
                        "TV": tv,
                        "qH": float(met["qH"]),
                        "Q80": float(met["Q80"]),
                        "R_Q10000": float(met["R_Q10000"]),
                    }
                )

    tv_vals /= float(seed_values.size)
    metrics_path = outdir / f"{OUTPUT_STEM}_metrics.csv"
    if metric_rows:
        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            for row in metric_rows:
                writer.writerow(row)

    mean_curves = np.nanmean(curves, axis=2)
    std_curves = np.nanstd(curves, axis=2, ddof=0)
    np.savez(
        outdir / f"{OUTPUT_STEM}_data.npz",
        betas=betas,
        Q=Q,
        model_order=np.asarray(model_order, dtype=object),
        seed_values=seed_values,
        holdout_seed=np.asarray([holdout_seed], dtype=np.int64),
        holdout_mode=np.asarray([holdout_mode], dtype=object),
        holdout_m_train=np.asarray([holdout_m_train], dtype=np.int64),
        curves=curves,
        mean_curves=mean_curves,
        std_curves=std_curves,
        y_target=y_target,
        y_unif=y_unif,
        tv_vals=tv_vals,
    )
    return mean_curves, std_curves, y_target, y_unif


def _write_run_config(
    args: argparse.Namespace,
    *,
    betas: np.ndarray,
    seed_values: np.ndarray,
    outdir: Path,
) -> None:
    outdir_abs = outdir.resolve()
    try:
        outdir_rel = str(outdir_abs.relative_to(ROOT))
    except ValueError:
        outdir_rel = str(outdir)

    legacy_repo_snapshot = (
        outdir_abs == (ROOT / "outputs" / "analysis" / "fig6_multiseed_all600_seeds42_46").resolve()
        and np.allclose(betas, np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64))
        and np.array_equal(seed_values, np.asarray([42, 43, 44, 45, 46], dtype=np.int64))
        and int(args.holdout_seed) == 46
        and str(args.holdout_mode) == "global"
        and int(args.iqp_steps) == 600
        and int(args.artr_epochs) == 600
        and int(args.maxent_steps) == 600
    )
    benchmark_seed_standard = np.asarray(BENCHMARK_MATCHED_INSTANCE_SEED_IDS, dtype=np.int64)
    run_config = {
        "selected_final_run": bool(legacy_repo_snapshot),
        "legacy_repo_snapshot": bool(legacy_repo_snapshot),
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "matches_active_standard_seed_schedule": bool(np.array_equal(seed_values, benchmark_seed_standard)),
        "script": "experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py",
        "self_contained_training_logic": True,
        "outdir": outdir_rel,
        "betas": [float(x) for x in betas.tolist()],
        "n": int(args.n),
        "seed_values": [int(x) for x in seed_values.tolist()],
        "seed_count": int(seed_values.size),
        "holdout_seed": int(args.holdout_seed),
        "holdout_mode": str(args.holdout_mode),
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
        "band_statistic": "mean_plus_minus_std",
        "benchmark_protocol": benchmark_protocol_metadata(
            betas=[float(x) for x in betas.tolist()],
            K=int(args.K),
            holdout_policy="fixed_holdout_seed_plus_111",
            fixed_holdout_seed=int(args.holdout_seed),
        ),
        "rerun_command": (
            "MPLCONFIGDIR=/tmp/mpl-cache python "
            "experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py "
            "--recompute 1 "
            f"--betas {args.betas} "
            f"--q80-search-max {int(args.q80_search_max)} "
            f"--seeds {args.seeds} "
            f"--holdout-seed {int(args.holdout_seed)} "
            f"--holdout-mode {str(args.holdout_mode)} "
            f"--iqp-steps {int(args.iqp_steps)} "
            f"--artr-epochs {int(args.artr_epochs)} "
            f"--maxent-steps {int(args.maxent_steps)} "
            f"--grid-cols {int(args.grid_cols)} "
            f"--qmax {int(args.qmax)} "
            f"--log-x {int(args.log_x)} "
            f"--outdir {Path(args.outdir).as_posix()}"
        ),
    }
    with (outdir / "RUN_CONFIG.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")
    write_training_protocol(
        outdir,
        experiment_name="Fig6 multiseed recovery grid",
        note="This run uses the shared 10-seed / 600-budget analysis standard.",
        source_relpath="experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py",
        metrics_note="Primary outputs are multiseed recovery curves; training defaults remain fixed at the shared 600-budget protocol.",
    )


def run() -> None:
    ap = argparse.ArgumentParser(description="Recompute a Fig6-style plot with multiseed uncertainty bands.")
    ap.add_argument(
        "--style-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig6_beta_sweep_recovery_grid" / "fig6_data_default.npz"),
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "analysis" / "fig6_multiseed_all600_seeds101_110"),
    )
    ap.add_argument("--betas", type=str, default="0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2")
    ap.add_argument(
        "--seeds",
        type=str,
        default=STANDARD_SEED_IDS_CSV,
        help="Comma-separated matched-instance seed IDs. The active analysis standard uses 10 seeds: 101..110.",
    )
    ap.add_argument(
        "--holdout-seed",
        type=int,
        default=BENCHMARK_HOLDOUT_SEED,
        help="Fixed holdout seed shared across matched instances; benchmark standard keeps this at 46.",
    )
    ap.add_argument(
        "--holdout-mode",
        type=str,
        default="global",
        choices=["global", "high_value", "random_global", "random_high_value"],
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=20)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=100000)
    ap.add_argument("--artr-epochs", type=int, default=600)
    ap.add_argument("--artr-d-model", type=int, default=64)
    ap.add_argument("--artr-heads", type=int, default=4)
    ap.add_argument("--artr-layers", type=int, default=2)
    ap.add_argument("--artr-ff", type=int, default=128)
    ap.add_argument("--artr-lr", type=float, default=1e-3)
    ap.add_argument("--artr-batch-size", type=int, default=256)
    ap.add_argument("--maxent-steps", type=int, default=600)
    ap.add_argument("--maxent-lr", type=float, default=5e-2)
    ap.add_argument("--grid-cols", type=int, default=4)
    ap.add_argument("--qmax", type=int, default=10000)
    ap.add_argument("--log-x", type=int, default=0, choices=[0, 1])
    ap.add_argument("--recompute", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    style_npz = Path(args.style_npz)
    if not style_npz.exists():
        raise FileNotFoundError(f"Missing style npz: {style_npz}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    with np.load(style_npz, allow_pickle=True) as z:
        Q_style = np.asarray(z["Q"], dtype=np.int64)
        model_order = [str(x) for x in z["model_order"].tolist()]
        model_labels = [str(x) for x in z["model_labels"].tolist()]
        style_color = [str(x) for x in z["style_color"].tolist()]
        style_ls = z["style_ls"].tolist()
        style_lw = [float(x) for x in z["style_lw"].tolist()]

    betas = _parse_float_list(args.betas)
    seed_values = _parse_int_list(args.seeds)
    Q = Q_style[Q_style <= int(args.qmax)]
    _write_run_config(args, betas=betas, seed_values=seed_values, outdir=outdir)

    data_npz = outdir / f"{OUTPUT_STEM}_data.npz"
    if data_npz.exists() and not bool(int(args.recompute)):
        with np.load(data_npz, allow_pickle=True) as z:
            mean_curves = np.asarray(z["mean_curves"], dtype=np.float64)
            std_curves = np.asarray(z["std_curves"], dtype=np.float64)
            y_target = np.asarray(z["y_target"], dtype=np.float64)
            y_unif = np.asarray(z["y_unif"], dtype=np.float64)
    else:
        if not HAS_PENNYLANE:
            raise RuntimeError("Pennylane is required for recompute mode.")
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for recompute mode.")
        mean_curves, std_curves, y_target, y_unif = _compute_curves(
            betas=betas,
            model_order=model_order,
            Q=Q,
            seed_values=seed_values,
            holdout_seed=int(args.holdout_seed),
            holdout_mode=str(args.holdout_mode),
            n=int(args.n),
            train_m=int(args.train_m),
            holdout_m_train=int(args.holdout_m_train),
            sigma=float(args.sigma),
            K=int(args.K),
            layers=int(args.layers),
            holdout_k=int(args.holdout_k),
            holdout_pool=int(args.holdout_pool),
            good_frac=float(args.good_frac),
            iqp_steps=int(args.iqp_steps),
            iqp_lr=float(args.iqp_lr),
            iqp_eval_every=int(args.iqp_eval_every),
            q80_thr=float(args.q80_thr),
            q80_search_max=int(args.q80_search_max),
            artr_epochs=int(args.artr_epochs),
            artr_d_model=int(args.artr_d_model),
            artr_heads=int(args.artr_heads),
            artr_layers=int(args.artr_layers),
            artr_ff=int(args.artr_ff),
            artr_lr=float(args.artr_lr),
            artr_batch_size=int(args.artr_batch_size),
            maxent_steps=int(args.maxent_steps),
            maxent_lr=float(args.maxent_lr),
            outdir=outdir,
        )

    ncols = max(1, int(args.grid_cols))
    nrows = int(np.ceil(len(betas) / ncols))
    panel_w = 3.0
    panel_h = 2.18
    fig_w = max(6.0, panel_w * ncols)
    fig_h = max(4.4, panel_h * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    x_max = int(args.qmax)
    x_min = 1 if bool(int(args.log_x)) else 0

    for i, beta in enumerate(betas):
        ax = axes[i]
        ax.plot(Q, y_target[i], color=COLOR_TARGET, linewidth=LW_TARGET, zorder=20)
        ax.plot(Q, y_unif[i], color=COLOR_GRAY, linewidth=LW_UNIFORM, linestyle="--", alpha=0.85, zorder=1)

        for j, _key in enumerate(model_order):
            mean_y = mean_curves[i, j]
            std_y = std_curves[i, j]
            color = style_color[j]
            ls = style_ls[j]
            lw = max(1.0, float(style_lw[j]) * LW_MODEL_SCALE)
            ax.fill_between(
                Q,
                np.clip(mean_y - std_y, 0.0, 1.0),
                np.clip(mean_y + std_y, 0.0, 1.0),
                color=color,
                alpha=0.10,
                zorder=5,
            )
            ax.plot(Q, mean_y, color=color, linestyle=ls, linewidth=lw, alpha=0.95, zorder=10 + j)
            ax.plot([float(Q[-1])], [float(mean_y[-1])], marker="o", markersize=4.2, color=color, alpha=0.95, zorder=12 + j)

        candidate_curves: Dict[str, np.ndarray] = {model_order[j]: mean_curves[i, j] for j in range(len(model_order))}
        candidate_curves["uniform_random"] = y_unif[i]
        candidate_colors: Dict[str, str] = {model_order[j]: style_color[j] for j in range(len(model_order))}
        candidate_colors["uniform_random"] = COLOR_GRAY

        winner_key = None
        winner_q80 = float("inf")
        for key, y_curve in candidate_curves.items():
            q80 = _first_q_crossing(Q.astype(np.float64), y_curve.astype(np.float64), float(args.q80_thr))
            if np.isfinite(q80) and q80 < winner_q80:
                winner_q80 = float(q80)
                winner_key = key

        if winner_key is not None and np.isfinite(winner_q80):
            q80_mark = float(np.clip(winner_q80, x_min, x_max))
            wcolor = candidate_colors[winner_key]
            y_q80 = float(np.interp(q80_mark, Q.astype(np.float64), candidate_curves[winner_key].astype(np.float64)))
            ax.axvspan(q80_mark, x_max, color="#FFFFFF", alpha=0.42, zorder=25)
            ax.axvline(q80_mark, color=wcolor, linestyle="--", linewidth=LW_Q80, alpha=0.95, zorder=28)
            ax.plot(
                [q80_mark],
                [y_q80],
                marker="o",
                markersize=MS_Q80,
                markerfacecolor=wcolor,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=30,
            )
            ax.text(
                float(min(x_max, q80_mark + 260.0)),
                0.07,
                "Q80",
                color=wcolor,
                fontsize=7,
                rotation=0,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
                zorder=31,
            )

        if bool(int(args.log_x)):
            ax.set_xscale("log")
            ax.set_xlim(1, x_max)
        else:
            ax.set_xlim(0, x_max)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(1.0, color=COLOR_GRAY, linestyle=":", alpha=0.6, linewidth=0.9)
        ax.set_title(fr"$\beta={beta:g}$")
        ax.set_xlabel("Q samples from model", fontsize=12)

        if i % ncols == 0:
            ax.set_ylabel(r"Recovery $R(Q)$")
        else:
            ax.tick_params(labelleft=False)

    for j in range(len(betas), len(axes)):
        axes[j].axis("off")

    if len(axes) > 0:
        legend = axes[0].legend(
            handles=_legend_handles_compact(model_labels, style_color, style_ls, style_lw),
            loc="lower right",
            bbox_to_anchor=(0.985, 0.03),
            fontsize=7.2,
            frameon=True,
            framealpha=1.0,
            facecolor="#FFFFFF",
            edgecolor="#D8D8D8",
            handlelength=2.7,
            labelspacing=0.24,
            borderpad=0.26,
            handletextpad=0.55,
            borderaxespad=0.0,
        )
        legend.set_zorder(60)

    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    if np.array_equal(seed_values, np.asarray(BENCHMARK_MATCHED_INSTANCE_SEED_IDS, dtype=np.int64)):
        metrics_csv = outdir / f"{OUTPUT_STEM}_metrics.csv"
        data_npz = outdir / f"{OUTPUT_STEM}_data.npz"
        experiment_id = (
            "fig6_base_multiseed_10seed"
            if np.allclose(betas, np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64))
            else "fig6_wide_multiseed_10seed"
        )
        title = (
            "Fig6 base recovery-grid benchmark (beta = 0.5..1.2)"
            if experiment_id == "fig6_base_multiseed_10seed"
            else "Fig6 wide recovery-grid benchmark (beta = 0.1..2.0)"
        )
        record_benchmark_run(
            experiment_id=experiment_id,
            title=title,
            run_config_path=outdir / "RUN_CONFIG.json",
            output_paths=[out_pdf, out_png],
            metrics_paths=[metrics_csv, data_npz],
            notes=[
                "10-seed active-standard multiseed recovery-grid run.",
                "Per-instance metrics and recovery curves are stored alongside the rendered panel grid.",
            ],
        )

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    run()
