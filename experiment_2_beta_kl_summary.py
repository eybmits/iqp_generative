#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 2: standalone beta-vs-KL summary with 10 matched seeds.

This script is self-contained and supports two modes:

1. recompute: run the full beta sweep locally and save per-seed / aggregated
   KL diagnostics for all five benchmark models.
2. rerender: rebuild the final PDF directly from a saved series CSV.

The active final-reporting protocol uses 10 matched seeds and fixes the
Transformer baseline to the medium capacity identified in Experiment 7:
`d_model=32`, `heads=4`, `layers=1`, `dim_ff=64`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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

        class Generator:
            def manual_seed(self, _seed: int) -> "_TorchStub.Generator":
                return self

        @staticmethod
        def device(_name: str) -> str:
            return "cpu"

    class _NNStub:
        class Module:
            pass

    class _FStub:
        pass

    torch = _TorchStub()  # type: ignore[assignment]
    nn = _NNStub()  # type: ignore[assignment]
    F = _FStub()  # type: ignore[assignment]

from training_protocol import STANDARD_SEED_IDS_CSV, write_training_protocol


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_2_beta_kl_summary.py"
OUTPUT_STEM = "experiment_2_beta_kl_summary"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM
DEFAULT_BETAS = ",".join(f"{x / 10:.1f}" for x in range(1, 21))

TRAIN_SAMPLE_OFFSET = 7
PARITY_BAND_OFFSET = 222
IQP_INIT_OFFSET = 10000
CLASSICAL_NNN_INIT_OFFSET = 30001
CLASSICAL_DENSE_INIT_OFFSET = 30004
TRANSFORMER_INIT_OFFSET = 35501
TRANSFORMER_DATALOADER_OFFSET = TRANSFORMER_INIT_OFFSET + 11
MAXENT_INIT_OFFSET = 36001

MEDIUM_TRANSFORMER = {
    "variant": "medium",
    "d_model": 32,
    "nhead": 4,
    "num_layers": 1,
    "dim_ff": 64,
}

FIG_W = 270.0 / 72.0
FIG_H = 185.52 / 72.0
POINTCLOUD_FIG_W = 320.0 / 72.0
POINTCLOUD_FIG_H = FIG_H
POINTCLOUD_LEFT = 0.17
POINTCLOUD_RIGHT = 0.985
POINTCLOUD_BOTTOM = 0.22
POINTCLOUD_TOP = 0.93
POINTCLOUD_XPAD_IN_STEPS = 0.35
PAPER_POINTCLOUD_PDF = "fig6_beta_kl_summary.pdf"

LEGEND_FONTSIZE = 7.2
BAND_ALPHA = 0.14
MAX_LABELED_X_TICKS = 10
X_LABEL_EVERY_N_BETAS = 3

MODEL_ORDER = [
    "iqp_parity_mse",
    "classical_nnn_fields_parity",
    "classical_dense_fields_xent",
    "classical_transformer_mle",
    "classical_maxent_parity",
]

MODEL_STYLE = {
    "iqp_parity_mse": {
        "label": "IQP (parity)",
        "color": "#D62728",
        "ls": "-",
        "lw": 2.35,
        "marker": "o",
    },
    "classical_nnn_fields_parity": {
        "label": "Ising+fields (NN+NNN)",
        "color": "#1f77b4",
        "ls": "-",
        "lw": 1.85,
        "marker": "o",
    },
    "classical_dense_fields_xent": {
        "label": "Dense Ising+fields (xent)",
        "color": "#8c564b",
        "ls": (0, (5, 2)),
        "lw": 1.85,
        "marker": "o",
    },
    "classical_transformer_mle": {
        "label": "AR Transformer (MLE)",
        "color": "#17becf",
        "ls": "--",
        "lw": 1.90,
        "marker": "o",
    },
    "classical_maxent_parity": {
        "label": "MaxEnt parity (P,z)",
        "color": "#9467bd",
        "ls": "-.",
        "lw": 1.90,
        "marker": "o",
    },
}

T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def apply_final_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": LEGEND_FONTSIZE,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "gray",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.03,
        }
    )


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def _try_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _parse_float_list(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    return np.asarray(vals, dtype=np.float64)


def _parse_int_list(s: str) -> np.ndarray:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    return np.asarray(vals, dtype=np.int64)


def _major_beta_ticks(betas: Sequence[float]) -> List[float]:
    beta_vals = [float(beta) for beta in betas]
    if len(beta_vals) <= MAX_LABELED_X_TICKS:
        return beta_vals
    step = max(2, int(X_LABEL_EVERY_N_BETAS))
    major = [beta_vals[idx] for idx in range(0, len(beta_vals) - 1, step)]
    if beta_vals[-1] not in major:
        base_spacing = min(abs(beta_vals[idx + 1] - beta_vals[idx]) for idx in range(len(beta_vals) - 1))
        if major and abs(beta_vals[-1] - major[-1]) <= 1.5 * base_spacing:
            major[-1] = beta_vals[-1]
        else:
            major.append(beta_vals[-1])
    return major


def _t_crit_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    df = int(n) - 1
    if df in T_CRIT_95:
        return float(T_CRIT_95[df])
    return 1.96


def _ci95_halfwidth(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    return float(_t_crit_95(int(arr.size)) * sd / math.sqrt(float(arr.size)))


def _reduce_seed_stats(values: np.ndarray) -> Dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n_seeds": 0,
            "min": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "ci95": float("nan"),
            "q3": float("nan"),
            "max": float("nan"),
        }
    q1, median, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "n_seeds": int(arr.size),
        "min": float(np.min(arr)),
        "q1": float(q1),
        "median": float(median),
        "mean": float(np.mean(arr)),
        "std": float(std),
        "ci95": float(_ci95_halfwidth(arr)),
        "q3": float(q3),
        "max": float(np.max(arr)),
    }


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def build_target_distribution_paper(n: int, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_target_distribution_score_tilt(n=n, beta=beta, even_parity_only=True)


def sample_indices(probs: np.ndarray, m: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    p = np.asarray(probs, dtype=np.float64)
    p = p / float(np.sum(p))
    return rng.choice(len(p), size=int(m), replace=True, p=p)


def empirical_dist(idxs: np.ndarray, N: int) -> np.ndarray:
    counts = np.bincount(np.asarray(idxs, dtype=np.int64), minlength=int(N))
    return (counts / max(1, int(np.sum(counts)))).astype(np.float64)


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


def get_iqp_pairs_nn_nnn(n: int) -> List[tuple[int, int]]:
    pairs: List[tuple[int, int]] = []
    for i in range(int(n)):
        pairs.append(tuple(sorted((i, (i + 1) % int(n)))))
        pairs.append(tuple(sorted((i, (i + 2) % int(n)))))
    return sorted(list(set(pairs)))


def iqp_circuit_zz_only(W, wires, pairs: List[tuple[int, int]], layers: int = 1) -> None:
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


def _all_pairs_dense(n: int) -> List[tuple[int, int]]:
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
    epochs: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_ff: int,
    lr: float,
    batch_size: int,
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
        generator=torch.Generator(device="cpu").manual_seed(int(seed) + TRANSFORMER_DATALOADER_OFFSET - TRANSFORMER_INIT_OFFSET),
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
    steps: int,
    lr: float,
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
    topology: str,
    include_fields: bool = True,
) -> np.ndarray:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for the classical Ising baselines.")

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


def _kl_pstar_to_q(p_star: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    pv = np.asarray(p_star, dtype=np.float64)
    qv = np.asarray(q, dtype=np.float64)
    pv = np.clip(pv, eps, 1.0)
    qv = np.clip(qv, eps, 1.0)
    pv = pv / float(np.sum(pv))
    qv = qv / float(np.sum(qv))
    return float(np.sum(pv * np.log(pv / qv)))


def _legend_handles(*, with_markers: bool = False) -> List[Line2D]:
    handles: List[Line2D] = []
    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        handles.append(
            Line2D(
                [0],
                [0],
                color=str(style["color"]),
                lw=float(style["lw"]),
                ls=style["ls"],
                label=str(style["label"]),
                marker=str(style.get("marker", "o")) if with_markers else "None",
                markersize=6.8 if with_markers else 0.0,
                markerfacecolor=str(style["color"]),
                markeredgecolor="white",
                markeredgewidth=0.8 if with_markers else 0.0,
            )
        )
    return handles


def _load_series_csv(series_csv: Path) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    with series_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "beta",
            "model_key",
            "model_label",
            "n_seeds",
            "min",
            "q1",
            "median",
            "mean",
            "std",
            "ci95",
            "q3",
            "max",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Series CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if model_key not in MODEL_ORDER:
                continue
            rows_out.append(
                {
                    "beta": float(row["beta"]),
                    "model_key": model_key,
                    "model_label": str(row["model_label"]),
                    "n_seeds": int(float(row["n_seeds"])),
                    "min": float(row["min"]),
                    "q1": float(row["q1"]),
                    "median": float(row["median"]),
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "ci95": float(row["ci95"]),
                    "q3": float(row["q3"]),
                    "max": float(row["max"]),
                }
            )
    if not rows_out:
        raise ValueError(f"No usable rows found in {series_csv}")
    return rows_out


def _load_metrics_csv(metrics_csv: Path) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"beta", "seed", "model_key", "model_label", "KL_pstar_to_q"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Metrics CSV is missing required columns: {missing}")
        for row in reader:
            model_key = str(row["model_key"])
            if model_key not in MODEL_ORDER:
                continue
            rows_out.append(
                {
                    "beta": float(row["beta"]),
                    "seed": int(float(row["seed"])),
                    "model_key": model_key,
                    "model_label": str(row["model_label"]),
                    "KL_pstar_to_q": float(row["KL_pstar_to_q"]),
                }
            )
    if not rows_out:
        raise ValueError(f"No usable rows found in {metrics_csv}")
    return rows_out


def _group_series(series_rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, np.ndarray]]:
    betas = sorted({float(row["beta"]) for row in series_rows})
    grouped: Dict[str, Dict[float, Dict[str, float]]] = defaultdict(dict)
    for row in series_rows:
        grouped[str(row["model_key"])][float(row["beta"])] = {
            "median": float(row["median"]),
            "q1": float(row["q1"]),
            "q3": float(row["q3"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "ci95": float(row["ci95"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
        }

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for model_key in MODEL_ORDER:
        median = np.full(len(betas), np.nan, dtype=np.float64)
        q1 = np.full(len(betas), np.nan, dtype=np.float64)
        q3 = np.full(len(betas), np.nan, dtype=np.float64)
        mean = np.full(len(betas), np.nan, dtype=np.float64)
        ci95 = np.full(len(betas), np.nan, dtype=np.float64)
        vmin = np.full(len(betas), np.nan, dtype=np.float64)
        vmax = np.full(len(betas), np.nan, dtype=np.float64)
        for idx, beta in enumerate(betas):
            if beta not in grouped.get(model_key, {}):
                continue
            rec = grouped[model_key][beta]
            median[idx] = float(rec["median"])
            q1[idx] = float(rec["q1"])
            q3[idx] = float(rec["q3"])
            mean[idx] = float(rec["mean"])
            ci95[idx] = float(rec["ci95"])
            vmin[idx] = float(rec["min"])
            vmax[idx] = float(rec["max"])
        out[model_key] = {
            "betas": np.asarray(betas, dtype=np.float64),
            "median": median,
            "q1": q1,
            "q3": q3,
            "mean": mean,
            "ci95": ci95,
            "min": vmin,
            "max": vmax,
        }
    return out


def _render_plot(series_rows: Sequence[Dict[str, object]], out_pdf: Path) -> None:
    apply_final_style()
    grouped = _group_series(series_rows)
    sample_betas = grouped[MODEL_ORDER[0]]["betas"]
    major_xticks = _major_beta_ticks(sample_betas.tolist())

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    ymin = float("inf")
    ymax = float("-inf")

    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        grp = grouped[model_key]
        x = grp["betas"]
        y = grp["median"]
        q1 = grp["q1"]
        q3 = grp["q3"]
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(q1) & np.isfinite(q3)
        if not np.any(mask):
            continue
        ax.fill_between(x[mask], q1[mask], q3[mask], color=str(style["color"]), alpha=BAND_ALPHA, lw=0.0)
        ax.plot(x[mask], y[mask], color=str(style["color"]), ls=style["ls"], lw=float(style["lw"]))
        ymin = min(ymin, float(np.nanmin(q1[mask])))
        ymax = max(ymax, float(np.nanmax(q3[mask])))

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p^* \parallel q)$")
    if sample_betas.size == 1:
        x0 = float(sample_betas[0])
        ax.set_xlim(x0 - 0.05, x0 + 0.05)
    else:
        ax.set_xlim(float(sample_betas.min()), float(sample_betas.max()))
    ax.set_xticks(major_xticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in major_xticks])
    if np.isfinite(ymin) and np.isfinite(ymax):
        pad = 0.10 * max(ymax - ymin, 1e-6)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.grid(True, ls="--", lw=0.5, alpha=0.25)

    legend = ax.legend(
        handles=_legend_handles(),
        loc="upper right",
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.65,
        handletextpad=0.5,
    )
    legend.set_zorder(100)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def _render_pointcloud(
    metrics_rows: Sequence[Dict[str, object]],
    out_pdf: Path,
    *,
    ymin_override: float | None = None,
    ymax_override: float | None = None,
) -> None:
    apply_final_style()
    grouped: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_betas = sorted({float(row["beta"]) for row in metrics_rows})
    for row in metrics_rows:
        grouped[str(row["model_key"])][float(row["beta"])].append(float(row["KL_pstar_to_q"]))

    major_xticks = _major_beta_ticks(all_betas)
    fig, ax = plt.subplots(figsize=(POINTCLOUD_FIG_W, POINTCLOUD_FIG_H))
    fig.subplots_adjust(
        left=POINTCLOUD_LEFT,
        right=POINTCLOUD_RIGHT,
        bottom=POINTCLOUD_BOTTOM,
        top=POINTCLOUD_TOP,
    )
    rng = np.random.default_rng(0)
    ymin = float("inf")
    ymax = float("-inf")

    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        color = str(style["color"])
        mean_x: List[float] = []
        mean_y: List[float] = []
        std_y: List[float] = []
        for beta in all_betas:
            vals = np.asarray(grouped.get(model_key, {}).get(float(beta), []), dtype=np.float64)
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.028, 0.028, size=vals.size)
            ax.scatter(
                np.full(vals.size, float(beta)) + jitter,
                vals,
                s=16,
                color=color,
                alpha=0.18,
                edgecolors="none",
                zorder=2,
            )
            mean_x.append(float(beta))
            mean_y.append(float(np.mean(vals)))
            std_y.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            ymin = min(ymin, float(np.min(vals)))
            ymax = max(ymax, float(np.max(vals)))

        if mean_x:
            x_arr = np.asarray(mean_x, dtype=np.float64)
            y_arr = np.asarray(mean_y, dtype=np.float64)
            sd_arr = np.asarray(std_y, dtype=np.float64)
            ax.errorbar(
                x_arr,
                y_arr,
                yerr=sd_arr,
                fmt="none",
                ecolor=color,
                elinewidth=1.15,
                capsize=2.6,
                capthick=1.15,
                alpha=0.55,
                zorder=3,
            )
            ax.plot(
                x_arr,
                y_arr,
                color=color,
                ls=style["ls"],
                lw=float(style["lw"]) * 0.85,
                alpha=0.9,
                zorder=4,
            )
            ax.scatter(
                x_arr,
                y_arr,
                s=28,
                color=color,
                alpha=0.95,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )

    ax.set_xlabel(r"$\beta$", labelpad=2.0)
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p^* \parallel q)$", labelpad=4.0)
    if len(all_betas) > 1:
        beta_step = min(abs(all_betas[idx + 1] - all_betas[idx]) for idx in range(len(all_betas) - 1))
    else:
        beta_step = 0.1
    x_pad = POINTCLOUD_XPAD_IN_STEPS * beta_step
    ax.set_xlim(float(min(all_betas)) - x_pad, float(max(all_betas)) + x_pad)
    ax.set_xticks(major_xticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in major_xticks])
    if ymin_override is not None or ymax_override is not None:
        y0 = float(ymin if ymin_override is None else ymin_override)
        y1 = float(ymax if ymax_override is None else ymax_override)
        ax.set_ylim(y0, y1)
    elif np.isfinite(ymin) and np.isfinite(ymax):
        pad = 0.08 * max(ymax - ymin, 1e-6)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.grid(True, ls="--", lw=0.5, alpha=0.25)

    legend = ax.legend(
        handles=_legend_handles(),
        loc="upper right",
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.65,
        handletextpad=0.5,
    )
    legend.set_zorder(100)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def _write_readme(
    path: Path,
    *,
    betas: np.ndarray,
    seed_values: np.ndarray,
    outdir: Path,
    metrics_csv: Path,
    series_csv: Path,
    data_npz: Path,
    out_pdf: Path,
    out_pointcloud_pdf: Path,
    paper_pointcloud_pdf: Path,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Experiment 2: Beta-KL Summary",
        "",
        "This directory contains the final beta-sweep KL summary used for the main reporting protocol.",
        "",
        "Protocol:",
        "",
        f"- betas: `{','.join(f'{float(x):.1f}' for x in betas.tolist())}`",
        f"- matched seeds: `{','.join(str(int(x)) for x in seed_values.tolist())}`",
        "- seed count: `10` by default under the active standard protocol",
        "- target distribution: even-parity score-tilted family",
        "- train sample count: `m=200`",
        f"- parity band: `sigma={float(args.sigma):g}`, `K={int(args.K)}`",
        f"- IQP parity budget: `steps={int(args.iqp_steps)}`, `lr={float(args.iqp_lr):g}`",
        f"- Ising+fields budgets: `steps={int(args.classical_steps)}`, `lr={float(args.classical_lr):g}`",
        f"- MaxEnt parity budget: `steps={int(args.maxent_steps)}`, `lr={float(args.maxent_lr):g}`",
        (
            "- Transformer baseline: "
            f"`variant={MEDIUM_TRANSFORMER['variant']}`, `d_model={MEDIUM_TRANSFORMER['d_model']}`, "
            f"`layers={MEDIUM_TRANSFORMER['num_layers']}`, `heads={MEDIUM_TRANSFORMER['nhead']}`, "
            f"`dim_ff={MEDIUM_TRANSFORMER['dim_ff']}`, `epochs={int(args.transformer_epochs)}`, "
            f"`lr={float(args.transformer_lr):g}`, `batch_size={int(args.transformer_batch_size)}`"
        ),
        "",
        "Plot semantics:",
        "",
        "- line: seedwise median KL over the matched-seed pool",
        "- band: interquartile range (Q1 to Q3)",
        "- saved artifacts additionally include mean, standard deviation, and 95% CI for each beta/model pair",
        "",
        "Saved artifacts:",
        "",
        f"- per-seed KL metrics: `{_try_rel(metrics_csv)}`",
        f"- aggregated beta series: `{_try_rel(series_csv)}`",
        f"- saved data cube: `{_try_rel(data_npz)}`",
        f"- final PDF: `{_try_rel(out_pdf)}`",
        f"- pointcloud PDF: `{_try_rel(out_pointcloud_pdf)}`",
        f"- paper export alias: `{_try_rel(paper_pointcloud_pdf)}`",
        "- local protocol doc: `TRAINING_PROTOCOL.md`",
        "",
        f"- source driver: `{SCRIPT_REL}`",
        f"- outdir: `{_try_rel(outdir)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recompute_series(
    *,
    betas: np.ndarray,
    seed_values: np.ndarray,
    outdir: Path,
    metrics_csv: Path,
    series_csv: Path,
    data_npz: Path,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for Experiment 2 recomputation.")
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for Experiment 2 recomputation.")

    bits_table = make_bits_table(int(args.n))
    kl_cube = np.full((betas.size, len(MODEL_ORDER), seed_values.size), np.nan, dtype=np.float64)
    metric_rows: List[Dict[str, object]] = []

    print(f"[experiment2] recomputing {betas.size} betas x {seed_values.size} seeds", flush=True)
    for bi, beta in enumerate(betas.tolist()):
        print(f"[experiment2] beta={float(beta):.1f} ({bi + 1}/{betas.size})", flush=True)
        p_star, _support, _scores = build_target_distribution_paper(int(args.n), float(beta))

        for si, seed in enumerate(seed_values.tolist()):
            print(f"[experiment2]   seed={int(seed)} ({si + 1}/{seed_values.size})", flush=True)
            idxs_train = sample_indices(p_star, int(args.train_m), seed=int(seed) + TRAIN_SAMPLE_OFFSET)
            emp = empirical_dist(idxs_train, p_star.size)

            alphas = sample_alphas(int(args.n), float(args.sigma), int(args.K), seed=int(seed) + PARITY_BAND_OFFSET)
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
                    seed_init=int(seed) + IQP_INIT_OFFSET + 7 * int(args.K),
                    eval_every=int(args.iqp_eval_every),
                ),
                "classical_nnn_fields_parity": _train_classical_boltzmann(
                    n=int(args.n),
                    steps=int(args.classical_steps),
                    lr=float(args.classical_lr),
                    seed_init=int(seed) + CLASSICAL_NNN_INIT_OFFSET,
                    P=P,
                    z_data=z_data,
                    loss_mode="parity_mse",
                    emp_dist=emp,
                    topology="nn_nnn",
                    include_fields=True,
                ),
                "classical_dense_fields_xent": _train_classical_boltzmann(
                    n=int(args.n),
                    steps=int(args.classical_steps),
                    lr=float(args.classical_lr),
                    seed_init=int(seed) + CLASSICAL_DENSE_INIT_OFFSET,
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
                    seed=int(seed) + TRANSFORMER_INIT_OFFSET,
                    epochs=int(args.transformer_epochs),
                    d_model=int(MEDIUM_TRANSFORMER["d_model"]),
                    nhead=int(MEDIUM_TRANSFORMER["nhead"]),
                    num_layers=int(MEDIUM_TRANSFORMER["num_layers"]),
                    dim_ff=int(MEDIUM_TRANSFORMER["dim_ff"]),
                    lr=float(args.transformer_lr),
                    batch_size=int(args.transformer_batch_size),
                ),
                "classical_maxent_parity": _train_maxent_parity(
                    P=P,
                    z_data=z_data,
                    seed=int(seed) + MAXENT_INIT_OFFSET,
                    steps=int(args.maxent_steps),
                    lr=float(args.maxent_lr),
                ),
            }

            for mi, model_key in enumerate(MODEL_ORDER):
                q_model = np.asarray(q_by_key[model_key], dtype=np.float64)
                kl = float(_kl_pstar_to_q(p_star, q_model))
                kl_cube[bi, mi, si] = kl
                metric_rows.append(
                    {
                        "beta": float(beta),
                        "seed": int(seed),
                        "model_key": str(model_key),
                        "model_label": str(MODEL_STYLE[model_key]["label"]),
                        "n": int(args.n),
                        "train_m": int(args.train_m),
                        "sigma": float(args.sigma),
                        "K": int(args.K),
                        "iqp_steps": int(args.iqp_steps),
                        "classical_steps": int(args.classical_steps),
                        "transformer_variant": str(MEDIUM_TRANSFORMER["variant"]),
                        "transformer_d_model": int(MEDIUM_TRANSFORMER["d_model"]),
                        "transformer_heads": int(MEDIUM_TRANSFORMER["nhead"]),
                        "transformer_layers": int(MEDIUM_TRANSFORMER["num_layers"]),
                        "transformer_dim_ff": int(MEDIUM_TRANSFORMER["dim_ff"]),
                        "transformer_epochs": int(args.transformer_epochs),
                        "maxent_steps": int(args.maxent_steps),
                        "KL_pstar_to_q": float(kl),
                    }
                )

    summary_rows: List[Dict[str, object]] = []
    for bi, beta in enumerate(betas.tolist()):
        for mi, model_key in enumerate(MODEL_ORDER):
            stats = _reduce_seed_stats(kl_cube[bi, mi])
            summary_rows.append(
                {
                    "beta": float(beta),
                    "model_key": str(model_key),
                    "model_label": str(MODEL_STYLE[model_key]["label"]),
                    **stats,
                }
            )

    _write_csv(metrics_csv, metric_rows)
    _write_csv(series_csv, summary_rows)
    np.savez(
        data_npz,
        betas=np.asarray(betas, dtype=np.float64),
        seed_values=np.asarray(seed_values, dtype=np.int64),
        model_order=np.asarray(MODEL_ORDER, dtype=object),
        kl_cube=kl_cube,
    )
    return summary_rows


def run() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2: beta-vs-KL summary with 10 matched seeds and medium Transformer.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--recompute", type=int, default=0, help="Set to 1 to recompute the full beta sweep before rendering.")
    ap.add_argument("--series-csv", type=str, default="")
    ap.add_argument("--metrics-csv", type=str, default="")
    ap.add_argument("--data-npz", type=str, default="")
    ap.add_argument("--betas", type=str, default=DEFAULT_BETAS)
    ap.add_argument("--seeds", type=str, default=STANDARD_SEED_IDS_CSV)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=20)
    ap.add_argument("--classical-steps", type=int, default=600)
    ap.add_argument("--classical-lr", type=float, default=0.05)
    ap.add_argument("--transformer-epochs", type=int, default=600)
    ap.add_argument("--transformer-lr", type=float, default=1e-3)
    ap.add_argument("--transformer-batch-size", type=int, default=256)
    ap.add_argument("--maxent-steps", type=int, default=600)
    ap.add_argument("--maxent-lr", type=float, default=0.05)
    ap.add_argument("--pointcloud-ymin", type=float, default=0.0)
    ap.add_argument("--pointcloud-ymax", type=float, default=4.0)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    series_csv = Path(args.series_csv).expanduser() if str(args.series_csv).strip() else (outdir / f"{OUTPUT_STEM}_series.csv")
    metrics_csv = Path(args.metrics_csv).expanduser() if str(args.metrics_csv).strip() else (outdir / f"{OUTPUT_STEM}_metrics_per_seed.csv")
    data_npz = Path(args.data_npz).expanduser() if str(args.data_npz).strip() else (outdir / f"{OUTPUT_STEM}_data.npz")
    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_pointcloud_pdf = outdir / f"{OUTPUT_STEM}_pointcloud.pdf"
    paper_pointcloud_pdf = outdir / PAPER_POINTCLOUD_PDF
    summary_json = outdir / f"{OUTPUT_STEM}_summary.json"
    readme_md = outdir / "README.md"

    betas = _parse_float_list(str(args.betas))
    seed_values = _parse_int_list(str(args.seeds))

    if int(args.recompute) == 1 or not series_csv.exists():
        series_rows = _recompute_series(
            betas=betas,
            seed_values=seed_values,
            outdir=outdir,
            metrics_csv=metrics_csv,
            series_csv=series_csv,
            data_npz=data_npz,
            args=args,
        )
    else:
        series_rows = _load_series_csv(series_csv)

    _render_plot(series_rows, out_pdf)
    metrics_rows = _load_metrics_csv(metrics_csv)
    _render_pointcloud(
        metrics_rows,
        out_pointcloud_pdf,
        ymin_override=float(args.pointcloud_ymin),
        ymax_override=float(args.pointcloud_ymax),
    )
    shutil.copyfile(out_pointcloud_pdf, paper_pointcloud_pdf)

    grouped = _group_series(series_rows)
    summary_payload = {
        "script": SCRIPT_REL,
        "outdir": _try_rel(outdir),
        "betas": [float(x) for x in betas.tolist()],
        "seed_values": [int(x) for x in seed_values.tolist()],
        "seed_count": int(seed_values.size),
        "plot_center": "median",
        "plot_band": "iqr",
        "secondary_statistics": "mean_std_ci95",
        "transformer_variant": dict(MEDIUM_TRANSFORMER),
        "models": {
            model_key: {
                "best_beta_by_median": float(
                    grouped[model_key]["betas"][int(np.nanargmin(grouped[model_key]["median"]))]
                ),
                "best_median_kl": float(np.nanmin(grouped[model_key]["median"])),
                "mean_at_best_beta": float(
                    grouped[model_key]["mean"][int(np.nanargmin(grouped[model_key]["median"]))]
                ),
                "ci95_at_best_beta": float(
                    grouped[model_key]["ci95"][int(np.nanargmin(grouped[model_key]["median"]))]
                ),
            }
            for model_key in MODEL_ORDER
        },
    }
    _write_json(summary_json, summary_payload)
    _write_json(
        outdir / "RUN_CONFIG.json",
        {
            "script": SCRIPT_REL,
            "outdir": _try_rel(outdir),
            "input_mode": "recompute" if int(args.recompute) == 1 or not series_csv.exists() else "series_csv",
            "betas": [float(x) for x in betas.tolist()],
            "seed_values": [int(x) for x in seed_values.tolist()],
            "seed_count": int(seed_values.size),
            "n": int(args.n),
            "train_m": int(args.train_m),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "layers": int(args.layers),
            "iqp_steps": int(args.iqp_steps),
            "iqp_lr": float(args.iqp_lr),
            "classical_steps": int(args.classical_steps),
            "classical_lr": float(args.classical_lr),
            "transformer_variant": dict(MEDIUM_TRANSFORMER),
            "transformer_epochs": int(args.transformer_epochs),
            "transformer_lr": float(args.transformer_lr),
            "transformer_batch_size": int(args.transformer_batch_size),
            "maxent_steps": int(args.maxent_steps),
            "maxent_lr": float(args.maxent_lr),
            "series_csv": _try_rel(series_csv),
            "metrics_csv": _try_rel(metrics_csv),
            "data_npz": _try_rel(data_npz),
            "summary_json": _try_rel(summary_json),
            "pdf": _try_rel(out_pdf),
            "pointcloud_pdf": _try_rel(out_pointcloud_pdf),
            "paper_pointcloud_pdf": _try_rel(paper_pointcloud_pdf),
            "pointcloud_ymin": float(args.pointcloud_ymin),
            "pointcloud_ymax": float(args.pointcloud_ymax),
            "plot_center": "median",
            "plot_band": "iqr",
            "secondary_statistics": "mean_std_ci95",
            "pdf_only": True,
        },
    )
    _write_json(
        outdir / "RERENDER_CONFIG.json",
        {
            "script": SCRIPT_REL,
            "series_csv": _try_rel(series_csv),
            "outdir": _try_rel(outdir),
            "pdf": out_pdf.name,
            "pointcloud_pdf": out_pointcloud_pdf.name,
            "paper_pointcloud_pdf": paper_pointcloud_pdf.name,
            "pointcloud_ymin": float(args.pointcloud_ymin),
            "pointcloud_ymax": float(args.pointcloud_ymax),
            "plot_center": "median",
            "plot_band": "iqr",
            "secondary_statistics": "mean_std_ci95",
            "rerender_command": f"python {SCRIPT_REL} --outdir {outdir.as_posix()} --series-csv {series_csv.as_posix()}",
        },
    )
    _write_readme(
        readme_md,
        betas=betas,
        seed_values=seed_values,
        outdir=outdir,
        metrics_csv=metrics_csv,
        series_csv=series_csv,
        data_npz=data_npz,
        out_pdf=out_pdf,
        out_pointcloud_pdf=out_pointcloud_pdf,
        paper_pointcloud_pdf=paper_pointcloud_pdf,
        args=args,
    )
    write_training_protocol(
        outdir,
        experiment_name="Experiment 2 beta KL summary",
        note="This run uses the shared 10-seed / 600-budget analysis standard with the medium Transformer selected in Experiment 7.",
        source_relpath=SCRIPT_REL,
        metrics_note="The plotted curve shows seedwise median KL with an interquartile band; mean and 95% CI are stored alongside the rerender artifacts.",
    )
    print(f"[experiment2] wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    run()
