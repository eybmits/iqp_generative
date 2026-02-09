#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 10: Recovery-best plots with stronger classical baselines.

This script extends the single "classical control" recovery plot by adding
the final selected baselines in the same figure:
  - IQP-QCBM with prob-MSE (same architecture, different objective)
  - Matched classical Ising control (NN+NNN, parity-MSE) [existing baseline]
  - Ising + local fields (NN+NNN, parity-MSE)
  - Dense Ising + local fields (all-to-all, xent)
  - Autoregressive Transformer (MLE)
  - MaxEnt parity model (same P features, same z moments)

Outputs:
  - 4_recovery_best_high_value_all_baselines.pdf
  - 4_recovery_best_high_value_all_baselines_metrics.csv
  - 4_recovery_best_high_value_all_baselines_meta.json
  - 4_recovery_best_global_all_baselines.pdf
  - 4_recovery_best_global_all_baselines_metrics.csv
  - 4_recovery_best_global_all_baselines_meta.json
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


LEGEND_STYLE = dict(
    loc="lower right",
    fontsize=7.0,
    frameon=True,
    framealpha=0.90,
    facecolor="white",
    edgecolor="none",
    handlelength=1.6,
    labelspacing=0.25,
    borderpad=0.25,
    handletextpad=0.5,
    borderaxespad=0.2,
)


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _pick_best_row(rows: List[Dict[str, str]], mode: str) -> Dict[str, str]:
    sub = [r for r in rows if str(r["holdout_mode"]) == mode]
    best = None
    for r in sub:
        q80 = float(r["Q80_iqp"])
        if not math.isfinite(q80):
            continue
        key = (q80, -float(r["qH_ratio_iqp"]))
        if best is None or key < best[0]:
            best = (key, r)
    if best is None:
        raise RuntimeError(f"No finite IQP row found for holdout_mode={mode}.")
    return best[1]


def _load_json_if_exists(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _all_pairs_dense(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


class _ARTransformer(nn.Module):
    def __init__(
        self,
        n: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n = n
        self.tok_emb = nn.Embedding(3, d_model)  # 0,1,bos(2)
        self.pos_emb = nn.Parameter(torch.zeros(1, n, d_model))
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
        # inp_tokens: B x n in {0,1,2}
        x = self.tok_emb(inp_tokens) + self.pos_emb[:, : inp_tokens.shape[1], :]
        t = inp_tokens.shape[1]
        causal = torch.triu(torch.ones(t, t, device=inp_tokens.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(x, mask=causal)
        logits = self.out(h).squeeze(-1)  # B x n
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
        raise RuntimeError("PyTorch is required for Transformer autoregressive baseline.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    x_train = torch.from_numpy(bits_table[idxs_train].astype(np.int64)).to(device)  # B x n
    ds = torch.utils.data.TensorDataset(x_train)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator(device="cpu").manual_seed(seed + 11),
    )

    model = _ARTransformer(
        n=n,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
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
    """
    MaxEnt parity model with same feature map P and same target moments z:
      q_theta(x) ∝ exp(theta^T phi(x)), phi_k(x)=P[k,x].

    Trains the convex dual:
      L(theta) = logsumexp(theta^T phi(x)) - theta^T z_data
    whose gradient is E_q[phi] - z_data.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for MaxEnt parity baseline.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    P_t = torch.from_numpy(P.astype(np.float32)).to(device)      # K x N
    z_t = torch.from_numpy(z_data.astype(np.float32)).to(device) # K
    K = P_t.shape[0]

    theta = nn.Parameter(torch.zeros(K, device=device))
    opt = torch.optim.Adam([theta], lr=float(lr))

    for _ in range(max(1, int(steps))):
        logits = torch.matmul(theta, P_t)  # N
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


def _build_holdout_mask(
    mode: str,
    p_star: np.ndarray,
    support: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    m_train_for_holdout: int,
    holdout_k: int,
    holdout_pool: int,
    seed: int,
) -> np.ndarray:
    if mode == "high_value":
        candidate_mask = good_mask
    elif mode == "global":
        candidate_mask = support.astype(bool)
    else:
        raise ValueError(f"Unsupported holdout mode: {mode}")

    return hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate_mask,
        bits_table=bits_table,
        m_train=m_train_for_holdout,
        holdout_k=holdout_k,
        pool_size=holdout_pool,
        seed=seed + 111,
    )


def _train_classical_boltzmann(
    n: int,
    layers: int,
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
    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for stronger classical baselines.")

    qml = hv.qml
    anp = hv.np

    if topology == "nn_nnn":
        pairs = hv.get_iqp_pairs_nn_nnn(n)
    elif topology == "dense":
        pairs = _all_pairs_dense(n)
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    bits = hv.make_bits_table(n)
    spins = 1.0 - 2.0 * bits.astype(np.float64)  # N x n
    N = spins.shape[0]

    pair_feats = np.zeros((len(pairs), N), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        pair_feats[k] = spins[:, i] * spins[:, j]

    feat_blocks = [pair_feats]
    if include_fields:
        feat_blocks.append(spins.T.copy())
    F = np.concatenate(feat_blocks, axis=0)  # num_features x N
    num_features = F.shape[0]

    F_t = anp.array(F, requires_grad=False)
    P_t = anp.array(P, requires_grad=False)
    z_t = anp.array(z_data, requires_grad=False)
    emp_t = anp.array(emp_dist, requires_grad=False)
    emp_t = emp_t / anp.sum(emp_t)

    rng = np.random.default_rng(seed_init)
    # No fake multi-layer multiplication: an Ising model has no meaningful
    # depth (summing coupling constants across layers is equivalent to a
    # single layer).  Use num_features parameters directly.
    theta = anp.array(0.01 * rng.standard_normal(num_features), requires_grad=True)
    opt = qml.AdamOptimizer(lr)

    def _softmax(logits):
        m = anp.max(logits)
        ex = anp.exp(logits - m)
        return ex / anp.sum(ex)

    def _q_from_theta(theta_flat):
        logits = anp.dot(theta_flat, F_t)
        return _softmax(logits)

    loss_name = str(loss_mode).lower()
    if loss_name not in ("parity_mse", "prob_mse", "xent"):
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    def _loss(theta_flat):
        q = _q_from_theta(theta_flat)
        if loss_name == "parity_mse":
            return anp.mean((z_t - P_t @ q) ** 2)
        if loss_name == "prob_mse":
            return anp.mean((q - emp_t) ** 2)
        q_clip = anp.clip(q, 1e-12, 1.0)
        return -anp.sum(emp_t * anp.log(q_clip))

    for _ in range(steps):
        theta, _ = opt.step_and_cost(_loss, theta)

    q_final = np.array(_q_from_theta(theta), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final = q_final / max(1e-15, float(q_final.sum()))
    return q_final.astype(np.float64)


def _metrics_rows(
    model_rows: List[Dict[str, object]],
    holdout_mask: np.ndarray,
    qH_unif: float,
    H_size: int,
    q80_thr: float,
    q80_search_max: int,
    p_star: np.ndarray,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in model_rows:
        q = row["q"]
        assert isinstance(q, np.ndarray)
        fit = _distribution_fit_metrics(q=q, p_star=p_star)
        met = hv.compute_metrics_for_q(q, holdout_mask, qH_unif, H_size, q80_thr, q80_search_max)
        out.append(
            {
                "model_key": row["key"],
                "model_label": row["label"],
                "qH": float(met["qH"]),
                "qH_ratio": float(met["qH_ratio"]),
                "Q80": float(met["Q80"]),
                "Q80_pred": float(met["Q80_pred"]),
                "R_Q1000": float(met["R_Q1000"]),
                "R_Q10000": float(met["R_Q10000"]),
                "fit_tv_to_pstar": float(fit["tv"]),
                "fit_js_dist_to_pstar": float(fit["js_dist"]),
                "fit_prob_mse_to_pstar": float(fit["prob_mse"]),
            }
        )
    return out


def _distribution_fit_metrics(q: np.ndarray, p_star: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    qv = np.asarray(q, dtype=np.float64)
    pv = np.asarray(p_star, dtype=np.float64)
    qv = np.clip(qv, eps, 1.0)
    pv = np.clip(pv, eps, 1.0)
    qv = qv / float(np.sum(qv))
    pv = pv / float(np.sum(pv))

    tv = 0.5 * float(np.sum(np.abs(qv - pv)))
    prob_mse = float(np.mean((qv - pv) ** 2))
    m = 0.5 * (qv + pv)
    kl_pm = float(np.sum(pv * np.log(pv / m)))
    kl_qm = float(np.sum(qv * np.log(qv / m)))
    js_div = 0.5 * (kl_pm + kl_qm)
    js_dist = float(np.sqrt(max(0.0, js_div)))
    return {"tv": tv, "prob_mse": prob_mse, "js_div": js_div, "js_dist": js_dist}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_recovery_all(
    p_star: np.ndarray,
    holdout_mask: np.ndarray,
    model_rows: List[Dict[str, object]],
    outpath: Path,
    title: str,
    Qmax: int = 10000,
) -> None:
    H = int(np.sum(holdout_mask))
    if H == 0:
        return

    Q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 120).astype(int)),
                np.linspace(1000, Qmax, 160).astype(int),
            ]
        )
    )
    Q = Q[Q <= Qmax]

    y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, Q)

    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)
    ax.plot(Q, y_star, color=hv.COLORS["target"], linewidth=2.0, label=r"Target $p^*$", zorder=10)

    for idx, row in enumerate(model_rows):
        q = row["q"]
        assert isinstance(q, np.ndarray)
        y = hv.expected_unique_fraction(q, holdout_mask, Q)
        ax.plot(
            Q,
            y,
            color=str(row["color"]),
            linestyle=row.get("ls", "-"),
            linewidth=float(row.get("lw", 1.9)),
            alpha=float(row.get("alpha", 1.0)),
            label=str(row["label"]),
            zorder=9 - idx,
        )

    ax.plot(Q, y_unif, color=hv.COLORS["gray"], linewidth=1.5, linestyle="--", alpha=0.9, label="Uniform", zorder=1)
    ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlim(0, Qmax)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_ylim(-0.02, 1.05)

    legend_handles = [Line2D([0], [0], color=hv.COLORS["target"], lw=2.0, label=r"Target $p^*$")]
    for row in model_rows:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=str(row["color"]),
                lw=float(row.get("lw", 1.9)),
                ls=row.get("ls", "-"),
                label=str(row["label"]),
            )
        )
    legend_handles.append(
        Line2D([0], [0], color=hv.COLORS["gray"], lw=1.5, ls="--", label="Uniform")
    )
    ax.legend(handles=legend_handles, **LEGEND_STYLE)

    fig.savefig(str(outpath))
    plt.close(fig)


def _plot_fit_distance_all(
    metrics_rows: List[Dict[str, object]],
    model_rows: List[Dict[str, object]],
    outpath: Path,
    title: str,
    metric_key: str = "fit_tv_to_pstar",
) -> None:
    if not metrics_rows:
        return

    label_by_key = {str(r["key"]): str(r["label"]) for r in model_rows}
    color_by_key = {str(r["key"]): str(r["color"]) for r in model_rows}

    rows = sorted(metrics_rows, key=lambda r: float(r[metric_key]))
    keys = [str(r["model_key"]) for r in rows]
    labels = [label_by_key.get(k, k) for k in keys]
    vals = [float(r[metric_key]) for r in rows]
    cols = [color_by_key.get(k, "#333333") for k in keys]

    fig, ax = plt.subplots(figsize=hv.fig_size("full", 3.2), constrained_layout=True)
    y = np.arange(len(vals))
    bars = ax.barh(y, vals, color=cols, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel(r"Distance to target: $D_{\mathrm{TV}}(q, p^*)$ (lower is better)")
    ax.set_title(title)

    xmax = max(vals) if vals else 1.0
    for b, v in zip(bars, vals):
        ax.text(
            float(v) + 0.01 * max(1e-9, xmax),
            b.get_y() + b.get_height() / 2.0,
            f"{v:.4f}",
            va="center",
            ha="left",
            fontsize=7,
        )

    fig.savefig(str(outpath))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--claim6-dir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "06_claim_fair_classical_baseline"),
    )
    ap.add_argument(
        "--claim7-dir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "07_claim_global_holdout_full_distribution"),
    )
    # Core config (used if legacy config.json is absent)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--good-frac", type=float, default=0.05)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--q80-search-max", type=int, default=200000)
    ap.add_argument("--iqp-steps", type=int, default=400)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--iqp-eval-every", type=int, default=50)
    ap.add_argument("--target-family", type=str, default="paper_even", choices=["paper_even", "paper"])
    ap.add_argument("--train-ms", type=str, default="1000,5000")
    ap.add_argument("--holdout-m-train", type=int, default=5000)
    # Best settings fallback (used if pair_rows/meta are absent)
    ap.add_argument("--high-seed", type=int, default=44)
    ap.add_argument("--high-train-m", type=int, default=5000)
    ap.add_argument("--high-layers", type=int, default=1)
    ap.add_argument("--high-sigma", type=float, default=2.0)
    ap.add_argument("--high-k", type=int, default=256)
    ap.add_argument("--global-seed", type=int, default=46)
    ap.add_argument("--global-train-m", type=int, default=5000)
    ap.add_argument("--global-layers", type=int, default=1)
    ap.add_argument("--global-sigma", type=float, default=2.0)
    ap.add_argument("--global-k", type=int, default=512)
    ap.add_argument("--artr-epochs", type=int, default=300)
    ap.add_argument("--artr-d-model", type=int, default=64)
    ap.add_argument("--artr-heads", type=int, default=4)
    ap.add_argument("--artr-layers", type=int, default=2)
    ap.add_argument("--artr-ff", type=int, default=128)
    ap.add_argument("--artr-lr", type=float, default=1e-3)
    ap.add_argument("--artr-batch-size", type=int, default=256)
    ap.add_argument("--maxent-steps", type=int, default=2500)
    ap.add_argument("--maxent-lr", type=float, default=5e-2)
    args = ap.parse_args()

    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required. Install it to run this script.")
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required. Install it to run this script.")

    claim6 = Path(args.claim6_dir)
    claim7 = Path(args.claim7_dir)
    claim6.mkdir(parents=True, exist_ok=True)
    claim7.mkdir(parents=True, exist_ok=True)

    cfg_raw = _load_json_if_exists(claim6 / "config.json")
    pair_rows: List[Dict[str, str]] = []
    pair_rows_path = claim6 / "pair_rows.csv"
    if pair_rows_path.exists():
        with pair_rows_path.open("r", encoding="utf-8") as f:
            pair_rows = list(csv.DictReader(f))

    hv.set_style(base=8)

    n = int(cfg_raw["n"]) if cfg_raw is not None and "n" in cfg_raw else int(args.n)
    beta = float(cfg_raw["beta"]) if cfg_raw is not None and "beta" in cfg_raw else float(args.beta)
    good_frac = float(cfg_raw["good_frac"]) if cfg_raw is not None and "good_frac" in cfg_raw else float(args.good_frac)
    holdout_k = int(cfg_raw["holdout_k"]) if cfg_raw is not None and "holdout_k" in cfg_raw else int(args.holdout_k)
    holdout_pool = int(cfg_raw["holdout_pool"]) if cfg_raw is not None and "holdout_pool" in cfg_raw else int(args.holdout_pool)
    q80_thr = float(cfg_raw["Q80_thr"]) if cfg_raw is not None and "Q80_thr" in cfg_raw else float(args.q80_thr)
    q80_search_max = int(cfg_raw["Q80_search_max"]) if cfg_raw is not None and "Q80_search_max" in cfg_raw else int(args.q80_search_max)
    iqp_steps = int(cfg_raw["iqp_steps"]) if cfg_raw is not None and "iqp_steps" in cfg_raw else int(args.iqp_steps)
    iqp_lr = float(cfg_raw["iqp_lr"]) if cfg_raw is not None and "iqp_lr" in cfg_raw else float(args.iqp_lr)
    iqp_eval_every = int(cfg_raw["iqp_eval_every"]) if cfg_raw is not None and "iqp_eval_every" in cfg_raw else int(args.iqp_eval_every)
    target_family = (
        str(cfg_raw.get("target_family", args.target_family)).strip().lower()
        if cfg_raw is not None
        else str(args.target_family).strip().lower()
    )
    if target_family == "paper":
        target_family = "paper_even"
    if target_family != "paper_even":
        raise ValueError("exp10 supports only target-family=paper_even.")
    train_ms = _parse_list_ints(str(cfg_raw["train_ms"])) if cfg_raw is not None and "train_ms" in cfg_raw else _parse_list_ints(str(args.train_ms))
    holdout_m_train_raw = cfg_raw.get("holdout_m_train") if cfg_raw is not None else None
    if holdout_m_train_raw is None:
        holdout_m_train = int(args.holdout_m_train) if args.holdout_m_train is not None else max(train_ms)
    else:
        holdout_m_train = int(holdout_m_train_raw)

    bits_table = hv.make_bits_table(n)
    p_star, support, scores = hv.build_target_distribution_paper(n, beta)
    good_mask = hv.topk_mask_by_scores(scores, support, frac=good_frac)

    for mode in ("high_value", "global"):
        seed: int
        train_m: int
        layers: int
        sigma: float
        K: int

        if pair_rows:
            best = _pick_best_row(pair_rows, mode)
            seed = int(float(best["seed"]))
            train_m = int(float(best["train_m"]))
            layers = int(float(best["layers"]))
            sigma = float(best["sigma"])
            K = int(float(best["K"]))
        else:
            outdir_mode = claim6 if mode == "high_value" else claim7
            mode_tag = "high_value" if mode == "high_value" else "global"
            meta = _load_json_if_exists(outdir_mode / f"4_recovery_best_{mode_tag}_all_baselines_meta.json")
            setting = None
            if meta is not None:
                setting_obj = meta.get("selected_setting")
                if isinstance(setting_obj, dict):
                    setting = setting_obj

            if setting is not None:
                seed = int(setting.get("seed", args.high_seed if mode == "high_value" else args.global_seed))
                train_m = int(setting.get("train_m", args.high_train_m if mode == "high_value" else args.global_train_m))
                layers = int(setting.get("layers", args.high_layers if mode == "high_value" else args.global_layers))
                sigma = float(setting.get("sigma", args.high_sigma if mode == "high_value" else args.global_sigma))
                K = int(setting.get("K", args.high_k if mode == "high_value" else args.global_k))
            else:
                if mode == "high_value":
                    seed = int(args.high_seed)
                    train_m = int(args.high_train_m)
                    layers = int(args.high_layers)
                    sigma = float(args.high_sigma)
                    K = int(args.high_k)
                else:
                    seed = int(args.global_seed)
                    train_m = int(args.global_train_m)
                    layers = int(args.global_layers)
                    sigma = float(args.global_sigma)
                    K = int(args.global_k)

        holdout_mask = _build_holdout_mask(
            mode=mode,
            p_star=p_star,
            support=support,
            good_mask=good_mask,
            bits_table=bits_table,
            m_train_for_holdout=holdout_m_train,
            holdout_k=holdout_k,
            holdout_pool=holdout_pool,
            seed=seed,
        )
        H_size = int(np.sum(holdout_mask))

        cfg = hv.Config(
            n=n,
            beta=beta,
            train_m=train_m,
            holdout_k=holdout_k,
            holdout_pool=holdout_pool,
            seed=seed,
            good_frac=good_frac,
            sigmas=[sigma],
            Ks=[K],
            Qmax=10000,
            Q80_thr=q80_thr,
            Q80_search_max=q80_search_max,
            target_family=target_family,
            adversarial=False,
            use_iqp=True,
            use_classical=True,
            iqp_steps=iqp_steps,
            iqp_lr=iqp_lr,
            iqp_eval_every=iqp_eval_every,
            iqp_layers=layers,
            iqp_loss="parity_mse",
            outdir=str(claim6 if mode == "high_value" else claim7),
        )

        # Rebuild reference models and supervision (P,z) at best setting.
        art = hv.rerun_single_setting(
            cfg=cfg,
            p_star=p_star,
            holdout_mask=holdout_mask,
            bits_table=bits_table,
            sigma=sigma,
            K=K,
            return_hist=False,
            iqp_loss="parity_mse",
        )
        q_iqp = art["q_iqp"]
        q_class = art["q_class"]
        P = art["P"]
        z_data = art["z"]
        assert isinstance(q_iqp, np.ndarray)
        assert isinstance(q_class, np.ndarray)
        assert isinstance(P, np.ndarray)
        assert isinstance(z_data, np.ndarray)

        # IQP control with prob-MSE objective.
        cfg_prob = hv.Config(**{**cfg.__dict__, "use_classical": False, "iqp_loss": "prob_mse"})
        art_prob = hv.rerun_single_setting(
            cfg=cfg_prob,
            p_star=p_star,
            holdout_mask=holdout_mask,
            bits_table=bits_table,
            sigma=sigma,
            K=K,
            return_hist=False,
            iqp_loss="prob_mse",
        )
        q_iqp_prob = art_prob["q_iqp"]
        assert isinstance(q_iqp_prob, np.ndarray)

        # Build empirical train distribution (for prob_mse/xent controls).
        p_train = p_star.copy()
        if H_size > 0:
            p_train[holdout_mask] = 0.0
            p_train /= p_train.sum()
        idxs_train = hv.sample_indices(p_train, train_m, seed=seed + 7)
        emp = hv.empirical_dist(idxs_train, p_star.size)

        q_nnn_fields_parity = _train_classical_boltzmann(
            n=n,
            layers=layers,
            steps=iqp_steps,
            lr=iqp_lr,
            seed_init=seed + 30001,
            P=P,
            z_data=z_data,
            loss_mode="parity_mse",
            emp_dist=emp,
            topology="nn_nnn",
            include_fields=True,
        )
        q_dense_fields_xent = _train_classical_boltzmann(
            n=n,
            layers=layers,
            steps=iqp_steps,
            lr=iqp_lr,
            seed_init=seed + 30004,
            P=P,
            z_data=z_data,
            loss_mode="xent",
            emp_dist=emp,
            topology="dense",
            include_fields=True,
        )
        q_transformer_mle = _train_transformer_autoregressive(
            bits_table=bits_table,
            idxs_train=idxs_train,
            n=n,
            seed=seed + 35501,
            epochs=args.artr_epochs,
            d_model=args.artr_d_model,
            nhead=args.artr_heads,
            num_layers=args.artr_layers,
            dim_ff=args.artr_ff,
            lr=args.artr_lr,
            batch_size=args.artr_batch_size,
        )
        q_maxent_parity = _train_maxent_parity(
            P=P,
            z_data=z_data,
            seed=seed + 36001,
            steps=args.maxent_steps,
            lr=args.maxent_lr,
        )

        model_rows: List[Dict[str, object]] = [
            {
                "key": "iqp_parity_mse",
                "label": "IQP (parity)",
                "q": q_iqp,
                "color": hv.COLORS["model"],
                "ls": "-",
                "lw": 2.2,
            },
            {
                "key": "iqp_prob_mse",
                "label": "IQP (prob-MSE)",
                "q": q_iqp_prob,
                "color": hv.COLORS["model_prob_mse"],
                "ls": "--",
                "lw": 2.0,
            },
            {
                "key": "classical_nnn_fields_parity",
                "label": "Ising+fields (NN+NNN)",
                "q": q_nnn_fields_parity,
                "color": "#005A9C",
                "ls": "-",
                "lw": 1.9,
            },
            {
                "key": "classical_dense_fields_xent",
                "label": "Dense Ising+fields (xent)",
                "q": q_dense_fields_xent,
                "color": "#8C564B",
                "ls": (0, (5, 2)),
                "lw": 1.9,
            },
            {
                "key": "classical_transformer_mle",
                "label": "AR Transformer (MLE)",
                "q": q_transformer_mle,
                "color": "#1AA7A1",
                "ls": "--",
                "lw": 2.0,
            },
            {
                "key": "classical_maxent_parity",
                "label": "MaxEnt parity (P,z)",
                "q": q_maxent_parity,
                "color": "#9467BD",
                "ls": "--",
                "lw": 2.1,
            },
        ]

        outdir = claim6 if mode == "high_value" else claim7
        pdf_path = outdir / f"4_recovery_best_{mode}_all_baselines.pdf"
        csv_path = outdir / f"4_recovery_best_{mode}_all_baselines_metrics.csv"
        fit_pdf_path = outdir / f"5_fit_distance_to_target_{mode}_all_baselines.pdf"
        fit_csv_path = outdir / f"5_fit_distance_to_target_{mode}_all_baselines.csv"
        meta_path = outdir / f"4_recovery_best_{mode}_all_baselines_meta.json"

        _plot_recovery_all(
            p_star=p_star,
            holdout_mask=holdout_mask,
            model_rows=model_rows,
            outpath=pdf_path,
            title=f"{mode} holdout | m={train_m}, L={layers}, sigma={sigma:g}, K={K}",
            Qmax=cfg.Qmax,
        )

        q_unif = np.ones_like(p_star) / p_star.size
        qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0
        m_rows = _metrics_rows(
            model_rows=model_rows,
            holdout_mask=holdout_mask,
            qH_unif=qH_unif,
            H_size=H_size,
            q80_thr=q80_thr,
            q80_search_max=q80_search_max,
            p_star=p_star,
        )
        _write_csv(csv_path, m_rows)
        fit_rows = sorted(m_rows, key=lambda r: float(r["fit_tv_to_pstar"]))
        _write_csv(fit_csv_path, fit_rows)
        _plot_fit_distance_all(
            metrics_rows=m_rows,
            model_rows=model_rows,
            outpath=fit_pdf_path,
            title=f"{mode} holdout | fit to target distribution",
            metric_key="fit_tv_to_pstar",
        )

        meta = {
            "holdout_mode": mode,
            "best_selection_source": "pair_rows.csv best finite IQP Q80 (tie-break qH_ratio_iqp)",
            "selected_setting": {
                "seed": seed,
                "train_m": train_m,
                "layers": layers,
                "sigma": sigma,
                "K": K,
            },
            "holdout": {
                "size": H_size,
                "p_star_holdout": float(p_star[holdout_mask].sum()),
            },
            "files": {
                "plot_pdf": str(pdf_path),
                "metrics_csv": str(csv_path),
                "fit_plot_pdf": str(fit_pdf_path),
                "fit_metrics_csv": str(fit_csv_path),
            },
            "selected_classical_baselines": [
                "classical_nnn_fields_parity",
                "classical_dense_fields_xent",
                "classical_transformer_mle",
                "classical_maxent_parity",
            ],
            "autoregressive_transformer_model": {
                "family": "Causal Transformer",
                "objective": "MLE",
                "epochs": int(args.artr_epochs),
                "d_model": int(args.artr_d_model),
                "n_heads": int(args.artr_heads),
                "n_layers": int(args.artr_layers),
                "dim_ff": int(args.artr_ff),
                "lr": float(args.artr_lr),
                "batch_size": int(args.artr_batch_size),
            },
            "maxent_parity_model": {
                "family": "Exponential family over parity features",
                "objective": "MaxEnt dual moment matching",
                "steps": int(args.maxent_steps),
                "lr": float(args.maxent_lr),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[Saved] {pdf_path}")
        print(f"[Saved] {csv_path}")
        print(f"[Saved] {fit_pdf_path}")
        print(f"[Saved] {fit_csv_path}")
        print(f"[Saved] {meta_path}")


if __name__ == "__main__":
    main()
