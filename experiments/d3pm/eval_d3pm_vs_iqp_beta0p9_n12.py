#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a simple binary D3PM at n=12, beta=0.9 and compare to IQP baseline.

This script is intentionally focused:
  - target setup: n=12, beta=0.9
  - train size: m=200 (matching the beta=0.9 fig3 setting)
  - metrics: TV_score, BSHS(Q), Composite, q_holdout, R_Q10000
  - comparison source: frozen fig3 CSV (IQP parity baseline)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _int_to_bits(value: int, n: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(n - 1, -1, -1)], dtype=np.int8)


def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in bits.tolist():
        out = (out << 1) | int(b)
    return int(out)


def _longest_zero_run_between_ones(bits: np.ndarray) -> int:
    ones = np.flatnonzero(bits == 1)
    if ones.size < 2:
        return 0
    gaps = ones[1:] - ones[:-1] - 1
    return int(np.max(gaps)) if gaps.size else 0


def _build_target_distribution(n: int, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper target p*: even parity support with exp(beta * score)."""
    N = 2 ** n
    bits_table = np.zeros((N, n), dtype=np.int8)
    support = np.zeros(N, dtype=bool)
    scores = np.full(N, -100.0, dtype=np.float64)
    for x in range(N):
        bits = _int_to_bits(x, n)
        bits_table[x] = bits
        if int(np.sum(bits)) % 2 == 0:
            support[x] = True
            scores[x] = 1.0 + float(_longest_zero_run_between_ones(bits))

    logits = np.full(N, -np.inf, dtype=np.float64)
    logits[support] = float(beta) * scores[support]
    max_logit = float(np.max(logits[support]))
    unnorm = np.zeros(N, dtype=np.float64)
    unnorm[support] = np.exp(logits[support] - max_logit)
    p_star = unnorm / max(1e-15, float(np.sum(unnorm)))
    return p_star, support, bits_table


def _sample_holdout_global(
    *,
    support_idx: np.ndarray,
    holdout_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if holdout_k <= 0:
        return np.array([], dtype=np.int64)
    if holdout_k > support_idx.size:
        raise ValueError(f"holdout_k={holdout_k} exceeds support size={support_idx.size}")
    return np.asarray(rng.choice(support_idx, size=holdout_k, replace=False), dtype=np.int64)


def _sample_holdout_global_smart(
    *,
    support_idx: np.ndarray,
    p_star: np.ndarray,
    bits_table: np.ndarray,
    holdout_k: int,
    holdout_pool: int,
    holdout_m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Global+Smart holdout construction (methodology-aligned approximation)."""
    if holdout_k <= 0:
        return np.array([], dtype=np.int64)
    if holdout_k > support_idx.size:
        raise ValueError(f"holdout_k={holdout_k} exceeds support size={support_idx.size}")

    tau_list = [1.0 / float(holdout_m), 0.5 / float(holdout_m), 0.25 / float(holdout_m), 0.0]
    cand = support_idx.copy()
    for tau in tau_list:
        m = support_idx[p_star[support_idx] >= float(tau)]
        if m.size >= holdout_k:
            cand = m
            if m.size >= holdout_pool:
                break

    pvals = p_star[cand]
    # Random jitter for stable stochastic tie-breaking per seed.
    jitter = rng.uniform(0.0, 1e-12, size=cand.size)
    order = np.argsort(-(pvals + jitter))
    pool = cand[order[: min(int(holdout_pool), cand.size)]]
    if pool.size < holdout_k:
        raise RuntimeError("Smart holdout pool smaller than holdout_k.")

    pool_bits = bits_table[pool].astype(np.int8)
    pool_p = p_star[pool]

    selected: list[int] = []
    available = np.ones(pool.size, dtype=bool)

    # Start from highest-probability candidate.
    first = int(np.argmax(pool_p))
    selected.append(first)
    available[first] = False

    # Min Hamming distance to selected set.
    dmin = np.sum(np.abs(pool_bits - pool_bits[first]), axis=1).astype(np.int64)
    dmin[first] = -1

    while len(selected) < holdout_k:
        idxs = np.flatnonzero(available)
        if idxs.size == 0:
            break
        dvals = dmin[idxs]
        best_d = int(np.max(dvals))
        cand_idx = idxs[dvals == best_d]
        if cand_idx.size > 1:
            pv = pool_p[cand_idx]
            best_p = float(np.max(pv))
            cand_idx = cand_idx[np.isclose(pv, best_p)]
        if cand_idx.size > 1:
            pick = int(rng.choice(cand_idx))
        else:
            pick = int(cand_idx[0])

        selected.append(pick)
        available[pick] = False
        dist_pick = np.sum(np.abs(pool_bits - pool_bits[pick]), axis=1).astype(np.int64)
        dmin = np.where(available, np.minimum(dmin, dist_pick), dmin)
        dmin[pick] = -1

    sel = pool[np.asarray(selected, dtype=np.int64)]
    return np.asarray(sel, dtype=np.int64)


def _sample_empirical_train(
    *,
    p_train: np.ndarray,
    train_m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    N = p_train.shape[0]
    return np.asarray(rng.choice(N, size=train_m, replace=True, p=p_train), dtype=np.int64)


def _prepare_score_buckets(
    *,
    p_star: np.ndarray,
    support_mask: np.ndarray,
    bits_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bucket_id per state, bucket_sizes, and p_star bucket masses."""
    N, n = bits_table.shape
    scores = np.full(N, -1, dtype=np.int64)
    for x in range(N):
        if support_mask[x]:
            scores[x] = _longest_zero_run_between_ones(bits_table[x])
    levels = np.unique(scores[scores >= 0])
    level_to_bucket = {int(lv): i for i, lv in enumerate(levels.tolist())}
    bucket_id = np.full(N, -1, dtype=np.int64)
    for x in range(N):
        s = int(scores[x])
        if s >= 0:
            bucket_id[x] = level_to_bucket[s]
    num_buckets = len(levels)
    bucket_sizes = np.zeros(num_buckets, dtype=np.int64)
    p_bucket = np.zeros(num_buckets, dtype=np.float64)
    for b in range(num_buckets):
        mask = bucket_id == b
        bucket_sizes[b] = int(np.sum(mask))
        p_bucket[b] = float(np.sum(p_star[mask]))
    p_bucket = p_bucket / max(1e-15, float(np.sum(p_bucket)))
    return bucket_id, bucket_sizes, p_bucket


def _compute_tv_score(
    *,
    q_state: np.ndarray,
    bucket_id: np.ndarray,
    p_bucket: np.ndarray,
) -> float:
    q_bucket = np.zeros_like(p_bucket)
    for b in range(p_bucket.shape[0]):
        q_bucket[b] = float(np.sum(q_state[bucket_id == b]))
    return 0.5 * float(np.sum(np.abs(q_bucket - p_bucket)))


def _compute_bshs_from_samples(
    *,
    sample_states: np.ndarray,
    bucket_id: np.ndarray,
    bucket_sizes: np.ndarray,
    p_bucket: np.ndarray,
) -> float:
    uniq = np.unique(sample_states.astype(np.int64))
    hit = np.zeros_like(p_bucket)
    for b in range(p_bucket.shape[0]):
        if bucket_sizes[b] <= 0:
            continue
        in_b = np.sum(bucket_id[uniq] == b)
        hit[b] = float(in_b) / float(bucket_sizes[b])
    return float(np.sum(p_bucket * hit))


def _recovery_from_q(
    *,
    q_state: np.ndarray,
    holdout_idx: np.ndarray,
    Q: int,
) -> float:
    if holdout_idx.size == 0:
        return float("nan")
    qh = q_state[holdout_idx]
    rec = 1.0 - np.power(np.clip(1.0 - qh, 0.0, 1.0), int(Q))
    return float(np.mean(rec))


class TimeMLP(nn.Module):
    def __init__(self, n_bits: int, timesteps: int, hidden_dim: int = 128, time_dim: int = 32) -> None:
        super().__init__()
        self.n_bits = int(n_bits)
        self.time_embed = nn.Embedding(int(timesteps) + 1, int(time_dim))
        self.net = nn.Sequential(
            nn.Linear(self.n_bits + int(time_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.n_bits),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x_in = torch.cat([x_t * 2.0 - 1.0, t_emb], dim=-1)
        return self.net(x_in)


@dataclass
class D3PMConfig:
    n: int
    timesteps: int
    beta_start: float
    beta_end: float
    hidden_dim: int
    lr: float
    steps: int
    batch_size: int
    device: str


class BinaryD3PM:
    def __init__(self, cfg: D3PMConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = TimeMLP(
            n_bits=cfg.n,
            timesteps=cfg.timesteps,
            hidden_dim=cfg.hidden_dim,
            time_dim=32,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        beta_steps = np.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, dtype=np.float64)
        beta_full = np.zeros(cfg.timesteps + 1, dtype=np.float64)
        beta_full[1:] = beta_steps
        flip_odd = np.zeros(cfg.timesteps + 1, dtype=np.float64)
        prod = 1.0
        for t in range(1, cfg.timesteps + 1):
            prod *= (1.0 - 2.0 * beta_full[t])
            flip_odd[t] = 0.5 * (1.0 - prod)
        self.beta_full = torch.tensor(beta_full, dtype=torch.float32, device=self.device)
        self.flip_odd = torch.tensor(flip_odd, dtype=torch.float32, device=self.device)

    def _sample_xt_given_x0(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p_flip = self.flip_odd[t].unsqueeze(1)
        flips = (torch.rand_like(x0) < p_flip).float()
        xt = torch.remainder(x0 + flips, 2.0)
        return xt

    def _posterior_prob_xprev_eq1(self, *, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        a = self.flip_odd[t - 1].unsqueeze(1)  # P(x_{t-1} != x0)
        b = self.beta_full[t].unsqueeze(1)  # step-t flip prob

        qprev1 = torch.where(x0 > 0.5, 1.0 - a, a)
        qprev0 = 1.0 - qprev1
        lik1 = torch.where(xt > 0.5, 1.0 - b, b)
        lik0 = 1.0 - lik1
        num = qprev1 * lik1
        den = num + qprev0 * lik0 + eps
        return num / den

    def train_from_bits(self, train_bits_np: np.ndarray, *, seed: int) -> None:
        torch.manual_seed(int(seed))
        x_train = torch.tensor(train_bits_np.astype(np.float32), dtype=torch.float32, device=self.device)
        n_train = int(x_train.shape[0])

        self.model.train()
        for step in range(1, self.cfg.steps + 1):
            idx = torch.randint(0, n_train, (self.cfg.batch_size,), device=self.device)
            x0 = x_train[idx]
            t = torch.randint(1, self.cfg.timesteps + 1, (self.cfg.batch_size,), device=self.device)
            xt = self._sample_xt_given_x0(x0=x0, t=t)
            logits = self.model(xt, t)
            p_pred = torch.sigmoid(logits)
            p_post = self._posterior_prob_xprev_eq1(x0=x0, xt=xt, t=t)
            p_pred = torch.clamp(p_pred, 1e-6, 1.0 - 1e-6)
            p_post = torch.clamp(p_post, 1e-6, 1.0 - 1e-6)
            # KL(q || p_theta) per bit, mean-reduced.
            kl = p_post * (torch.log(p_post) - torch.log(p_pred)) + (1.0 - p_post) * (
                torch.log(1.0 - p_post) - torch.log(1.0 - p_pred)
            )
            loss = torch.mean(kl)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            if step == 1 or step % 100 == 0 or step == self.cfg.steps:
                print(f"[train] step={step:4d}/{self.cfg.steps} loss={float(loss.detach().cpu().item()):.6f}")

    @torch.no_grad()
    def sample_states(
        self,
        *,
        num_samples: int,
        seed: int,
        batch_size: int = 8192,
    ) -> np.ndarray:
        self.model.eval()
        out: list[np.ndarray] = []
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        weights = (2 ** torch.arange(self.cfg.n - 1, -1, -1, device=self.device, dtype=torch.int64)).view(1, -1)
        remain = int(num_samples)
        while remain > 0:
            bs = min(batch_size, remain)
            x = torch.bernoulli(0.5 * torch.ones((bs, self.cfg.n), device=self.device), generator=gen)
            for t in range(self.cfg.timesteps, 0, -1):
                tt = torch.full((bs,), t, dtype=torch.long, device=self.device)
                probs = torch.sigmoid(self.model(x, tt))
                x = torch.bernoulli(probs, generator=gen)
            states = torch.sum(x.to(torch.int64) * weights, dim=1)
            out.append(states.cpu().numpy().astype(np.int64))
            remain -= bs
        return np.concatenate(out, axis=0)


def _device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _iter_rows(df: pd.DataFrame) -> Iterable[dict]:
    for _, row in df.iterrows():
        yield dict(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run D3PM vs IQP (n=12, beta=0.9) evaluation.")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-protocol", type=str, default="global_smart", choices=["global_smart", "global_random"])
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m", type=int, default=5000)
    ap.add_argument("--seeds", type=str, default="101,102,103,104,105,106,107,108,109,110,111,112")
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--beta-start", type=float, default=0.01)
    ap.add_argument("--beta-end", type=float, default=0.20)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-samples", type=int, default=200000)
    ap.add_argument("--q-eval", type=int, default=1000)
    ap.add_argument(
        "--recovery-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig6_beta_sweep_recovery_grid" / "fig6_data_default.npz"),
        help="NPZ that provides the Q grid for recovery curves (uses key 'Q').",
    )
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument(
        "--iqp-csv",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "final_plots"
            / "fig3_tv_bshs_seedmean_scatter"
            / "tv_bshs_points_multiseed_beta_q1000_no_iqp_mse_beta0p9_newseeds12.csv"
        ),
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "d3pm_vs_iqp_n12_beta0p9"),
    )
    args = ap.parse_args()

    seeds = _parse_ints(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided.")

    device = _device_auto() if args.device == "auto" else args.device
    print(f"[info] device={device} n={args.n} beta={args.beta} seeds={seeds}")

    p_star, support_mask, bits_table = _build_target_distribution(n=int(args.n), beta=float(args.beta))
    support_idx = np.flatnonzero(support_mask)
    N = p_star.shape[0]
    print(f"[info] state_space={N} support={support_idx.size}")

    bucket_id, bucket_sizes, p_bucket = _prepare_score_buckets(
        p_star=p_star,
        support_mask=support_mask,
        bits_table=bits_table,
    )

    rec_npz = Path(args.recovery_npz)
    if rec_npz.exists():
        with np.load(rec_npz, allow_pickle=True) as zrec:
            q_grid = np.asarray(zrec["Q"], dtype=np.int64) if "Q" in zrec.files else np.arange(0, 10001, 40, dtype=np.int64)
    else:
        q_grid = np.arange(0, 10001, 40, dtype=np.int64)
    if q_grid.ndim != 1 or q_grid.size == 0:
        raise ValueError("Invalid recovery Q grid.")

    rows: list[dict] = []
    d3pm_curves_seed = np.zeros((len(seeds), q_grid.size), dtype=np.float64)
    for seed_idx, seed in enumerate(seeds):
        print(f"\n[run] seed={seed}")
        rng = np.random.default_rng(int(seed))
        if args.holdout_protocol == "global_smart":
            holdout_idx = _sample_holdout_global_smart(
                support_idx=support_idx,
                p_star=p_star,
                bits_table=bits_table,
                holdout_k=int(args.holdout_k),
                holdout_pool=int(args.holdout_pool),
                holdout_m=int(args.holdout_m),
                rng=rng,
            )
        else:
            holdout_idx = _sample_holdout_global(support_idx=support_idx, holdout_k=int(args.holdout_k), rng=rng)
        p_train = p_star.copy()
        p_train[holdout_idx] = 0.0
        p_train = p_train / max(1e-15, float(np.sum(p_train)))

        train_states = _sample_empirical_train(p_train=p_train, train_m=int(args.train_m), rng=rng)
        train_bits = bits_table[train_states].astype(np.float32)

        cfg = D3PMConfig(
            n=int(args.n),
            timesteps=int(args.timesteps),
            beta_start=float(args.beta_start),
            beta_end=float(args.beta_end),
            hidden_dim=int(args.hidden_dim),
            lr=float(args.lr),
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            device=device,
        )
        model = BinaryD3PM(cfg)
        model.train_from_bits(train_bits, seed=int(seed))

        eval_states = model.sample_states(num_samples=int(args.eval_samples), seed=int(seed) + 1000)
        q_counts = np.bincount(eval_states, minlength=N).astype(np.float64)
        q_state = q_counts / max(1e-15, float(np.sum(q_counts)))

        qeval_states = model.sample_states(num_samples=int(args.q_eval), seed=int(seed) + 2000)
        tv_score = _compute_tv_score(q_state=q_state, bucket_id=bucket_id, p_bucket=p_bucket)
        bshs = _compute_bshs_from_samples(
            sample_states=qeval_states,
            bucket_id=bucket_id,
            bucket_sizes=bucket_sizes,
            p_bucket=p_bucket,
        )
        composite = float(bshs * (1.0 - tv_score))

        q_holdout = float(np.sum(q_state[holdout_idx])) if holdout_idx.size else float("nan")
        r_q10000 = _recovery_from_q(q_state=q_state, holdout_idx=holdout_idx, Q=10000)
        r_q1000 = _recovery_from_q(q_state=q_state, holdout_idx=holdout_idx, Q=1000)
        if holdout_idx.size:
            qh = np.clip(q_state[holdout_idx], 0.0, 1.0)
            d3pm_curves_seed[seed_idx] = np.mean(
                1.0 - np.power(np.clip(1.0 - qh[None, :], 0.0, 1.0), q_grid[:, None]),
                axis=1,
            )
        else:
            d3pm_curves_seed[seed_idx] = np.nan

        rows.append(
            {
                "model_key": "d3pm_standard",
                "model_label": "D3PM (MLP)",
                "n": int(args.n),
                "beta": float(args.beta),
                "seed": int(seed),
                "train_m": int(args.train_m),
                "holdout_k": int(args.holdout_k),
                "holdout_protocol": str(args.holdout_protocol),
                "holdout_pool": int(args.holdout_pool),
                "holdout_m": int(args.holdout_m),
                "timesteps": int(args.timesteps),
                "steps": int(args.steps),
                "TV_score": tv_score,
                "BSHS": bshs,
                "Composite": composite,
                "q_holdout": q_holdout,
                "R_Q1000": r_q1000,
                "R_Q10000": r_q10000,
                "q_eval": int(args.q_eval),
                "eval_samples": int(args.eval_samples),
            }
        )
        print(
            "[metrics] "
            f"TV_score={tv_score:.6f} "
            f"BSHS={bshs:.6f} "
            f"Composite={composite:.6f} "
            f"q_holdout={q_holdout:.6f} "
            f"R_Q10000={r_q10000:.6f}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d3pm_df = pd.DataFrame(rows)
    d3pm_path = outdir / "d3pm_seed_metrics_n12_beta0p9.csv"
    d3pm_df.to_csv(d3pm_path, index=False)

    # CSV compatible with existing fig7 baseline-overlay schema.
    overlay_df = d3pm_df[["n", "model_key", "q_holdout", "R_Q10000", "seed"]].copy()
    overlay_path = outdir / "d3pm_overlay_baseline_schema.csv"
    overlay_df.to_csv(overlay_path, index=False)

    summary_cols = ["TV_score", "BSHS", "Composite", "q_holdout", "R_Q1000", "R_Q10000"]
    d3pm_summary = d3pm_df[summary_cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    d3pm_summary["model_key"] = "d3pm_standard"

    iqp_path = Path(args.iqp_csv)
    if iqp_path.exists():
        iqp_df = pd.read_csv(iqp_path)
        iqp_sub = iqp_df[
            (iqp_df["model_key"] == "iqp_parity_mse")
            & np.isclose(iqp_df["beta"].astype(float).to_numpy(), float(args.beta))
        ].copy()
        if not iqp_sub.empty:
            iqp_summary = iqp_sub[["TV_score", "BSHS", "Composite"]].agg(["mean", "std"]).T.reset_index()
            iqp_summary = iqp_summary.rename(columns={"index": "metric"})
            iqp_summary["model_key"] = "iqp_parity_mse"
            combined = pd.concat(
                [d3pm_summary[["model_key", "metric", "mean", "std"]], iqp_summary[["model_key", "metric", "mean", "std"]]],
                axis=0,
                ignore_index=True,
            )
        else:
            combined = d3pm_summary[["model_key", "metric", "mean", "std"]].copy()
    else:
        combined = d3pm_summary[["model_key", "metric", "mean", "std"]].copy()

    summary_path = outdir / "comparison_summary.csv"
    combined.to_csv(summary_path, index=False)

    curve_npz = outdir / "d3pm_recovery_curves_n12_beta0p9.npz"
    np.savez_compressed(
        curve_npz,
        Q=q_grid,
        seeds=np.asarray(seeds, dtype=np.int64),
        d3pm_curves_seed=d3pm_curves_seed,
        d3pm_curve_mean=np.nanmean(d3pm_curves_seed, axis=0),
        d3pm_curve_std=np.nanstd(d3pm_curves_seed, axis=0),
    )

    print(f"\n[saved] {d3pm_path}")
    print(f"[saved] {overlay_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {curve_npz}")
    print("\n[summary]")
    for row in _iter_rows(combined):
        print(
            f"  {row['model_key']:>15s} | {row['metric']:<10s} "
            f"mean={float(row['mean']):.6f} std={float(row['std']):.6f}"
        )


if __name__ == "__main__":
    main()
