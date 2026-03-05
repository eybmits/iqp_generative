#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strong autoregressive Transformer baseline vs IQP snapshot at n=12.

This script trains an AR Transformer on sample data from p_train and evaluates
recovery exactly on the full 2^n table.
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BOS_TOKEN = 2
VOCAB_SIZE = 3


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _beta_tag(beta: float) -> str:
    return f"{float(beta):.1f}".replace(".", "p")


def _int_to_bits(value: int, n: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(n - 1, -1, -1)], dtype=np.int8)


def _longest_zero_run_between_ones(bits: np.ndarray) -> int:
    ones = np.flatnonzero(bits == 1)
    if ones.size < 2:
        return 0
    gaps = ones[1:] - ones[:-1] - 1
    return int(np.max(gaps)) if gaps.size else 0


def _build_target_distribution(n: int, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    jitter = rng.uniform(0.0, 1e-12, size=cand.size)
    order = np.argsort(-(pvals + jitter))
    pool = cand[order[: min(int(holdout_pool), cand.size)]]
    if pool.size < holdout_k:
        raise RuntimeError("Smart holdout pool smaller than holdout_k.")

    pool_bits = bits_table[pool].astype(np.int8)
    pool_p = p_star[pool]
    selected: list[int] = []
    available = np.ones(pool.size, dtype=bool)

    first = int(np.argmax(pool_p))
    selected.append(first)
    available[first] = False
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
        pick = int(rng.choice(cand_idx)) if cand_idx.size > 1 else int(cand_idx[0])
        selected.append(pick)
        available[pick] = False
        dist_pick = np.sum(np.abs(pool_bits - pool_bits[pick]), axis=1).astype(np.int64)
        dmin = np.where(available, np.minimum(dmin, dist_pick), dmin)
        dmin[pick] = -1

    return np.asarray(pool[np.asarray(selected, dtype=np.int64)], dtype=np.int64)


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
    N, _ = bits_table.shape
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
        m = bucket_id == b
        bucket_sizes[b] = int(np.sum(m))
        p_bucket[b] = float(np.sum(p_star[m]))
    p_bucket = p_bucket / max(1e-15, float(np.sum(p_bucket)))
    return bucket_id, bucket_sizes, p_bucket


def _compute_tv_score(q_state: np.ndarray, bucket_id: np.ndarray, p_bucket: np.ndarray) -> float:
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


def _recovery_from_q(q_state: np.ndarray, holdout_idx: np.ndarray, Q: int) -> float:
    if holdout_idx.size == 0:
        return float("nan")
    qh = q_state[holdout_idx]
    rec = 1.0 - np.power(np.clip(1.0 - qh, 0.0, 1.0), int(Q))
    return float(np.mean(rec))


class ARTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_bits: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_bits = int(n_bits)
        self.d_model = int(d_model)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_bits, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ff_mult) * self.d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, 2)
        self.register_buffer("causal_mask", torch.triu(torch.ones(self.n_bits, self.n_bits, dtype=torch.bool), diagonal=1))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, tokens_in: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(tokens_in) + self.pos_emb[:, : tokens_in.shape[1], :]
        x = self.encoder(x, mask=self.causal_mask[: tokens_in.shape[1], : tokens_in.shape[1]])
        x = self.norm(x)
        return self.head(x)


@dataclass
class TrainConfig:
    n: int
    d_model: int
    n_layers: int
    n_heads: int
    ff_mult: int
    dropout: float
    steps: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    eval_interval: int
    device: str


def _device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_ar_training_tensors(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # bits: [N, n] with 0/1 ints
    n = bits.shape[1]
    inp = np.zeros((bits.shape[0], n), dtype=np.int64)
    inp[:, 0] = BOS_TOKEN
    if n > 1:
        inp[:, 1:] = bits[:, :-1].astype(np.int64)
    tgt = bits.astype(np.int64)
    return inp, tgt


def _apply_lr_schedule(opt: torch.optim.Optimizer, base_lr: float, step: int, total_steps: int, warmup_steps: int) -> None:
    if step <= warmup_steps:
        scale = float(step) / float(max(1, warmup_steps))
    else:
        p = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        p = min(max(p, 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * p))
    lr = float(base_lr) * scale
    for g in opt.param_groups:
        g["lr"] = lr


def _cross_entropy_bits(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, 2), tgt.reshape(-1))


def _eval_nll(model: ARTransformer, inp: torch.Tensor, tgt: torch.Tensor, batch_size: int) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i in range(0, inp.shape[0], batch_size):
            j = min(i + batch_size, inp.shape[0])
            logits = model(inp[i:j])
            loss = _cross_entropy_bits(logits, tgt[i:j])
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_transformer(
    *,
    cfg: TrainConfig,
    train_bits: np.ndarray,
    val_bits: np.ndarray,
    seed: int,
) -> tuple[ARTransformer, dict]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    device = torch.device(cfg.device)
    model = ARTransformer(
        n_bits=cfg.n,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    train_in_np, train_tg_np = _build_ar_training_tensors(train_bits.astype(np.int64))
    val_in_np, val_tg_np = _build_ar_training_tensors(val_bits.astype(np.int64))
    train_in = torch.tensor(train_in_np, dtype=torch.long, device=device)
    train_tg = torch.tensor(train_tg_np, dtype=torch.long, device=device)
    val_in = torch.tensor(val_in_np, dtype=torch.long, device=device)
    val_tg = torch.tensor(val_tg_np, dtype=torch.long, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    n_train = train_in.shape[0]
    for step in range(1, cfg.steps + 1):
        _apply_lr_schedule(opt, cfg.lr, step, cfg.steps, cfg.warmup_steps)
        idx = torch.randint(0, n_train, (cfg.batch_size,), device=device)
        logits = model(train_in[idx])
        loss = _cross_entropy_bits(logits, train_tg[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
        opt.step()

        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
            val_nll = _eval_nll(model, val_in, val_tg, batch_size=max(256, cfg.batch_size))
            if val_nll < best_val:
                best_val = val_nll
                best_state = copy.deepcopy(model.state_dict())
            lr_now = float(opt.param_groups[0]["lr"])
            print(
                f"[train] step={step:4d}/{cfg.steps} "
                f"train_ce={float(loss.detach().cpu().item()):.6f} "
                f"val_ce={val_nll:.6f} lr={lr_now:.2e}"
            )

    model.load_state_dict(best_state)
    return model, {"best_val_ce": best_val}


@torch.no_grad()
def exact_q_table(
    *,
    model: ARTransformer,
    bits_table: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    inp_np, tgt_np = _build_ar_training_tensors(bits_table.astype(np.int64))
    inp = torch.tensor(inp_np, dtype=torch.long, device=device)
    tgt = torch.tensor(tgt_np, dtype=torch.long, device=device)
    out = np.zeros(bits_table.shape[0], dtype=np.float64)
    for i in range(0, inp.shape[0], int(batch_size)):
        j = min(i + int(batch_size), inp.shape[0])
        logits = model(inp[i:j])
        logp = F.log_softmax(logits, dim=-1)
        gathered = logp.gather(-1, tgt[i:j].unsqueeze(-1)).squeeze(-1)
        seq_logp = torch.sum(gathered, dim=1)
        out[i:j] = torch.exp(seq_logp).cpu().numpy().astype(np.float64)
    s = float(np.sum(out))
    if s <= 0.0:
        raise RuntimeError("Exact q-table sum is non-positive.")
    return out / s


def _iter_rows(df: pd.DataFrame) -> Iterable[dict]:
    for _, row in df.iterrows():
        yield dict(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Strong AR-Transformer baseline vs IQP at n=12.")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--seeds", type=str, default="101,102,103,104,105,106,107,108,109,110,111,112")
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--val-m", type=int, default=200)
    ap.add_argument("--holdout-k", type=int, default=20)
    ap.add_argument("--holdout-protocol", type=str, default="global_smart", choices=["global_smart", "global_random"])
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--holdout-m", type=int, default=5000)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--ff-mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=120)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--q-eval", type=int, default=1000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument(
        "--recovery-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig6_beta_sweep_recovery_grid" / "fig6_data_default.npz"),
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "claims" / "transformer_vs_iqp_n12"),
    )
    args = ap.parse_args()

    if args.d_model % args.n_heads != 0:
        raise ValueError(f"d-model ({args.d_model}) must be divisible by n-heads ({args.n_heads}).")

    seeds = _parse_ints(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided.")

    device = _device_auto() if args.device == "auto" else args.device
    print(
        f"[info] device={device} n={args.n} beta={args.beta} seeds={seeds} "
        f"arch=({args.d_model}d,{args.n_layers}L,{args.n_heads}H)"
    )

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
        raise ValueError("Invalid Q grid.")

    rows: list[dict] = []
    curves_seed = np.zeros((len(seeds), q_grid.size), dtype=np.float64)

    cfg = TrainConfig(
        n=int(args.n),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        ff_mult=int(args.ff_mult),
        dropout=float(args.dropout),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        grad_clip=float(args.grad_clip),
        eval_interval=int(args.eval_interval),
        device=device,
    )

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
            holdout_idx = _sample_holdout_global(
                support_idx=support_idx,
                holdout_k=int(args.holdout_k),
                rng=rng,
            )

        p_train = p_star.copy()
        p_train[holdout_idx] = 0.0
        p_train = p_train / max(1e-15, float(np.sum(p_train)))

        train_states = _sample_empirical_train(p_train=p_train, train_m=int(args.train_m), rng=rng)
        val_states = _sample_empirical_train(p_train=p_train, train_m=int(args.val_m), rng=rng)
        train_bits = bits_table[train_states].astype(np.int64)
        val_bits = bits_table[val_states].astype(np.int64)

        model, train_info = train_transformer(
            cfg=cfg,
            train_bits=train_bits,
            val_bits=val_bits,
            seed=int(seed),
        )
        q_state = exact_q_table(model=model, bits_table=bits_table, device=device, batch_size=1024)

        qeval_states = np.asarray(rng.choice(N, size=int(args.q_eval), replace=True, p=q_state), dtype=np.int64)
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
            curves_seed[seed_idx] = np.mean(
                1.0 - np.power(np.clip(1.0 - qh[None, :], 0.0, 1.0), q_grid[:, None]),
                axis=1,
            )
        else:
            curves_seed[seed_idx] = np.nan

        rows.append(
            {
                "model_key": "transformer_strong_ar",
                "model_label": "Transformer AR (strong)",
                "n": int(args.n),
                "beta": float(args.beta),
                "seed": int(seed),
                "train_m": int(args.train_m),
                "val_m": int(args.val_m),
                "holdout_k": int(args.holdout_k),
                "holdout_protocol": str(args.holdout_protocol),
                "holdout_pool": int(args.holdout_pool),
                "holdout_m": int(args.holdout_m),
                "d_model": int(args.d_model),
                "n_layers": int(args.n_layers),
                "n_heads": int(args.n_heads),
                "ff_mult": int(args.ff_mult),
                "dropout": float(args.dropout),
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "best_val_ce": float(train_info["best_val_ce"]),
                "TV_score": tv_score,
                "BSHS": bshs,
                "Composite": composite,
                "q_holdout": q_holdout,
                "R_Q1000": r_q1000,
                "R_Q10000": r_q10000,
                "q_eval": int(args.q_eval),
            }
        )
        print(
            "[metrics] "
            f"TV_score={tv_score:.6f} "
            f"BSHS={bshs:.6f} "
            f"Composite={composite:.6f} "
            f"q_holdout={q_holdout:.6f} "
            f"R_Q10000={r_q10000:.6f} "
            f"best_val_ce={float(train_info['best_val_ce']):.6f}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = _beta_tag(float(args.beta))

    df = pd.DataFrame(rows)
    seed_csv = outdir / f"transformer_seed_metrics_n12_beta{tag}.csv"
    df.to_csv(seed_csv, index=False)

    overlay_csv = outdir / f"transformer_overlay_baseline_schema_beta{tag}.csv"
    df[["n", "model_key", "q_holdout", "R_Q10000", "seed"]].to_csv(overlay_csv, index=False)

    summary_cols = ["TV_score", "BSHS", "Composite", "q_holdout", "R_Q1000", "R_Q10000", "best_val_ce"]
    summary = df[summary_cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    summary["model_key"] = "transformer_strong_ar"
    summary_csv = outdir / f"transformer_summary_n12_beta{tag}.csv"
    summary.to_csv(summary_csv, index=False)

    curve_npz = outdir / f"transformer_recovery_curves_n12_beta{tag}.npz"
    np.savez_compressed(
        curve_npz,
        beta=float(args.beta),
        Q=q_grid,
        seeds=np.asarray(seeds, dtype=np.int64),
        transformer_curves_seed=curves_seed,
        transformer_curve_mean=np.nanmean(curves_seed, axis=0),
        transformer_curve_std=np.nanstd(curves_seed, axis=0),
    )

    print(f"\n[saved] {seed_csv}")
    print(f"[saved] {overlay_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {curve_npz}")
    print("\n[summary]")
    for row in _iter_rows(summary):
        print(
            f"  {row['model_key']:>22s} | {row['metric']:<10s} "
            f"mean={float(row['mean']):.6f} std={float(row['std']):.6f}"
        )


if __name__ == "__main__":
    main()

