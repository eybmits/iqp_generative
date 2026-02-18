#!/usr/bin/env python3
"""Single-run standard IQP-QCBM pipeline with coverage + fit plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iqp_generative import core as hv


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = ROOT / "outputs" / "test_standard_iqp"


def _normalize(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = float(np.sum(p))
    if s <= eps:
        return np.ones_like(p, dtype=np.float64) / max(1, p.size)
    return p / s


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(p)
    q = _normalize(q)
    return 0.5 * float(np.sum(np.abs(p - q)))


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize(p)
    q = _normalize(q)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _build_holdout_mask(
    mode: str,
    p_star: np.ndarray,
    support: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    holdout_m_train: int,
    holdout_k: int,
    holdout_pool: int,
    seed: int,
) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "global":
        candidate = support.astype(bool)
    elif mode == "high_value":
        candidate = good_mask
    else:
        raise ValueError("holdout-mode must be 'global' or 'high_value'.")

    return hv.select_holdout_smart(
        p_star=p_star,
        good_mask=candidate,
        bits_table=bits_table,
        m_train=int(holdout_m_train),
        holdout_k=int(holdout_k),
        pool_size=int(holdout_pool),
        seed=int(seed) + 111,
    )


def _build_cvfs_roi_mask(
    idxs_train: np.ndarray,
    N: int,
    roi_size: int,
    split_frac: float,
    seed: int,
    policy: str = "rare",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a pseudo-unseen ROI from a train/train split (A/B) without target leakage:
      - policy='rare': prioritize states seen in A but unseen in B, then low-frequency in B
      - policy='frequent': prioritize high-frequency states in B (and A as tie-break)
      - policy='mixed': blend frequent + rare halves
    """
    idxs = np.asarray(idxs_train, dtype=np.int64)
    if idxs.size < 2:
        raise ValueError("Need at least 2 training samples for CVFS split.")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(idxs.size)
    cut = int(np.clip(round(float(split_frac) * idxs.size), 1, idxs.size - 1))
    idxs_a = idxs[perm[:cut]]
    idxs_b = idxs[perm[cut:]]

    c_a = np.bincount(idxs_a, minlength=int(N))
    c_b = np.bincount(idxs_b, minlength=int(N))

    unseen_in_b = np.where((c_a > 0) & (c_b == 0))[0]
    total_seen = np.where((c_a + c_b) > 0)[0]

    k = int(np.clip(roi_size, 1, N))
    chosen = []
    policy = str(policy).lower()

    if policy == "frequent":
        # Focus on high-probability region inferred from training split.
        score = (c_b.astype(np.float64) + 0.5 * c_a.astype(np.float64))
        pool = np.where(score > 0)[0]
        if pool.size == 0:
            pool = np.arange(N, dtype=np.int64)
        order = np.argsort(-(score[pool] + 1e-6 * rng.random(pool.size)))
        chosen = pool[order[:k]].tolist()
    elif policy == "mixed":
        k_hi = max(1, int(round(0.5 * k)))
        k_lo = max(1, k - k_hi)
        score_hi = (c_b.astype(np.float64) + 0.5 * c_a.astype(np.float64))
        pool_hi = np.where(score_hi > 0)[0]
        if pool_hi.size == 0:
            pool_hi = np.arange(N, dtype=np.int64)
        order_hi = np.argsort(-(score_hi[pool_hi] + 1e-6 * rng.random(pool_hi.size)))
        chosen.extend(pool_hi[order_hi[:k_hi]].tolist())

        # Rare half
        if unseen_in_b.size > 0:
            w = c_a[unseen_in_b].astype(np.float64)
            order = np.argsort(w + 1e-6 * rng.random(w.size))
            chosen.extend(unseen_in_b[order[:k_lo]].tolist())
        if len(chosen) < k:
            chosen_set = set(chosen)
            pool = np.array([i for i in total_seen.tolist() if i not in chosen_set], dtype=np.int64)
            if pool.size > 0:
                wb = c_b[pool].astype(np.float64)
                order = np.argsort(wb + 1e-6 * rng.random(wb.size))
                chosen.extend(pool[order[: (k - len(chosen))]].tolist())
    else:
        # Default: rare policy.
        if unseen_in_b.size > 0:
            if unseen_in_b.size <= k:
                chosen.extend(unseen_in_b.tolist())
            else:
                w = c_a[unseen_in_b].astype(np.float64)
                order = np.argsort(w + 1e-6 * rng.random(w.size))
                chosen.extend(unseen_in_b[order[:k]].tolist())
        if len(chosen) < k and total_seen.size > 0:
            needed = k - len(chosen)
            chosen_set = set(chosen)
            pool = np.array([i for i in total_seen.tolist() if i not in chosen_set], dtype=np.int64)
            if pool.size > 0:
                wb = c_b[pool].astype(np.float64)
                order = np.argsort(wb + 1e-6 * rng.random(wb.size))
                chosen.extend(pool[order[:needed]].tolist())

    if len(chosen) == 0:
        # Fallback: use all seen states if split degenerates.
        if total_seen.size == 0:
            total_seen = np.unique(idxs)
        take = int(min(k, total_seen.size))
        order = np.argsort(c_b[total_seen] + 1e-6 * rng.random(total_seen.size))
        chosen = total_seen[order[:take]].tolist()

    roi_mask = np.zeros(int(N), dtype=bool)
    roi_mask[np.asarray(chosen, dtype=np.int64)] = True

    info = {
        "roi_policy": str(policy),
        "split_cut": int(cut),
        "size_a": int(idxs_a.size),
        "size_b": int(idxs_b.size),
        "roi_size": int(np.sum(roi_mask)),
        "unseen_in_b_pool": int(unseen_in_b.size),
        "total_seen_union": int(total_seen.size),
    }
    return roi_mask, info


def _train_iqp_from_alphas(
    *,
    args: argparse.Namespace,
    alphas: np.ndarray,
    bits_table: np.ndarray,
    emp: np.ndarray,
    holdout_mask: np.ndarray,
    qH_unif: float,
    H_size: int,
    seed_init: int,
) -> Tuple[np.ndarray, float, Optional[Dict[str, np.ndarray]], Dict[str, float]]:
    P = hv.build_parity_matrix(alphas, bits_table)
    z = P @ emp
    q_iqp, train_loss, hist = hv.train_iqp_qcbm(
        n=int(args.n),
        layers=int(args.iqp_layers),
        steps=int(args.iqp_steps),
        lr=float(args.iqp_lr),
        P=P,
        z_data=z,
        seed_init=int(seed_init),
        eval_every=int(args.iqp_eval_every),
        return_hist=True,
        loss_mode="parity_mse",
        xent_emp=emp,
    )
    metrics = hv.compute_metrics_for_q(
        q=q_iqp,
        holdout_mask=holdout_mask,
        qH_unif=qH_unif,
        H_size=H_size,
        Q80_thr=float(args.Q80_thr),
        Q80_search_max=int(args.Q80_search_max),
    )
    return q_iqp, float(train_loss), hist, metrics


def _plot_coverage(
    outpath: Path,
    Q: np.ndarray,
    y_target: np.ndarray,
    y_iqp: np.ndarray,
    y_uniform: np.ndarray,
    title: str,
    y_cvfs: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.8), constrained_layout=True)
    ax.plot(Q, y_target, color=hv.COLORS["target"], linewidth=2.1, label="Target")
    ax.plot(Q, y_iqp, color=hv.COLORS["model"], linewidth=2.1, label="IQP-QCBM (random parities)")
    if y_cvfs is not None:
        ax.plot(
            Q,
            y_cvfs,
            color=hv.COLORS["model_prob_mse"],
            linewidth=2.1,
            label="IQP-QCBM (CVFS parities)",
        )
    ax.plot(Q, y_uniform, color=hv.COLORS["gray"], linewidth=1.8, linestyle="--", label="Uniform")
    ax.set_xscale("log")
    ax.set_xlim(max(1, int(np.min(Q))), int(np.max(Q)))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Q (sample budget)")
    ax.set_ylabel("Recovery R(Q)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(outpath)
    plt.close(fig)


def _plot_fit(
    outpath: Path,
    fit: Dict[str, float],
    include_cvfs: bool = False,
) -> None:
    labels = ["Target", "Train empirical"]
    x = np.arange(len(labels))
    w = 0.26 if include_cvfs else 0.36

    tv_iqp = [fit["tv_to_target_iqp"], fit["tv_to_train_iqp"]]
    tv_uni = [fit["tv_to_target_uniform"], fit["tv_to_train_uniform"]]
    tv_cvfs = [fit["tv_to_target_cvfs"], fit["tv_to_train_cvfs"]] if include_cvfs else None

    kl_iqp = [fit["kl_to_target_iqp"], fit["kl_to_train_iqp"]]
    kl_uni = [fit["kl_to_target_uniform"], fit["kl_to_train_uniform"]]
    kl_cvfs = [fit["kl_to_target_cvfs"], fit["kl_to_train_cvfs"]] if include_cvfs else None

    fig, axes = plt.subplots(1, 2, figsize=hv.fig_size("full", 2.9), constrained_layout=True)

    ax = axes[0]
    if include_cvfs:
        ax.bar(x - w, tv_iqp, width=w, color=hv.COLORS["model"], alpha=0.9, label="IQP random")
        ax.bar(x, tv_cvfs, width=w, color=hv.COLORS["model_prob_mse"], alpha=0.9, label="IQP CVFS")
        ax.bar(x + w, tv_uni, width=w, color=hv.COLORS["gray"], alpha=0.8, label="Uniform")
    else:
        ax.bar(x - w / 2, tv_iqp, width=w, color=hv.COLORS["model"], alpha=0.9, label="IQP")
        ax.bar(x + w / 2, tv_uni, width=w, color=hv.COLORS["gray"], alpha=0.8, label="Uniform")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("TV distance")
    ax.set_title("Fit (TV)")
    ax.legend(frameon=False)

    ax = axes[1]
    if include_cvfs:
        ax.bar(x - w, kl_iqp, width=w, color=hv.COLORS["model"], alpha=0.9, label="IQP random")
        ax.bar(x, kl_cvfs, width=w, color=hv.COLORS["model_prob_mse"], alpha=0.9, label="IQP CVFS")
        ax.bar(x + w, kl_uni, width=w, color=hv.COLORS["gray"], alpha=0.8, label="Uniform")
    else:
        ax.bar(x - w / 2, kl_iqp, width=w, color=hv.COLORS["model"], alpha=0.9, label="IQP")
        ax.bar(x + w / 2, kl_uni, width=w, color=hv.COLORS["gray"], alpha=0.8, label="Uniform")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("KL divergence")
    ax.set_title("Fit (KL)")

    fig.savefig(outpath)
    plt.close(fig)


def _run_cmd(cmd: List[str], cwd: Path = ROOT) -> None:
    cmd_s = " ".join(shlex.quote(x) for x in cmd)
    print(f"[run] {cmd_s}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        tail = "\n".join(proc.stdout.strip().splitlines()[-20:])
        if tail:
            print(tail)
    if proc.returncode != 0:
        err_tail = "\n".join(proc.stderr.strip().splitlines()[-60:])
        if err_tail:
            print(err_tail)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd_s}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_nature_comms_profile(args: argparse.Namespace) -> None:
    root_out = ROOT / "outputs" / "paper_even_final"
    root_out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    exp39_out = root_out / "39_claim_loss_ablation_nature"
    exp40_global_out = root_out / "40_claim_visibility_causal_global"
    exp40_high_out = root_out / "40_claim_visibility_causal_high_value"
    exp41_out = root_out / "41_claim_fairness_report"
    exp43_out = root_out / "34_claim_beta_sweep_bestparams"
    exp99_out = root_out / "99_stats_tables"
    fig_out = ROOT / "outputs" / "paper_figures_nature_v1"

    smoke = bool(args.smoke)
    extra_smoke_39: List[str] = ["--smoke", "--steps", "120", "--n-boot", "300"] if smoke else []
    extra_smoke_40: List[str] = ["--smoke", "--steps", "120", "--n-perm", "1000"] if smoke else []
    extra_smoke_38: List[str] = ["--n-perm", "1000", "--n-boot", "1000"] if smoke else []

    _run_cmd(
        [
            py,
            str(ROOT / "experiments" / "legacy" / "exp39_loss_ablation_nature.py"),
            "--outdir",
            str(exp39_out),
            "--global-sigma",
            "1",
            "--global-K",
            "512",
            "--high-sigma",
            "2",
            "--high-K",
            "256",
            *extra_smoke_39,
        ]
    )

    _run_cmd(
        [
            py,
            str(ROOT / "experiments" / "legacy" / "exp40_visibility_causal.py"),
            "--outdir",
            str(exp40_global_out),
            "--holdout-mode",
            "global",
            "--sigma",
            "1",
            "--K",
            "512",
            *extra_smoke_40,
        ]
    )
    _run_cmd(
        [
            py,
            str(ROOT / "experiments" / "legacy" / "exp40_visibility_causal.py"),
            "--outdir",
            str(exp40_high_out),
            "--holdout-mode",
            "high_value",
            "--sigma",
            "2",
            "--K",
            "256",
            *extra_smoke_40,
        ]
    )

    _run_cmd(
        [
            py,
            str(ROOT / "experiments" / "legacy" / "exp41_fairness_baseline_report.py"),
            "--outdir",
            str(exp41_out),
        ]
    )

    # Full run only: execute the fair classical baseline matrix across (mode, m, beta, seed).
    if not smoke:
        _run_cmd(
            [
                py,
                str(ROOT / "experiments" / "legacy" / "exp43_fair_baselines_matrix.py"),
                "--outdir",
                str(exp43_out),
                "--holdout-modes",
                "global,high_value",
                "--train-ms",
                "200,1000,5000",
                "--betas",
                "0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4",
                "--seeds",
                "42,43,44,45,46",
            ]
        )

    _run_cmd(
        [
            py,
            str(ROOT / "experiments" / "legacy" / "exp38_stats_aggregate.py"),
            "--input-root",
            str(root_out),
            "--outdir",
            str(exp99_out),
            "--manifest",
            str(ROOT / "docs" / "eval_manifest_v1.md"),
            *extra_smoke_38,
        ]
    )

    _run_cmd(
        [
            py,
            str(ROOT / "experiments" / "legacy" / "exp42_make_nature_figure_set.py"),
            "--input-root",
            str(root_out),
            "--outdir",
            str(fig_out),
        ]
    )

    expected = [
        exp39_out / "loss_ablation_metrics_long.csv",
        exp39_out / "forest_q80_ratio_parity_vs_refs.pdf",
        exp40_global_out / "visibility_causal_metrics.csv",
        exp40_high_out / "visibility_causal_metrics.csv",
        exp41_out / "model_budget_table.csv",
        exp41_out / "fairness_report.md",
        exp99_out / "main_table.csv",
        exp99_out / "supp_table.csv",
        exp99_out / "significance_report.md",
        fig_out / "fig1_discovery_axis_budget_law.pdf",
        fig_out / "fig2_loss_ablation_parity_vs_mmd_nll.pdf",
        fig_out / "fig3_visibility_simplex_mechanism.pdf",
        fig_out / "fig4_fair_baselines_robustness_beta_m.pdf",
        fig_out / "extended_data_index.md",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        miss_lines = "\n".join(f"- {p}" for p in missing)
        raise RuntimeError(f"Missing expected artifacts:\n{miss_lines}")

    hash_lines = []
    for p in expected:
        if p.suffix.lower() == ".pdf":
            hash_lines.append(f"{_sha256(p)}  {p}")
    hash_path = fig_out / "main_figure_hashes.sha256"
    hash_path.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    print("[Done] nature_comms_v1 profile completed.")
    print(f"[Artifacts] {root_out}")
    print(f"[Figures] {fig_out}")
    print(f"[Hashes] {hash_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run IQP evaluation pipelines. "
            "Use --profile nature_comms_v1 for one-command Nature package reproduction."
        )
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="standard",
        choices=["standard", "nature_comms_v1"],
        help="Pipeline profile: standard single run or full nature package.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run reduced workload for quick validation (only for nature_comms_v1 profile).",
    )
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))

    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--train-m", type=int, default=1000)
    parser.add_argument("--good-frac", type=float, default=0.05)

    parser.add_argument("--holdout-mode", type=str, default="global", choices=["global", "high_value"])
    parser.add_argument("--holdout-k", type=int, default=20)
    parser.add_argument("--holdout-pool", type=int, default=400)
    parser.add_argument("--holdout-m-train", type=int, default=5000)

    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--with-cvfs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cvfs-mix-frac", type=float, default=0.5)
    parser.add_argument("--cvfs-split-frac", type=float, default=0.5)
    parser.add_argument("--cvfs-roi-size", type=int, default=128)
    parser.add_argument("--cvfs-candidate-pool", type=int, default=8192)
    parser.add_argument("--cvfs-diversity-gamma", type=float, default=0.15)
    parser.add_argument(
        "--cvfs-roi-policy",
        type=str,
        default="auto",
        choices=["auto", "rare", "frequent", "mixed"],
    )

    parser.add_argument("--iqp-steps", type=int, default=600)
    parser.add_argument("--iqp-lr", type=float, default=0.05)
    parser.add_argument("--iqp-eval-every", type=int, default=50)
    parser.add_argument("--iqp-layers", type=int, default=1)

    parser.add_argument("--Qmax", type=int, default=10000)
    parser.add_argument("--Q80-thr", type=float, default=0.8)
    parser.add_argument("--Q80-search-max", type=int, default=200000)

    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    if str(args.profile).lower() == "nature_comms_v1":
        _run_nature_comms_profile(args)
        return

    if not hv.HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required. Install with: pip install pennylane")

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bits_table = hv.make_bits_table(int(args.n))
    p_star, support, scores = hv.build_target_distribution_paper(int(args.n), float(args.beta))
    good_mask = hv.topk_mask_by_scores(scores, support, frac=float(args.good_frac))

    holdout_mask = _build_holdout_mask(
        mode=str(args.holdout_mode),
        p_star=p_star,
        support=support,
        good_mask=good_mask,
        bits_table=bits_table,
        holdout_m_train=int(args.holdout_m_train),
        holdout_k=int(args.holdout_k),
        holdout_pool=int(args.holdout_pool),
        seed=int(args.seed),
    )

    N = int(p_star.size)
    H_size = int(np.sum(holdout_mask))

    p_train = p_star.copy()
    if H_size > 0:
        p_train[holdout_mask] = 0.0
        p_train = _normalize(p_train)

    idxs_train = hv.sample_indices(p_train, int(args.train_m), seed=int(args.seed) + 7)
    emp = hv.empirical_dist(idxs_train, N)

    q_unif = np.ones(N, dtype=np.float64) / N
    qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0
    base_seed_init = int(args.seed) + 10000 + 7 * int(args.K)

    alphas_random = hv.sample_alphas(int(args.n), float(args.sigma), int(args.K), seed=int(args.seed) + 222)
    q_iqp, train_loss, hist, iqp_metrics = _train_iqp_from_alphas(
        args=args,
        alphas=alphas_random,
        bits_table=bits_table,
        emp=emp,
        holdout_mask=holdout_mask,
        qH_unif=qH_unif,
        H_size=H_size,
        seed_init=base_seed_init,
    )

    q_cvfs = None
    train_loss_cvfs = None
    hist_cvfs = None
    cvfs_metrics = None
    cvfs_info = None
    cvfs_roi_info = None
    if bool(args.with_cvfs):
        roi_policy = str(args.cvfs_roi_policy).lower()
        if roi_policy == "auto":
            roi_policy = "frequent" if str(args.holdout_mode).lower() == "high_value" else "rare"
        roi_mask_cvfs, cvfs_roi_info = _build_cvfs_roi_mask(
            idxs_train=idxs_train,
            N=N,
            roi_size=int(args.cvfs_roi_size),
            split_frac=float(args.cvfs_split_frac),
            seed=int(args.seed) + 333,
            policy=roi_policy,
        )
        mix = float(np.clip(args.cvfs_mix_frac, 0.0, 1.0))
        K_active = int(round(int(args.K) * mix))
        K_active = max(1, min(int(args.K), K_active))
        K_random = int(args.K) - K_active

        alphas_active, cvfs_info = hv.select_alphas_active(
            n=int(args.n),
            sigma=float(args.sigma),
            K=int(K_active),
            seed=int(args.seed) + 444,
            bits_table=bits_table,
            roi_mask=roi_mask_cvfs,
            candidate_pool=int(args.cvfs_candidate_pool),
            diversity_gamma=float(args.cvfs_diversity_gamma),
            normalize_by_full_space=True,
        )
        if K_random > 0:
            alphas_rand_tail = hv.sample_alphas(
                int(args.n),
                float(args.sigma),
                int(K_random),
                seed=int(args.seed) + 445,
            )
            alphas_cvfs = np.vstack([alphas_active, alphas_rand_tail]).astype(np.int8, copy=False)
        else:
            alphas_cvfs = alphas_active.astype(np.int8, copy=False)

        # Deduplicate once and refill by random masks to keep exactly K features.
        alphas_cvfs = np.unique(alphas_cvfs, axis=0)
        refill_iter = 0
        while alphas_cvfs.shape[0] < int(args.K):
            refill_iter += 1
            add = hv.sample_alphas(
                int(args.n),
                float(args.sigma),
                int(args.K) - int(alphas_cvfs.shape[0]),
                seed=int(args.seed) + 446 + 7919 * refill_iter,
            )
            alphas_cvfs = np.unique(np.vstack([alphas_cvfs, add]), axis=0)
        if alphas_cvfs.shape[0] > int(args.K):
            alphas_cvfs = alphas_cvfs[: int(args.K)]

        cvfs_info = dict(cvfs_info or {})
        cvfs_info["mix_frac"] = float(mix)
        cvfs_info["K_active"] = int(K_active)
        cvfs_info["K_random"] = int(K_random)
        cvfs_info["K_unique_final"] = int(alphas_cvfs.shape[0])

        q_cvfs, train_loss_cvfs, hist_cvfs, cvfs_metrics = _train_iqp_from_alphas(
            args=args,
            alphas=alphas_cvfs,
            bits_table=bits_table,
            emp=emp,
            holdout_mask=holdout_mask,
            qH_unif=qH_unif,
            H_size=H_size,
            seed_init=base_seed_init,
        )

    Q = np.unique(np.rint(np.geomspace(1, max(2, int(args.Qmax)), 180)).astype(int))
    Q = Q[Q >= 1]

    y_target = hv.expected_unique_fraction(p_star, holdout_mask, Q) if H_size > 0 else np.zeros_like(Q, dtype=np.float64)
    y_iqp = hv.expected_unique_fraction(q_iqp, holdout_mask, Q) if H_size > 0 else np.zeros_like(Q, dtype=np.float64)
    y_cvfs = hv.expected_unique_fraction(q_cvfs, holdout_mask, Q) if (H_size > 0 and q_cvfs is not None) else None
    y_uniform = hv.expected_unique_fraction(q_unif, holdout_mask, Q) if H_size > 0 else np.zeros_like(Q, dtype=np.float64)

    coverage_title = f"Coverage | mode={args.holdout_mode}, n={args.n}, m={args.train_m}, sigma={args.sigma}, K={args.K}"
    _plot_coverage(
        outpath=outdir / "coverage_recovery_curve.pdf",
        Q=Q,
        y_target=y_target,
        y_iqp=y_iqp,
        y_cvfs=y_cvfs,
        y_uniform=y_uniform,
        title=coverage_title,
    )

    fit = {
        "tv_to_target_iqp": _tv(p_star, q_iqp),
        "tv_to_train_iqp": _tv(emp, q_iqp),
        "tv_to_target_uniform": _tv(p_star, q_unif),
        "tv_to_train_uniform": _tv(emp, q_unif),
        "kl_to_target_iqp": _kl(p_star, q_iqp),
        "kl_to_train_iqp": _kl(emp, q_iqp),
        "kl_to_target_uniform": _kl(p_star, q_unif),
        "kl_to_train_uniform": _kl(emp, q_unif),
    }
    if q_cvfs is not None:
        fit["tv_to_target_cvfs"] = _tv(p_star, q_cvfs)
        fit["tv_to_train_cvfs"] = _tv(emp, q_cvfs)
        fit["kl_to_target_cvfs"] = _kl(p_star, q_cvfs)
        fit["kl_to_train_cvfs"] = _kl(emp, q_cvfs)

    _plot_fit(outpath=outdir / "fit_distance_plot.pdf", fit=fit, include_cvfs=(q_cvfs is not None))

    coverage_rows = [{
        "Q": int(q),
        "R_target": float(rt),
        "R_iqp": float(ri),
        "R_cvfs": (float(rc) if y_cvfs is not None else np.nan),
        "R_uniform": float(ru),
    } for q, rt, ri, ru, rc in zip(
        Q.tolist(),
        y_target.tolist(),
        y_iqp.tolist(),
        y_uniform.tolist(),
        (y_cvfs.tolist() if y_cvfs is not None else [None] * len(Q)),
    )]

    with (outdir / "coverage_curve.csv").open("w", encoding="utf-8") as f:
        f.write("Q,R_target,R_iqp,R_cvfs,R_uniform\n")
        for r in coverage_rows:
            f.write(
                f"{r['Q']},{r['R_target']:.12g},{r['R_iqp']:.12g},"
                f"{r['R_cvfs']:.12g},{r['R_uniform']:.12g}\n"
            )

    result = {
        "config": vars(args),
        "holdout_size": int(H_size),
        "p_star_holdout": float(p_star[holdout_mask].sum()) if H_size > 0 else 0.0,
        "train_loss_iqp_random": float(train_loss),
        "train_loss_iqp_cvfs": (float(train_loss_cvfs) if train_loss_cvfs is not None else None),
        "iqp_metrics_random": {k: float(v) for k, v in iqp_metrics.items()},
        "iqp_metrics_cvfs": ({k: float(v) for k, v in cvfs_metrics.items()} if cvfs_metrics is not None else None),
        "cvfs_selection_info": cvfs_info,
        "cvfs_roi_info": cvfs_roi_info,
        "fit_metrics": {k: float(v) for k, v in fit.items()},
        "outputs": {
            "coverage_plot": str(outdir / "coverage_recovery_curve.pdf"),
            "fit_plot": str(outdir / "fit_distance_plot.pdf"),
            "coverage_csv": str(outdir / "coverage_curve.csv"),
        },
    }

    with (outdir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if hist is not None:
        hv.plot_training_dynamics_generic(
            hist=hist,
            outpath=str(outdir / "train_dynamics_iqp_random.pdf"),
            color=hv.COLORS["model"],
            ylab="Moment MSE loss",
        )
    if hist_cvfs is not None:
        hv.plot_training_dynamics_generic(
            hist=hist_cvfs,
            outpath=str(outdir / "train_dynamics_iqp_cvfs.pdf"),
            color=hv.COLORS["model_prob_mse"],
            ylab="Moment MSE loss",
        )

    print("[Done] Standard IQP baseline pipeline completed.")
    print(f"[Outdir] {outdir}")
    print(f"[Q80 random] {iqp_metrics['Q80']}")
    print(f"[qH_ratio random] {iqp_metrics['qH_ratio']:.4f}")
    if cvfs_metrics is not None:
        print(f"[Q80 cvfs] {cvfs_metrics['Q80']}")
        print(f"[qH_ratio cvfs] {cvfs_metrics['qH_ratio']:.4f}")


if __name__ == "__main__":
    main()
