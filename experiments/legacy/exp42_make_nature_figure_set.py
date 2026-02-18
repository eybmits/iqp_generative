#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble a compact 4-figure Nature-style set from the curated outputs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _pick_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 4 main figures for nature package.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_figures_nature_v1"),
    )
    ap.add_argument(
        "--input-root",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final"),
    )
    args = ap.parse_args()

    hv.set_style(base=8)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inp = Path(args.input_root)

    # Fig1: Discovery-axis + budget law (reuse validated scatter if available).
    fig1_src = _pick_existing(
        [
            inp / "35_claim_budgetlaw_global_m200_custom" / "budgetlaw_scatter_global_m200_iqp_sigmak_sweep_overlay_no2x5x.pdf",
            inp / "02_claim_budget_law" / "fig1_recovery_paper_overlay.pdf",
        ]
    )
    if fig1_src is not None:
        shutil.copy2(fig1_src, outdir / "fig1_discovery_axis_budget_law.pdf")
    else:
        # Fallback: build a clean discovery panel from loss-ablation recovery curves.
        p39_curves = inp / "39_claim_loss_ablation_nature" / "loss_ablation_recovery_curves_long.csv"
        if p39_curves.exists():
            d = pd.read_csv(p39_curves)
            sub = d[(d["holdout_mode"] == "global") & (d["m"] == 200)]
            if not sub.empty:
                mean = sub.groupby(["model", "Q"], as_index=False)["R"].mean()
                fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
                for model, color, lw, ls in [
                    ("target", "#111111", 2.2, "-"),
                    ("iqp_parity", "#C40000", 2.7, "-"),
                    ("iqp_mmd", "#1F77B4", 2.0, "-"),
                    ("iqp_xent", "#4C78A8", 2.0, "--"),
                    ("uniform", "#6E6E6E", 1.7, "--"),
                ]:
                    cur = mean[mean["model"] == model].sort_values("Q")
                    if cur.empty:
                        continue
                    ax.plot(cur["Q"], cur["R"], color=color, linewidth=lw, linestyle=ls, label=model)
                ax.set_xlim(0, 10000)
                ax.set_ylim(0.0, 1.02)
                ax.set_xlabel(r"$Q$ samples from model")
                ax.set_ylabel(r"Recovery $R(Q)$")
                ax.grid(True, alpha=0.14, linestyle="--")
                ax.legend(frameon=False, fontsize=7, loc="lower right")
                _save(fig, outdir / "fig1_discovery_axis_budget_law.pdf")

    # Fig2: Loss ablation parity vs MMD vs NLL from exp39 metrics.
    p39 = inp / "39_claim_loss_ablation_nature" / "loss_ablation_metrics_long.csv"
    if p39.exists():
        df = pd.read_csv(p39)
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), constrained_layout=True)

        # panel a: Q80 by train_m (averaged over beta+seed) for global
        g = df[df["holdout_mode"] == "global"].groupby(["m", "loss"], as_index=False)["Q80"].mean()
        for loss, color, ls in [("parity_mse", "#C40000", "-"), ("mmd", "#1F77B4", "-"), ("xent", "#4C78A8", "--")]:
            cur = g[g["loss"] == loss].sort_values("m")
            if cur.empty:
                continue
            axes[0].plot(cur["m"], cur["Q80"], marker="o", color=color, linestyle=ls, linewidth=2.0, label=loss)
        axes[0].set_xlabel("train m")
        axes[0].set_ylabel("Q80")
        axes[0].set_title("Global holdout")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.14, linestyle="--")
        axes[0].legend(
            handles=[
                plt.Line2D([0], [0], color="#C40000", lw=2.0, label="IQP Parity"),
                plt.Line2D([0], [0], color="#1F77B4", lw=2.0, label="IQP MMD"),
                plt.Line2D([0], [0], color="#4C78A8", lw=2.0, ls="--", label="IQP NLL"),
            ],
            frameon=False,
            fontsize=7,
            loc="upper right",
        )

        # panel b: forest ratio from exp39
        p_forest = inp / "39_claim_loss_ablation_nature" / "forest_q80_ratio_rows.csv"
        if p_forest.exists():
            ff = pd.read_csv(p_forest).sort_values(["holdout_mode", "m", "comparison"]).reset_index(drop=True)
            y = np.arange(ff.shape[0])
            x = ff["ratio_mean"].to_numpy(np.float64)
            lo = x - ff["ratio_ci_lo"].to_numpy(np.float64)
            hi = ff["ratio_ci_hi"].to_numpy(np.float64) - x
            axes[1].errorbar(x, y, xerr=np.vstack([lo, hi]), fmt="o", color="#222222", capsize=2, lw=1.0)
            axes[1].axvline(1.0, color="#777777", linestyle="--", linewidth=1.0)
            axes[1].set_yticks(y)
            axes[1].set_yticklabels([f"{r['holdout_mode']}, m={int(r['m'])}, {r['comparison']}" for _, r in ff.iterrows()], fontsize=6.5)
            axes[1].set_xlabel("Q80 ratio (Parity / Ref)")
            axes[1].grid(True, axis="x", alpha=0.14, linestyle="--")
            axes[1].set_title("Parity advantage summary")

        _save(fig, outdir / "fig2_loss_ablation_parity_vs_mmd_nll.pdf")

    # Fig3: Mechanistic causality from exp40 (accept legacy+split output locations).
    p40_candidates = [
        inp / "40_claim_visibility_causal" / "visibility_causal_metrics.csv",
        inp / "40_claim_visibility_causal_global" / "visibility_causal_metrics.csv",
        inp / "40_claim_visibility_causal_high_value" / "visibility_causal_metrics.csv",
    ]
    vis_frames = []
    for p40 in p40_candidates:
        if not p40.exists():
            continue
        d = pd.read_csv(p40)
        if "holdout_mode" not in d.columns:
            name = str(p40.parent.name).lower()
            mode = "high_value" if "high" in name else "global"
            d = d.copy()
            d["holdout_mode"] = mode
        vis_frames.append(d)
    if vis_frames:
        df = pd.concat(vis_frames, ignore_index=True)
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), constrained_layout=True)

        order = ["low", "mid", "high"]
        mode_colors = {"global": "#C40000", "high_value": "#1F77B4"}
        g = (
            df.groupby(["holdout_mode", "holdout_quantile"], as_index=False)["Q80"]
            .mean()
            .copy()
        )
        g["rank"] = g["holdout_quantile"].map({k: i for i, k in enumerate(order)})
        x = np.arange(len(order))
        for mode in sorted(g["holdout_mode"].dropna().unique().tolist()):
            cur = g[g["holdout_mode"] == mode].sort_values("rank")
            if cur.empty:
                continue
            axes[0].plot(
                cur["rank"],
                cur["Q80"],
                "o-",
                color=mode_colors.get(mode, "#444444"),
                linewidth=2.0,
                label=mode.replace("_", " "),
            )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(order)
        axes[0].set_ylabel("Q80")
        axes[0].set_title("Visibility quantile -> discovery cost")
        axes[0].grid(True, alpha=0.14, linestyle="--")
        axes[0].legend(frameon=False, fontsize=7, loc="upper right")

        for mode in sorted(df["holdout_mode"].dropna().unique().tolist()):
            cur = df[df["holdout_mode"] == mode]
            axes[1].scatter(
                cur["simplex_fillin_holdout"],
                cur["qh_gain_over_uniform"],
                color=mode_colors.get(mode, "#444444"),
                s=18,
                alpha=0.85,
                label=mode.replace("_", " "),
            )
        axes[1].set_xlabel("Simplex fill-in (holdout)")
        axes[1].set_ylabel("q(H) gain over uniform")
        axes[1].set_title("Simplex fill-in mechanism")
        axes[1].grid(True, alpha=0.14, linestyle="--")
        axes[1].legend(frameon=False, fontsize=7, loc="best")
        _save(fig, outdir / "fig3_visibility_simplex_mechanism.pdf")

    # Fig4: Fair baselines + robustness over beta/m from existing summary CSVs.
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), constrained_layout=True)
    had_any = False
    for idx, mode in enumerate(["global", "high_value"]):
        ax = axes[idx]
        for m, color in [(200, "#C40000"), (1000, "#A51616"), (5000, "#5A0E0E")]:
            p = inp / "34_claim_beta_sweep_bestparams" / f"{mode}_m{m}_sigma1_k512" / "summary" / "beta_sweep_metrics_long.csv"
            if mode == "high_value":
                p = inp / "34_claim_beta_sweep_bestparams" / f"{mode}_m{m}_sigma2_k256" / "summary" / "beta_sweep_metrics_long.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            par = d[d["model_key"] == "iqp_parity_mse"][["beta", "Q80"]].rename(columns={"Q80": "Q80_par"})
            cls = d[d["model_key"] != "iqp_parity_mse"].groupby("beta", as_index=False)["Q80"].min().rename(columns={"Q80": "Q80_best_cls"})
            z = pd.merge(par, cls, on="beta", how="inner")
            if z.empty:
                continue
            had_any = True
            ratio = z["Q80_par"].to_numpy(np.float64) / np.maximum(z["Q80_best_cls"].to_numpy(np.float64), 1e-12)
            ax.plot(z["beta"], ratio, marker="o", color=color, linewidth=1.8, label=f"m={m}")
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Q80 ratio (Parity / best classical)")
        ax.set_title(f"{mode} holdout")
        ax.grid(True, alpha=0.14, linestyle="--")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=7)
    if had_any:
        _save(fig, outdir / "fig4_fair_baselines_robustness_beta_m.pdf")
    else:
        plt.close(fig)
        # Fallback: build robustness panel from exp39 parity-vs-reference losses.
        p39 = inp / "39_claim_loss_ablation_nature" / "loss_ablation_metrics_long.csv"
        if p39.exists():
            d = pd.read_csv(p39)
            fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), constrained_layout=True)
            made = False
            for idx, mode in enumerate(["global", "high_value"]):
                ax = axes[idx]
                dm = d[d["holdout_mode"] == mode]
                if dm.empty:
                    ax.set_axis_off()
                    continue
                for m, color in [(200, "#C40000"), (1000, "#A51616"), (5000, "#5A0E0E")]:
                    dmm = dm[dm["m"] == m]
                    if dmm.empty:
                        continue
                    par = dmm[dmm["loss"] == "parity_mse"][["beta", "Q80"]].rename(columns={"Q80": "Q80_par"})
                    ref = (
                        dmm[dmm["loss"] != "parity_mse"]
                        .groupby("beta", as_index=False)["Q80"]
                        .min()
                        .rename(columns={"Q80": "Q80_best_ref"})
                    )
                    z = pd.merge(par, ref, on="beta", how="inner")
                    if z.empty:
                        continue
                    made = True
                    ratio = z["Q80_par"].to_numpy(np.float64) / np.maximum(z["Q80_best_ref"].to_numpy(np.float64), 1e-12)
                    ax.plot(z["beta"], ratio, marker="o", color=color, linewidth=1.8, label=f"m={m}")
                ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel("Q80 ratio (Parity / best reference)")
                ax.set_title(f"{mode} holdout")
                ax.grid(True, alpha=0.14, linestyle="--")
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(frameon=False, fontsize=7)
            if made:
                _save(fig, outdir / "fig4_fair_baselines_robustness_beta_m.pdf")
            else:
                plt.close(fig)

    # Extended data index
    ext = outdir / "extended_data_index.md"
    lines = [
        "# Extended Data Index",
        "",
        "- Full beta/seed panels remain in `outputs/paper_even_final/34_claim_beta_sweep_bestparams/`.",
        "- Loss ablation long metrics in `outputs/paper_even_final/39_claim_loss_ablation_nature/`.",
        "- Visibility intervention metrics in `outputs/paper_even_final/40_claim_visibility_causal_global/` and `outputs/paper_even_final/40_claim_visibility_causal_high_value/`.",
        "- Statistical tables in `outputs/paper_even_final/99_stats_tables/`.",
    ]
    ext.write_text("\n".join(lines), encoding="utf-8")

    print(f"[saved] {outdir}")


if __name__ == "__main__":
    main()
