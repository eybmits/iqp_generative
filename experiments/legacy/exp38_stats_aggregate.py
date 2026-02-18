#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate paper_even_final metrics into a unified long schema and statistics tables.

Outputs:
  outputs/paper_even_final/99_stats_tables/main_table.csv
  outputs/paper_even_final/99_stats_tables/supp_table.csv
  outputs/paper_even_final/99_stats_tables/significance_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


REQUIRED_COLS = [
    "seed",
    "holdout_mode",
    "beta",
    "m",
    "sigma",
    "K",
    "model",
    "loss",
    "Q80",
    "AUC_R_0_10000",
    "qH",
    "qH_ratio",
    "fit_tv",
]


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x) -> float:
    try:
        return int(float(x))
    except Exception:
        return float("nan")


def _infer_holdout_mode(path: Path, row: pd.Series) -> str:
    if "holdout_mode" in row and pd.notna(row["holdout_mode"]):
        return str(row["holdout_mode"])
    p = str(path).lower()
    if "high_value" in p or "high-value" in p:
        return "high_value"
    if "global" in p:
        return "global"
    return "unknown"


def _infer_model_loss(row: pd.Series) -> Tuple[str, str]:
    model = ""
    loss = ""

    if "model_key" in row and pd.notna(row["model_key"]):
        model = str(row["model_key"])
    elif "model_label" in row and pd.notna(row["model_label"]):
        model = str(row["model_label"])
    elif "model" in row and pd.notna(row["model"]):
        model = str(row["model"])

    if "loss" in row and pd.notna(row["loss"]):
        loss = str(row["loss"]).lower()

    m = model.lower()
    if not loss:
        if "parity" in m:
            loss = "parity_mse"
        elif "prob" in m and "mse" in m:
            loss = "prob_mse"
        elif "xent" in m or "mle" in m:
            loss = "xent"
        elif "mmd" in m:
            loss = "mmd"
        elif "classical" in m or "ising" in m or "transformer" in m or "maxent" in m:
            loss = "classical"
        else:
            loss = "unknown"

    if not model:
        if loss == "parity_mse":
            model = "iqp_parity"
        elif loss == "mmd":
            model = "iqp_mmd"
        elif loss == "xent":
            model = "iqp_xent"
        elif loss == "prob_mse":
            model = "iqp_prob_mse"
        else:
            model = "unknown"
    return model, loss


def _derive_auc(row: pd.Series) -> float:
    if "AUC_R_0_10000" in row and pd.notna(row["AUC_R_0_10000"]):
        return _safe_float(row["AUC_R_0_10000"])
    r1 = _safe_float(row.get("R_Q1000", np.nan))
    r2 = _safe_float(row.get("R_Q10000", np.nan))
    if not np.isfinite(r1) or not np.isfinite(r2):
        return float("nan")
    # piecewise linear approximation normalized by [0,10000]
    a = 0.5 * (0.0 + r1) * 1000.0
    b = 0.5 * (r1 + r2) * 9000.0
    return float((a + b) / 10000.0)


def _standardize_row(path: Path, row: pd.Series) -> Dict[str, float | str]:
    model, loss = _infer_model_loss(row)
    out: Dict[str, float | str] = {
        "seed": _safe_int(row.get("seed", np.nan)),
        "holdout_mode": _infer_holdout_mode(path, row),
        "beta": _safe_float(row.get("beta", np.nan)),
        "m": _safe_int(row.get("m", row.get("train_m", np.nan))),
        "sigma": _safe_float(row.get("sigma", np.nan)),
        "K": _safe_int(row.get("K", np.nan)),
        "model": model,
        "loss": loss,
        "Q80": _safe_float(row.get("Q80", np.nan)),
        "AUC_R_0_10000": _derive_auc(row),
        "qH": _safe_float(row.get("qH", np.nan)),
        "qH_ratio": _safe_float(row.get("qH_ratio", np.nan)),
        "fit_tv": _safe_float(row.get("fit_tv", row.get("fit_tv_to_pstar", np.nan))),
    }
    return out


def _collect_source_csvs(input_root: Path) -> List[Path]:
    files = []
    for p in input_root.rglob("*.csv"):
        name = p.name.lower()
        if "metrics" in name or "pair_rows" in name or "raw_rows" in name:
            files.append(p)
    return sorted(files)


def _to_standard_long(input_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for path in _collect_source_csvs(input_root):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        # prefer long-format rows that have direct per-model columns
        if "model_key" in df.columns or "model_label" in df.columns or "loss" in df.columns:
            for _, r in df.iterrows():
                rows.append(_standardize_row(path, r))
            continue

        # Fallback for wide pair rows: extract iqp/classical fields if available.
        wide_iqp_cols = {"Q80_iqp", "qH_iqp", "qH_ratio_iqp"}
        wide_cls_cols = {"Q80_class", "qH_class", "qH_ratio_class"}
        if wide_iqp_cols.issubset(set(df.columns)):
            for _, r in df.iterrows():
                r_iqp = r.copy()
                r_iqp["model"] = "iqp_parity"
                r_iqp["loss"] = "parity_mse"
                r_iqp["Q80"] = r.get("Q80_iqp", np.nan)
                r_iqp["qH"] = r.get("qH_iqp", np.nan)
                r_iqp["qH_ratio"] = r.get("qH_ratio_iqp", np.nan)
                if "fit_prob_mse_iqp" in df.columns:
                    r_iqp["fit_tv"] = r.get("fit_prob_mse_iqp", np.nan)
                rows.append(_standardize_row(path, r_iqp))
        if wide_cls_cols.issubset(set(df.columns)):
            for _, r in df.iterrows():
                r_cl = r.copy()
                r_cl["model"] = "classical_control"
                r_cl["loss"] = "classical"
                r_cl["Q80"] = r.get("Q80_class", np.nan)
                r_cl["qH"] = r.get("qH_class", np.nan)
                r_cl["qH_ratio"] = r.get("qH_ratio_class", np.nan)
                rows.append(_standardize_row(path, r_cl))

    out = pd.DataFrame(rows)
    if out.empty:
        for c in REQUIRED_COLS:
            out[c] = []
        return out

    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[REQUIRED_COLS]
    # keep rows with minimally required analysis fields
    out = out[np.isfinite(pd.to_numeric(out["Q80"], errors="coerce")) | np.isfinite(pd.to_numeric(out["qH_ratio"], errors="coerce"))]
    return out


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / (a.size * b.size))


def _paired_perm_pvalue(delta: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return float("nan")
    obs = abs(float(np.mean(delta)))
    count = 0
    for _ in range(int(n_perm)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=delta.size)
        stat = abs(float(np.mean(delta * signs)))
        if stat >= obs - 1e-15:
            count += 1
    return float((count + 1) / (n_perm + 1))


def _bootstrap_ci_mean(x: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, x.size, size=x.size)
        means[i] = float(np.mean(x[idx]))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _holm_correct(pvals: Iterable[float]) -> List[float]:
    p = np.array([float(v) for v in pvals], dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def _summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    groups = ["holdout_mode", "m", "beta", "model", "loss"]
    out = (
        df.groupby(groups, dropna=False)
        .agg(
            n=("Q80", "count"),
            Q80_mean=("Q80", "mean"),
            Q80_std=("Q80", "std"),
            Q80_median=("Q80", "median"),
            Q80_iqr=("Q80", lambda s: float(np.nanquantile(s, 0.75) - np.nanquantile(s, 0.25))),
            AUC_mean=("AUC_R_0_10000", "mean"),
            qH_mean=("qH", "mean"),
            qH_ratio_mean=("qH_ratio", "mean"),
            fit_tv_mean=("fit_tv", "mean"),
        )
        .reset_index()
    )
    return out


def _significance(df: pd.DataFrame, n_perm: int, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # parity as anchor
    parity = df[df["loss"] == "parity_mse"].copy()
    others = df[df["loss"] != "parity_mse"].copy()
    rows = []
    if parity.empty or others.empty:
        return pd.DataFrame(rows)

    regime_cols = ["holdout_mode", "m", "beta"]
    for regime, g_par in parity.groupby(regime_cols, dropna=False):
        g_oth = others
        for col, val in zip(regime_cols, regime):
            g_oth = g_oth[g_oth[col] == val]
        if g_oth.empty:
            continue
        for comp_loss, g_comp in g_oth.groupby("loss", dropna=False):
            merged = pd.merge(
                g_par[["seed", "Q80"]],
                g_comp[["seed", "Q80"]],
                on="seed",
                how="inner",
                suffixes=("_parity", "_comp"),
            )
            if merged.empty:
                continue
            q_par = merged["Q80_parity"].to_numpy(np.float64)
            q_cmp = merged["Q80_comp"].to_numpy(np.float64)
            delta = q_par - q_cmp
            ratio = q_par / np.maximum(q_cmp, 1e-12)
            p_raw = _paired_perm_pvalue(delta=delta, n_perm=n_perm, rng=rng)
            ci_lo, ci_hi = _bootstrap_ci_mean(delta, n_boot=n_boot, rng=rng)
            r_lo, r_hi = _bootstrap_ci_mean(ratio, n_boot=n_boot, rng=rng)
            rows.append(
                {
                    "holdout_mode": regime[0],
                    "m": regime[1],
                    "beta": regime[2],
                    "comparison": f"parity_mse_vs_{comp_loss}",
                    "n_pairs": int(merged.shape[0]),
                    "delta_Q80_mean": float(np.mean(delta)),
                    "delta_Q80_ci_lo": float(ci_lo),
                    "delta_Q80_ci_hi": float(ci_hi),
                    "Q80_ratio_mean": float(np.mean(ratio)),
                    "Q80_ratio_ci_lo": float(r_lo),
                    "Q80_ratio_ci_hi": float(r_hi),
                    "cliffs_delta_Q80": _cliffs_delta(q_par, q_cmp),
                    "p_value_raw": float(p_raw),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_value_holm"] = _holm_correct(out["p_value_raw"].tolist())
    return out


def _write_significance_report(path: Path, sig: pd.DataFrame, manifest: Path | None) -> None:
    lines = ["# Significance Report", ""]
    if manifest is not None:
        lines.append(f"- Protocol manifest: `{manifest}`")
    lines.append(f"- Number of significance rows: {len(sig)}")
    lines.append("")
    if sig.empty:
        lines.append("No comparable parity-vs-other paired rows were found.")
    else:
        for _, r in sig.sort_values(["p_value_holm", "p_value_raw"]).head(40).iterrows():
            lines.append(
                f"- [{r['holdout_mode']}] m={int(r['m']) if np.isfinite(r['m']) else 'nan'} beta={r['beta']:.2f} "
                f"{r['comparison']}: ratio={r['Q80_ratio_mean']:.3f} "
                f"(95% CI {r['Q80_ratio_ci_lo']:.3f},{r['Q80_ratio_ci_hi']:.3f}), "
                f"p={r['p_value_raw']:.4g}, Holm={r['p_value_holm']:.4g}, "
                f"Cliff={r['cliffs_delta_Q80']:.3f}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate stats into paper_even_final/99_stats_tables.")
    ap.add_argument("--input-root", type=str, default=str(ROOT / "outputs" / "paper_even_final"))
    ap.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "paper_even_final" / "99_stats_tables"))
    ap.add_argument("--manifest", type=str, default=str(ROOT / "docs" / "eval_manifest_v1.md"))
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    input_root = Path(args.input_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long_df = _to_standard_long(input_root)
    for c in REQUIRED_COLS:
        if c not in long_df.columns:
            long_df[c] = np.nan
    long_df = long_df[REQUIRED_COLS].copy()

    supp_table = long_df.copy()
    main_table = _summary_stats(long_df)
    sig = _significance(long_df, n_perm=int(args.n_perm), n_boot=int(args.n_boot), seed=int(args.seed))

    supp_path = outdir / "supp_table.csv"
    main_path = outdir / "main_table.csv"
    sig_path = outdir / "significance_tests.csv"
    report_path = outdir / "significance_report.md"
    unified_path = outdir / "metrics_long_unified.csv"

    long_df.to_csv(unified_path, index=False)
    supp_table.to_csv(supp_path, index=False)
    main_table.to_csv(main_path, index=False)
    sig.to_csv(sig_path, index=False)

    manifest = Path(args.manifest)
    _write_significance_report(report_path, sig=sig, manifest=manifest if manifest.exists() else None)

    print(f"[saved] {unified_path}")
    print(f"[saved] {main_path}")
    print(f"[saved] {supp_path}")
    print(f"[saved] {sig_path}")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()

