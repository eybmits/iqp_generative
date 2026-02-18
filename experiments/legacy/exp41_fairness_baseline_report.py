#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a fairness/protocol report for strong baseline comparisons.

Outputs:
  outputs/paper_even_final/41_claim_fairness_report/fairness_report.md
  outputs/paper_even_final/41_claim_fairness_report/model_budget_table.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _count_transformer_params(
    n: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_ff: int,
) -> int:
    try:
        from experiments.legacy.exp10_strong_classical_recovery import _ARTransformer  # type: ignore
        import torch  # type: ignore
    except Exception:
        return -1
    model = _ARTransformer(
        n=int(n),
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_ff=int(dim_ff),
        dropout=0.0,
    ).to(torch.device("cpu"))
    return int(sum(p.numel() for p in model.parameters()))


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fairness report for baseline protocol.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "41_claim_fairness_report"),
    )
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=400)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--artr-epochs", type=int, default=300)
    ap.add_argument("--artr-d-model", type=int, default=64)
    ap.add_argument("--artr-heads", type=int, default=4)
    ap.add_argument("--artr-layers", type=int, default=2)
    ap.add_argument("--artr-ff", type=int, default=128)
    ap.add_argument("--artr-lr", type=float, default=1e-3)
    ap.add_argument("--maxent-steps", type=int, default=2500)
    ap.add_argument("--maxent-lr", type=float, default=5e-2)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = int(args.n)
    pairs_nnn = hv.get_iqp_pairs_nn_nnn(n)
    n_pairs_nnn = int(len(pairs_nnn))
    n_pairs_dense = int(n * (n - 1) // 2)

    tr_params = _count_transformer_params(
        n=n,
        d_model=int(args.artr_d_model),
        nhead=int(args.artr_heads),
        num_layers=int(args.artr_layers),
        dim_ff=int(args.artr_ff),
    )

    rows = [
        {
            "model": "IQP Parity / IQP NLL / IQP MMD",
            "family": "iqp_qcbm",
            "parameter_count": int(n_pairs_nnn * int(args.layers)),
            "train_budget": f"steps={int(args.iqp_steps)}, lr={float(args.iqp_lr):g}",
            "data_access": "train samples only",
            "objective": "parity_mse | xent | mmd",
            "fairness_note": "same architecture, same optimization budget across losses",
        },
        {
            "model": "Ising+fields (NN+NNN)",
            "family": "classical_boltzmann",
            "parameter_count": int(n_pairs_nnn + n),
            "train_budget": f"steps={int(args.iqp_steps)}, lr={float(args.iqp_lr):g}",
            "data_access": "train samples only",
            "objective": "parity_mse",
            "fairness_note": "matched objective+steps to IQP parity branch",
        },
        {
            "model": "Dense Ising+fields",
            "family": "classical_boltzmann",
            "parameter_count": int(n_pairs_dense + n),
            "train_budget": f"steps={int(args.iqp_steps)}, lr={float(args.iqp_lr):g}",
            "data_access": "train samples only",
            "objective": "xent",
            "fairness_note": "strong dense classical control",
        },
        {
            "model": "AR Transformer (MLE)",
            "family": "autoregressive",
            "parameter_count": int(tr_params),
            "train_budget": (
                f"epochs={int(args.artr_epochs)}, lr={float(args.artr_lr):g}, "
                f"d_model={int(args.artr_d_model)}, heads={int(args.artr_heads)}, "
                f"layers={int(args.artr_layers)}, ff={int(args.artr_ff)}"
            ),
            "data_access": "train samples only",
            "objective": "MLE (xent)",
            "fairness_note": "strong likelihood baseline",
        },
        {
            "model": "MaxEnt parity (P,z)",
            "family": "maxent",
            "parameter_count": -1,
            "train_budget": f"steps={int(args.maxent_steps)}, lr={float(args.maxent_lr):g}",
            "data_access": "parity moments from train samples",
            "objective": "moment matching",
            "fairness_note": "apples-to-apples parity-information competitor",
        },
    ]
    _write_csv(outdir / "model_budget_table.csv", rows)

    md = []
    md.append("# Fairness Baseline Report")
    md.append("")
    md.append("## Protocol commitments")
    md.append("- Same dataset family (`paper_even`) and same holdout protocol per regime.")
    md.append("- Same seed list (`42..46`) for paired comparisons.")
    md.append("- Holdout remains untouched for hyperparameter tuning.")
    md.append("- Baseline selection based on pre-registered metric (`Q80`) and paired statistics.")
    md.append("")
    md.append("## Baseline set")
    for r in rows:
        md.append(
            f"- **{r['model']}**: params={r['parameter_count']}, budget={r['train_budget']}, "
            f"objective={r['objective']}. {r['fairness_note']}."
        )
    md.append("")
    md.append("## Inner tuning rule")
    md.append("- Tuning is allowed only on train/validation split within non-holdout training mass.")
    md.append("- Final test metrics are computed once on fixed holdout states.")
    md.append("")
    md.append("## Comparator policy")
    md.append("- Report best classical comparator per regime from pre-declared model set.")
    md.append("- No post-hoc model insertion after observing holdout metrics.")
    (outdir / "fairness_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[saved] {outdir / 'model_budget_table.csv'}")
    print(f"[saved] {outdir / 'fairness_report.md'}")


if __name__ == "__main__":
    main()
