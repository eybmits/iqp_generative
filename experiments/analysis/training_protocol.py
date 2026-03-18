#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared active training protocol for analysis experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "analysis-standard-10seeds-all600-v1"
STANDARD_SEED_IDS = tuple(range(101, 111))
STANDARD_SEED_IDS_CSV = ",".join(str(x) for x in STANDARD_SEED_IDS)
STANDARD_SEED_COUNT = len(STANDARD_SEED_IDS)
STANDARD_TRAINING_BUDGET = 600
STANDARD_SEED_SCHEDULE_CSV = "docs/benchmark_seed_schedule_10seeds.csv"
CANONICAL_DOC_REL = "experiments/analysis/STANDARD_TRAINING_PROTOCOL.md"


def standard_seed_list() -> list[int]:
    return [int(x) for x in STANDARD_SEED_IDS]


def is_standard_seed_run(seed_values: Sequence[int]) -> bool:
    return [int(x) for x in seed_values] == standard_seed_list()


def protocol_markdown(
    *,
    experiment_name: str,
    note: str = "",
    source_relpath: str = "",
    metrics_note: str = "",
) -> str:
    lines = [
        "# Training Protocol",
        "",
        f"- protocol version: `{PROTOCOL_VERSION}`",
        f"- experiment: `{experiment_name}`",
        f"- active seed IDs: `{STANDARD_SEED_IDS_CSV}`",
        f"- seed count: `{STANDARD_SEED_COUNT}`",
        f"- shared training budget: `600`",
        "",
        "Active defaults used across the analysis experiment drivers:",
        "",
        "- IQP parity: `iqp_steps=600`, `lr=0.05`",
        "- Ising+fields (NN+NNN): `steps=600`, `lr=0.05`",
        "- Dense Ising+fields: `steps=600`, `lr=0.05`",
        "- AR Transformer: `epochs=600`, `lr=1e-3`, `batch_size=256`",
        "- MaxEnt parity: `steps=600`, `lr=0.05`",
        "- restart policy: `single run, no restarts`",
        "",
        "Statistics convention:",
        "",
        "- use `SD` to report between-seed / between-instance spread",
        "- use `95% CI` when the uncertainty of the mean itself is the quantity of interest",
        "- do not use `SE` as the default error bar in summary figures or tables",
        "",
        "Shared randomness order per matched instance:",
        "",
        "1. sample `D_train`",
        "2. sample the parity band `Omega` when the model uses parity features",
        "3. initialize each model with its model-specific initialization seed",
        "",
        "This file records the active analysis standard going forward. Historical artifacts with",
        "different seed counts or lighter budgets remain legacy snapshots and should not be",
        "interpreted as the current default protocol.",
    ]
    if metrics_note.strip():
        lines.extend(["", f"- experiment-specific metric note: {metrics_note.strip()}"])
    if note.strip():
        lines.extend(["", f"- note: {note.strip()}"])
    if source_relpath.strip():
        lines.extend(["", f"- source driver: `{source_relpath.strip()}`"])
    lines.extend(
        [
            "",
            f"- canonical protocol doc: `{CANONICAL_DOC_REL}`",
            f"- canonical seed schedule: `{STANDARD_SEED_SCHEDULE_CSV}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_training_protocol(
    outdir: Path,
    *,
    experiment_name: str,
    note: str = "",
    source_relpath: str = "",
    metrics_note: str = "",
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "TRAINING_PROTOCOL.md"
    path.write_text(
        protocol_markdown(
            experiment_name=experiment_name,
            note=note,
            source_relpath=source_relpath,
            metrics_note=metrics_note,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_seed_schedule_csv(path: Path, *, extra_columns: Iterable[tuple[str, int]] | None = None) -> None:
    rows = ["seed_index,seed_id"]
    for idx, seed in enumerate(STANDARD_SEED_IDS, start=1):
        fields = [str(idx), str(int(seed))]
        if extra_columns is not None:
            for _name, offset in extra_columns:
                fields.append(str(int(seed) + int(offset)))
        rows.append(",".join(fields))
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
