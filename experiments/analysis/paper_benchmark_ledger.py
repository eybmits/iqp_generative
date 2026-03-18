#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central paper-side ledger for benchmark-standard 20-seed experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from reviewer_disclosure_contract import reviewer_disclosure_contract
except ImportError:
    from experiments.analysis.reviewer_disclosure_contract import reviewer_disclosure_contract

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "paper_benchmark_ledger.md"
HISTORY_PATH = ROOT / "docs" / "paper_benchmark_run_history.json"
BENCHMARK_SEEDS = list(range(101, 121))

EXPERIMENT_SPECS: List[Dict[str, object]] = [
    {
        "experiment_id": "fig2_fixed_beta_sigmak_kl_20seed",
        "paper_target": "Fig. 3 / Table III fixed-beta sigma-K KL study",
        "status_kind": "planned",
        "artifact_hint": "No dedicated 20-seed analysis driver exists yet.",
    },
    {
        "experiment_id": "fig3_fixed_beta_kl_bshs_20seed",
        "paper_target": "Fig. 4 fixed-beta KL-BSHS boxplot at beta = 0.9",
        "status_kind": "artifact_dir",
        "artifact_path": "outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600",
    },
    {
        "experiment_id": "fig6_base_multiseed_20seed",
        "paper_target": "Base beta sweep recovery grid (beta = 0.5..1.2)",
        "status_kind": "artifact_dir",
        "artifact_path": "outputs/analysis/fig6_multiseed_all600_seeds101_120",
    },
    {
        "experiment_id": "fig6_base_q80_summary_20seed",
        "paper_target": "Base beta-vs-Q80 summary",
        "status_kind": "run_config_substring",
        "run_config_path": "outputs/analysis/fig6_beta_q80_summary/RUN_CONFIG.json",
        "required_substrings": ["fig6_multiseed_all600_seeds101_120"],
    },
    {
        "experiment_id": "fig6_wide_multiseed_20seed",
        "paper_target": "Wide beta sweep recovery grid (beta = 0.1..2.0)",
        "status_kind": "artifact_dir",
        "artifact_path": "outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_120",
    },
    {
        "experiment_id": "fig6_wide_q80_summary_iqr_20seed",
        "paper_target": "Wide beta-vs-Q80 summary (median + IQR)",
        "status_kind": "run_config_substring",
        "run_config_path": "outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr/RUN_CONFIG.json",
        "required_substrings": ["fig6_multiseed_beta0p1_2p0_all600_seeds101_120"],
    },
    {
        "experiment_id": "fig6_wide_q80_summary_iqr_seed_traces_20seed",
        "paper_target": "Wide beta-vs-Q80 summary with seed traces",
        "status_kind": "run_config_substring",
        "run_config_path": "outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/RUN_CONFIG.json",
        "required_substrings": ["fig6_multiseed_beta0p1_2p0_all600_seeds101_120"],
    },
    {
        "experiment_id": "fig6_wide_q80_summary_mean_std_20seed",
        "paper_target": "Wide beta-vs-Q80 summary (mean +/- std)",
        "status_kind": "run_config_substring",
        "run_config_path": "outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/RUN_CONFIG.json",
        "required_substrings": ["fig6_multiseed_beta0p1_2p0_all600_seeds101_120"],
    },
]


def is_benchmark_20seed_run(seed_values: Sequence[int]) -> bool:
    return [int(x) for x in seed_values] == BENCHMARK_SEEDS


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _load_history() -> Dict[str, object]:
    if HISTORY_PATH.exists():
        return _load_json(HISTORY_PATH)
    return {
        "schema_version": 1,
        "generated_from": "experiments/analysis/paper_benchmark_ledger.py",
        "runs": [],
    }


def _extract_seed_values(run_config: Dict[str, object]) -> List[int]:
    raw = run_config.get("seed_values", run_config.get("seeds", []))
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return []


def _extract_betas(run_config: Dict[str, object]) -> List[float]:
    if isinstance(run_config.get("betas"), list):
        return [float(x) for x in run_config["betas"]]
    if "beta" in run_config:
        return [float(run_config["beta"])]
    return []


def _config_subset(run_config: Dict[str, object], keys: Sequence[str]) -> Dict[str, object]:
    return {str(k): run_config[k] for k in keys if k in run_config}


def _format_scalar(value: object) -> str:
    if isinstance(value, float):
        text = f"{value:.6f}"
        return text.rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(_format_scalar(v) for v in value)
    return str(value)


def _seed_label(seed_values: Sequence[int]) -> str:
    vals = [int(x) for x in seed_values]
    if not vals:
        return "n/a"
    if vals == BENCHMARK_SEEDS:
        return "101..120 (20 seeds)"
    return ", ".join(str(x) for x in vals)


def _latest_runs_by_experiment(history: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    latest: Dict[str, Dict[str, object]] = {}
    runs = history.get("runs", [])
    if not isinstance(runs, list):
        return latest
    for raw in runs:
        if not isinstance(raw, dict):
            continue
        experiment_id = str(raw.get("experiment_id", ""))
        recorded_at = str(raw.get("recorded_at_utc", ""))
        prev = latest.get(experiment_id)
        if prev is None or recorded_at > str(prev.get("recorded_at_utc", "")):
            latest[experiment_id] = raw
    return latest


def _experiment_status_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for spec in EXPERIMENT_SPECS:
        experiment_id = str(spec["experiment_id"])
        paper_target = str(spec["paper_target"])
        status_kind = str(spec["status_kind"])
        if status_kind == "planned":
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "paper_target": paper_target,
                    "status": "planned",
                    "artifact": str(spec["artifact_hint"]),
                }
            )
            continue

        if status_kind == "artifact_dir":
            artifact_path = ROOT / str(spec["artifact_path"])
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "paper_target": paper_target,
                    "status": "available" if artifact_path.exists() else "missing",
                    "artifact": _rel(artifact_path),
                }
            )
            continue

        if status_kind == "run_config_substring":
            run_config_path = ROOT / str(spec["run_config_path"])
            if not run_config_path.exists():
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "paper_target": paper_target,
                        "status": "missing",
                        "artifact": _rel(run_config_path),
                    }
                )
                continue
            payload_text = run_config_path.read_text(encoding="utf-8")
            required_substrings = [str(x) for x in spec.get("required_substrings", [])]
            matches = all(token in payload_text for token in required_substrings)
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "paper_target": paper_target,
                    "status": "available" if matches else "legacy/out-of-sync",
                    "artifact": _rel(run_config_path),
                }
            )
    return rows


def record_benchmark_run(
    *,
    experiment_id: str,
    title: str,
    run_config_path: Path,
    output_paths: Sequence[Path],
    metrics_paths: Sequence[Path] = (),
    notes: Sequence[str] = (),
) -> None:
    run_config = _load_json(run_config_path)
    seed_values = _extract_seed_values(run_config)
    if not is_benchmark_20seed_run(seed_values):
        refresh_paper_benchmark_ledger()
        return

    history = _load_history()
    runs = history.setdefault("runs", [])
    if not isinstance(runs, list):
        runs = []
        history["runs"] = runs

    entry = {
        "recorded_at_utc": _now_utc(),
        "experiment_id": str(experiment_id),
        "title": str(title),
        "run_config_path": _rel(run_config_path),
        "outdir": str(run_config.get("outdir", _rel(run_config_path.parent))),
        "seed_values": seed_values,
        "betas": _extract_betas(run_config),
        "config_subset": _config_subset(
            run_config,
            [
                "beta",
                "betas",
                "q_eval",
                "q80_thr",
                "band_stat",
                "n",
                "train_m",
                "sigma",
                "K",
                "holdout_seed",
                "holdout_mode",
                "holdout_k",
                "holdout_pool",
                "holdout_m_train",
                "iqp_steps",
                "iqp_lr",
                "artr_epochs",
                "artr_d_model",
                "artr_heads",
                "artr_layers",
                "artr_ff",
                "artr_lr",
                "artr_batch_size",
                "maxent_steps",
                "maxent_lr",
            ],
        ),
        "output_paths": [_rel(Path(p)) for p in output_paths],
        "metrics_paths": [_rel(Path(p)) for p in metrics_paths],
        "notes": [str(x) for x in notes],
    }
    runs.append(entry)
    history["last_updated_utc"] = _now_utc()
    _write_json(HISTORY_PATH, history)
    _write_markdown_ledger(history)


def refresh_paper_benchmark_ledger() -> None:
    history = _load_history()
    history["last_updated_utc"] = _now_utc()
    _write_json(HISTORY_PATH, history)
    _write_markdown_ledger(history)


def _write_markdown_ledger(history: Dict[str, object]) -> None:
    contract = reviewer_disclosure_contract()
    protocol = contract["canonical_matched_instance_protocol"]
    budgets = contract["training_budget_fairness"]
    hyper = contract["model_hyperparameters"]
    capacity = contract["capacity_fairness"]
    stats = contract["statistics_reporting_protocol"]
    robustness = contract["pre_specified_robustness_axes"]
    package = contract["reproducibility_package_contents"]
    status_rows = _experiment_status_rows()
    latest_runs = _latest_runs_by_experiment(history)
    history_runs = history.get("runs", [])
    if not isinstance(history_runs, list):
        history_runs = []

    lines: List[str] = [
        "# Paper Benchmark Ledger",
        "",
        f"_Auto-generated from `experiments/analysis/paper_benchmark_ledger.py`. Last updated: {history.get('last_updated_utc', 'n/a')}_",
        "",
        "This file is the central paper-side benchmark note for the current repository state.",
        "It combines the static disclosure needed by the manuscript with the live registry of benchmark-standard 20-seed experiment runs.",
        "",
        "## Draft Sync",
        "",
        "- Current repository benchmark standard: `20` matched seeds `101..120`.",
        "- Current matched-instance count for the wide beta sweep: `20 betas x 20 seeds = 400`.",
        "- Any paper draft that still says `10` seeds or `200` matched instances is out of sync with the current repo standard.",
        "- Frozen final artifacts under `outputs/final_plots/` remain historical snapshots and are not the benchmark-standard source of truth.",
        "",
        "## Current 20-Seed Experiment Status",
        "",
        "| Experiment | Paper Target | Status | Artifact |",
        "| --- | --- | --- | --- |",
    ]
    for row in status_rows:
        lines.append(
            f"| `{row['experiment_id']}` | {row['paper_target']} | {row['status']} | `{row['artifact']}` |"
        )

    lines.extend(
        [
            "",
            "## Static Benchmark Disclosure",
            "",
            "### Matched Instances",
            "",
            f"- Index: `{protocol['index']}`",
            f"- Beta values: `{_format_scalar(protocol['beta_values'])}`",
            f"- Seed IDs: `{_seed_label(protocol['seed_ids'])}`",
            f"- Total matched instances in the wide sweep: `{protocol['matched_instances_total']}`",
            f"- Shared train data within instance: `{protocol['shared_train_dataset_within_instance']}`",
            f"- Shared parity band within instance: `{protocol['shared_parity_band_within_instance']}`",
            f"- Definition text: {protocol['definition_text']}",
            "",
            "### Randomness Stack",
            "",
        ]
    )
    for step in contract["exact_randomness_stack"]["per_instance_order"]:
        lines.append(f"- {step}")
    lines.extend(
        [
            "",
            "Derived seed formulas used in the current analysis drivers:",
        ]
    )
    for key, value in contract["exact_randomness_stack"]["current_analysis_seed_formulas"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "### Training Budgets",
            "",
            "| Model | Optimizer | LR | Budget | Early stopping | Batch size | Max objective evaluations |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    budget_rows = [
        ("iqp_parity_mse", "IQP parity"),
        ("classical_nnn_fields_parity", "Ising+fields (NN+NNN)"),
        ("classical_dense_fields_xent", "Dense Ising+fields"),
        ("classical_transformer_mle", "AR Transformer"),
        ("classical_maxent_parity", "MaxEnt parity"),
    ]
    for key, label in budget_rows:
        item = budgets[key]
        budget_value = item.get("steps", item.get("epochs", "n/a"))
        lines.append(
            "| {label} | {optimizer} | {lr} | {budget} | {early} | {batch} | {evals} |".format(
                label=label,
                optimizer=item["optimizer"],
                lr=_format_scalar(item["learning_rate"]),
                budget=_format_scalar(budget_value),
                early=item["early_stopping"],
                batch=_format_scalar(item["batch_size"]),
                evals=_format_scalar(item["max_objective_evaluations"]),
            )
        )

    lines.extend(
        [
            "",
            "### Model Hyperparameters",
            "",
            "| Model | Architecture / features | Capacity | Key settings |",
            "| --- | --- | --- | --- |",
            (
                "| IQP parity / IQP-MSE / IQP-MLE | "
                f"{hyper['iqp_parity_mse']['parameterization']} | "
                f"{hyper['iqp_parity_mse']['trainable_parameters_n12_layers1']} parameters | "
                f"lr={hyper['iqp_parity_mse']['learning_rate']}, steps={hyper['iqp_parity_mse']['steps']}, "
                f"restart={hyper['iqp_parity_mse']['restart_policy']} |"
            ),
            (
                "| Ising+fields (NN+NNN) | "
                f"{hyper['classical_nnn_fields_parity']['feature_set']} | "
                f"{hyper['classical_nnn_fields_parity']['feature_count_n12']} features | "
                f"solver={hyper['classical_nnn_fields_parity']['solver']}, "
                f"search_budget={hyper['classical_nnn_fields_parity']['search_budget']}, "
                f"regularization={hyper['classical_nnn_fields_parity']['regularization']} |"
            ),
            (
                "| Dense Ising+fields | "
                f"{hyper['classical_dense_fields_xent']['feature_set']} | "
                f"{hyper['classical_dense_fields_xent']['feature_count_n12']} features | "
                f"solver={hyper['classical_dense_fields_xent']['solver']}, "
                f"search_budget={hyper['classical_dense_fields_xent']['search_budget']}, "
                f"regularization={hyper['classical_dense_fields_xent']['regularization']} |"
            ),
            (
                "| MaxEnt parity | "
                f"{hyper['classical_maxent_parity']['feature_set']} | "
                f"{hyper['classical_maxent_parity']['feature_count_default_K512']} parameters | "
                f"solver={hyper['classical_maxent_parity']['solver']}, "
                f"search_budget={hyper['classical_maxent_parity']['search_budget']} |"
            ),
            (
                "| AR Transformer | "
                "autoregressive MLE on big-endian bit ordering | "
                f"{capacity['classical_transformer_mle_parameters_n12_default']} parameters | "
                f"d_model={hyper['classical_transformer_mle']['d_model']}, "
                f"layers={hyper['classical_transformer_mle']['layers']}, "
                f"heads={hyper['classical_transformer_mle']['heads']}, "
                f"dropout={hyper['classical_transformer_mle']['dropout']}, "
                f"weight_decay={hyper['classical_transformer_mle']['weight_decay']}, "
                f"lr={hyper['classical_transformer_mle']['learning_rate']}, "
                f"epochs={hyper['classical_transformer_mle']['epochs']} |"
            ),
            "",
            "### Transformer Transparency",
            "",
            "- The AR Transformer is intentionally over-documented because it is the easiest baseline to challenge as under-tuned.",
            f"- Bit ordering: `{hyper['classical_transformer_mle']['bit_ordering']}`",
            f"- d_model: `{hyper['classical_transformer_mle']['d_model']}`",
            f"- layers: `{hyper['classical_transformer_mle']['layers']}`",
            f"- heads: `{hyper['classical_transformer_mle']['heads']}`",
            f"- dropout: `{hyper['classical_transformer_mle']['dropout']}`",
            f"- weight decay: `{hyper['classical_transformer_mle']['weight_decay']}`",
            f"- learning rate: `{hyper['classical_transformer_mle']['learning_rate']}`",
            f"- epochs: `{hyper['classical_transformer_mle']['epochs']}`",
            f"- batch size: `{hyper['classical_transformer_mle']['batch_size']}`",
            f"- early stopping: `{hyper['classical_transformer_mle']['early_stopping']}`",
            f"- final selected config: `{hyper['classical_transformer_mle']['final_selected_config']}`",
            "",
            "### Restarts, Statistics, and Package Contents",
            "",
        ]
    )
    for key, value in contract["restart_policy"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            f"- Paired tests: `{_format_scalar(stats['paired_tests'])}`",
            f"- Paired sample size: `{stats['paired_test_sample_size']}`",
            f"- Sweep summaries: `{_format_scalar(stats['sweep_summaries'])}`",
            f"- Capacity note: {capacity['note']}",
            f"- Robustness axes: `sample_complexity_m={_format_scalar(robustness['sample_complexity_m'])}`, "
            f"`elite_thresholds={_format_scalar(robustness['elite_thresholds'])}`, "
            f"`larger_n_pilots={_format_scalar(robustness['larger_n_pilots'])}`",
            f"- Package contents: `{_format_scalar(package)}`",
            "",
            "## Latest Registered Benchmark-Standard Runs",
            "",
        ]
    )

    if not latest_runs:
        lines.append("No benchmark-standard 20-seed runs have been registered yet by the auto-logger.")
    else:
        for experiment_id in sorted(latest_runs):
            entry = latest_runs[experiment_id]
            lines.extend(
                [
                    f"### `{experiment_id}`",
                    "",
                    f"- Title: {entry.get('title', '')}",
                    f"- Recorded at: `{entry.get('recorded_at_utc', 'n/a')}`",
                    f"- Run config: `{entry.get('run_config_path', 'n/a')}`",
                    f"- Outdir: `{entry.get('outdir', 'n/a')}`",
                    f"- Seeds: `{_seed_label(entry.get('seed_values', []))}`",
                    f"- Betas: `{_format_scalar(entry.get('betas', []))}`",
                ]
            )
            config_subset = entry.get("config_subset", {})
            if isinstance(config_subset, dict) and config_subset:
                lines.append("- Key config:")
                for key in sorted(config_subset):
                    lines.append(f"  - `{key}`: `{_format_scalar(config_subset[key])}`")
            output_paths = entry.get("output_paths", [])
            if isinstance(output_paths, list) and output_paths:
                lines.append("- Outputs:")
                for path in output_paths:
                    lines.append(f"  - `{path}`")
            metrics_paths = entry.get("metrics_paths", [])
            if isinstance(metrics_paths, list) and metrics_paths:
                lines.append("- Metrics:")
                for path in metrics_paths:
                    lines.append(f"  - `{path}`")
            notes = entry.get("notes", [])
            if isinstance(notes, list) and notes:
                lines.append("- Notes:")
                for note in notes:
                    lines.append(f"  - {note}")
            lines.append("")

    lines.extend(["## Run History", ""])
    if not history_runs:
        lines.append("No runs recorded yet.")
    else:
        for idx, raw in enumerate(reversed(history_runs), start=1):
            if not isinstance(raw, dict):
                continue
            lines.append(
                f"{idx}. `{raw.get('recorded_at_utc', 'n/a')}` | `{raw.get('experiment_id', 'n/a')}` | "
                f"`{raw.get('outdir', 'n/a')}` | seeds `{_seed_label(raw.get('seed_values', []))}`"
            )

    DOC_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    refresh_paper_benchmark_ledger()
