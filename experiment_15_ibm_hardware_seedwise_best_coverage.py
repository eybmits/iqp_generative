#!/usr/bin/env python3
"""Experiment 15: IBM hardware sampling for seedwise-best IQP parity vs IQP MSE coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from final_plot_style import apply_final_style, save_pdf
from model_labels import IQP_MSE_LABEL, IQP_PARITY_LABEL
from training_protocol import write_training_protocol

from experiment_3_beta_quality_coverage import (
    ELITE_FRAC,
    MODEL_STYLE,
    TRAIN_SAMPLE_OFFSET,
    _kl_pstar_to_q,
    build_target_distribution_paper,
    get_iqp_pairs_nn_nnn,
    quality_coverage_for_q,
    topk_mask_by_scores,
)

try:
    from qiskit import QuantumCircuit, qasm2
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    HAS_QISKIT = True
except Exception:  # pragma: no cover - runtime environment dependent
    QuantumCircuit = None  # type: ignore[assignment]
    Statevector = None  # type: ignore[assignment]
    generate_preset_pass_manager = None  # type: ignore[assignment]
    QiskitRuntimeService = None  # type: ignore[assignment]
    SamplerV2 = None  # type: ignore[assignment]
    qasm2 = None  # type: ignore[assignment]
    HAS_QISKIT = False


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = Path(__file__).name
OUTPUT_STEM = "experiment_15_ibm_hardware_seedwise_best_coverage"
DEFAULT_OUTDIR = ROOT / "plots" / "experiment_15_ibm_hardware_seedwise_best_coverage"
DEFAULT_EXPERIMENT12_DIR = ROOT / "plots" / "experiment_12_global_best_iqp_vs_mse"
DEFAULT_BUDGETS = (1000, 2000, 5000)
DEFAULT_SHOTS = 10_000
MODEL_KEYS = ("parity_seedwise_best", "iqp_mse")
MODEL_LABELS = {
    "parity_seedwise_best": f"{IQP_PARITY_LABEL} (seedwise best)",
    "iqp_mse": IQP_MSE_LABEL,
}
MODEL_COLORS = {
    "parity_seedwise_best": str(MODEL_STYLE["iqp_parity_mse"]["color"]),
    "iqp_mse": "#4A90E2",
}
MODEL_MARKERS = {
    "parity_seedwise_best": "o",
    "iqp_mse": "s",
}
TWO_QUBIT_OPS = {
    "cx",
    "cz",
    "ecr",
    "rzz",
    "swap",
    "iswap",
    "cp",
    "crx",
    "cry",
    "crz",
    "csx",
    "rxx",
    "ryy",
}


def _try_rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ci95_halfwidth(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(arr.size))


def _reduce_seed_stats(values: np.ndarray) -> Dict[str, float]:
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
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "n_seeds": int(arr.size),
        "min": float(np.min(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": std,
        "ci95": float(_ci95_halfwidth(arr)),
        "q3": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def _backend_name(backend: object) -> str:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        try:
            return str(name_attr())
        except Exception:
            pass
    if name_attr is not None:
        return str(name_attr)
    return str(backend)


def _operation_counts(qc: QuantumCircuit) -> Tuple[int, int]:
    ops = Counter({str(k): int(v) for k, v in qc.count_ops().items()})
    total_ops = int(sum(ops.values()))
    twoq_ops = int(sum(v for k, v in ops.items() if str(k) in TWO_QUBIT_OPS))
    return total_ops, twoq_ops


def _extract_bitstrings(pub_result: object) -> Sequence[str]:
    data = getattr(pub_result, "data", None)
    if data is None:
        raise RuntimeError("Sampler result does not expose a data payload.")
    containers: List[object] = []
    for attr_name in ("meas", "c"):
        attr = getattr(data, attr_name, None)
        if attr is not None:
            containers.append(attr)
    if hasattr(data, "values"):
        try:
            containers.extend(list(data.values()))
        except Exception:
            pass
    for container in containers:
        if hasattr(container, "get_bitstrings"):
            return list(container.get_bitstrings())
        if hasattr(container, "get_counts"):
            counts = container.get_counts()
            bitstrings: List[str] = []
            for bits, count in counts.items():
                bitstrings.extend([str(bits)] * int(count))
            return bitstrings
    raise RuntimeError("Sampler result does not expose measurement bitstrings or counts.")


def _bitstrings_to_count_vector(bitstrings: Sequence[str], n: int) -> np.ndarray:
    counts = np.zeros(2**int(n), dtype=np.int64)
    for bits in bitstrings:
        counts[int(str(bits), 2)] += 1
    return counts


def _make_iqp_qiskit_circuit(
    *,
    n: int,
    layers: int,
    weights: np.ndarray,
    measure: bool,
) -> QuantumCircuit:
    qc = QuantumCircuit(int(n), int(n) if measure else 0)
    pairs = get_iqp_pairs_nn_nnn(int(n))
    idx = 0
    for q in range(int(n)):
        qc.h(q)
    for _ in range(int(layers)):
        for i, j in pairs:
            qc.rzz(float(weights[idx]), i, j)
            idx += 1
        for q in range(int(n)):
            qc.h(q)
    if measure:
        qc.measure(range(int(n)), range(int(n)))
    return qc


def _build_seedwise_best_parity(exp12_data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    parity_kl_grid = np.asarray(exp12_data["parity_kl_grid"], dtype=np.float64)
    parity_weights = np.asarray(exp12_data["parity_weights"], dtype=np.float64)
    sigma_values = np.asarray(exp12_data["sigma_values"], dtype=np.float64)
    k_values = np.asarray(exp12_data["k_values"], dtype=np.int64)
    n_seeds = int(parity_kl_grid.shape[0])
    flat_idx = np.argmin(parity_kl_grid.reshape(n_seeds, -1), axis=1)
    sigma_idx, k_idx = np.unravel_index(flat_idx, parity_kl_grid.shape[1:])
    return {
        "best_sigma": np.asarray(sigma_values[sigma_idx], dtype=np.float64),
        "best_k": np.asarray(k_values[k_idx], dtype=np.int64),
        "best_kl": np.asarray(parity_kl_grid[np.arange(n_seeds), sigma_idx, k_idx], dtype=np.float64),
        "best_weights": np.asarray(parity_weights[np.arange(n_seeds), sigma_idx, k_idx], dtype=np.float64),
    }


def _build_model_weights_by_key(exp12_data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    seedwise = _build_seedwise_best_parity(exp12_data)
    return {
        "parity_seedwise_best": np.asarray(seedwise["best_weights"], dtype=np.float64),
        "iqp_mse": np.asarray(exp12_data["mse_weights"], dtype=np.float64),
    }


def _load_experiment12_inputs(experiment12_dir: Path) -> Dict[str, object]:
    summary_path = experiment12_dir / "experiment_12_global_best_iqp_vs_mse_summary.json"
    data_path = experiment12_dir / "experiment_12_global_best_iqp_vs_mse_data.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Experiment 12 summary: {summary_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing Experiment 12 data: {data_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    exp12_data = np.load(data_path, allow_pickle=False)
    return {
        "summary": summary,
        "data": exp12_data,
        "summary_path": summary_path,
        "data_path": data_path,
    }


def _resolve_service(account_name: str | None):
    saved_accounts = sorted(QiskitRuntimeService.saved_accounts().keys())
    if account_name:
        return QiskitRuntimeService(name=account_name), saved_accounts
    return QiskitRuntimeService(), saved_accounts


def _choose_backend(service, *, backend_name: str, min_qubits: int):
    if backend_name != "auto":
        return service.backend(backend_name), {"selection_mode": "explicit", "candidates": []}

    backends = service.backends(
        min_num_qubits=int(min_qubits),
        simulator=False,
        operational=True,
    )
    candidates = []
    for backend in backends:
        pending = None
        try:
            pending = int(backend.status().pending_jobs)
        except Exception:
            pending = int(getattr(backend, "pending_jobs", 10**9))
        candidates.append(
            {
                "name": _backend_name(backend),
                "num_qubits": int(getattr(backend, "num_qubits", min_qubits)),
                "pending_jobs": int(pending),
            }
        )
    if not candidates:
        raise RuntimeError(f"No operational IBM backends found with at least {min_qubits} qubits.")
    candidates_sorted = sorted(candidates, key=lambda row: (row["pending_jobs"], row["name"]))
    chosen_name = str(candidates_sorted[0]["name"])
    return service.backend(chosen_name), {"selection_mode": "auto", "candidates": candidates_sorted}


def _render_summary_plot(
    *,
    out_pdf: Path,
    out_png: Path,
    budgets: Sequence[int],
    per_seed_rows: List[Dict[str, object]],
) -> None:
    apply_final_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    ax_cov, ax_seed = axes

    xs = np.arange(len(budgets), dtype=float)
    all_cov_values: List[float] = []
    for model_key in MODEL_KEYS:
        color = MODEL_COLORS[model_key]
        marker = MODEL_MARKERS[model_key]
        label = MODEL_LABELS[model_key]
        exact_means: List[float] = []
        exact_ci: List[float] = []
        hardware_means: List[float] = []
        hardware_ci: List[float] = []
        model_rows = [row for row in per_seed_rows if row["model_key"] == model_key]
        for budget in budgets:
            exact_vals = np.asarray([float(row[f"exact_coverage_Q{int(budget)}"]) for row in model_rows], dtype=np.float64)
            hardware_vals = np.asarray(
                [float(row[f"hardware_coverage_Q{int(budget)}"]) for row in model_rows],
                dtype=np.float64,
            )
            exact_means.append(float(np.mean(exact_vals)))
            exact_ci.append(float(_ci95_halfwidth(exact_vals)))
            hardware_means.append(float(np.mean(hardware_vals)))
            hardware_ci.append(float(_ci95_halfwidth(hardware_vals)))
        exact_arr = np.asarray(exact_means, dtype=np.float64)
        hardware_arr = np.asarray(hardware_means, dtype=np.float64)
        hardware_ci_arr = np.asarray(hardware_ci, dtype=np.float64)
        all_cov_values.extend(exact_arr.tolist())
        all_cov_values.extend((hardware_arr - hardware_ci_arr).tolist())
        all_cov_values.extend((hardware_arr + hardware_ci_arr).tolist())
        ax_cov.plot(xs, exact_means, ls="--", lw=1.8, color=color, alpha=0.65, marker=marker, label=f"{label} simulation")
        ax_cov.plot(xs, hardware_means, ls="-", lw=2.3, color=color, marker=marker, label=f"{label} hardware")
        ax_cov.fill_between(
            xs,
            hardware_arr - hardware_ci_arr,
            hardware_arr + hardware_ci_arr,
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )

    ax_cov.set_xticks(xs, [str(int(b)) for b in budgets])
    ax_cov.set_xlabel("Coverage budget Q")
    ax_cov.set_ylabel("Quality coverage on elite unseen states")
    ax_cov.set_title("Coverage curves")
    ax_cov.grid(axis="y", alpha=0.25)
    if all_cov_values:
        ymin = float(np.min(all_cov_values))
        ymax = float(np.max(all_cov_values))
        span = max(1e-6, ymax - ymin)
        pad = max(6e-4, 0.015 * span)
        ax_cov.set_ylim(ymin - pad, ymax + pad)
    ax_cov.legend(frameon=True, fontsize=8.6, ncol=1, loc="best")

    q_focus = int(max(budgets))
    parity_rows = {int(row["seed"]): row for row in per_seed_rows if row["model_key"] == "parity_seedwise_best"}
    mse_rows = {int(row["seed"]): row for row in per_seed_rows if row["model_key"] == "iqp_mse"}
    seed_values = sorted(set(parity_rows.keys()) & set(mse_rows.keys()))
    if not seed_values:
        seed_values = sorted({int(row["seed"]) for row in per_seed_rows})
    y = np.arange(len(seed_values), dtype=float)
    parity_cov = np.asarray([float(parity_rows[seed][f"hardware_coverage_Q{q_focus}"]) for seed in seed_values], dtype=np.float64)
    mse_cov = np.asarray([float(mse_rows[seed][f"hardware_coverage_Q{q_focus}"]) for seed in seed_values], dtype=np.float64)
    for yi, seed in enumerate(seed_values):
        ax_seed.plot([mse_cov[yi], parity_cov[yi]], [yi, yi], color="#CFCFCF", lw=1.2, zorder=1)
    ax_seed.scatter(mse_cov, y, s=54, color=MODEL_COLORS["iqp_mse"], marker=MODEL_MARKERS["iqp_mse"], label=MODEL_LABELS["iqp_mse"], zorder=3)
    ax_seed.scatter(
        parity_cov,
        y,
        s=54,
        color=MODEL_COLORS["parity_seedwise_best"],
        marker=MODEL_MARKERS["parity_seedwise_best"],
        label=MODEL_LABELS["parity_seedwise_best"],
        zorder=3,
    )
    ax_seed.set_yticks(y, [str(seed) for seed in seed_values])
    ax_seed.set_xlabel(f"Hardware quality coverage (Q={q_focus})")
    ax_seed.set_ylabel("Seed")
    ax_seed.set_title("Per-seed hardware coverage")
    ax_seed.grid(axis="x", alpha=0.25)
    ax_seed.legend(frameon=True, fontsize=8.6, loc="best")

    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    save_pdf(fig, out_pdf)


def _write_readme(
    *,
    path: Path,
    summary_payload: Dict[str, object],
    plot_pdf_rel: str,
    plot_png_rel: str,
) -> None:
    lines = [
        "# Experiment 13 Summary",
        "",
        "IBM hardware sampling for the Experiment 12 seedwise-best IQP parity setting versus IQP MSE.",
        "",
        "Key settings:",
        "",
        f"- backend: `{summary_payload['backend_name']}`",
        f"- account: `{summary_payload['account_name_used']}`",
        f"- beta: `{float(summary_payload['beta']):g}`",
        f"- n: `{int(summary_payload['n'])}`",
        f"- train sample size: `{int(summary_payload['train_m'])}`",
        f"- layers: `{int(summary_payload['layers'])}`",
        "- parity selection: `seedwise-best oracle over (sigma, K) per seed`",
        f"- shots per circuit: `{int(summary_payload['shots'])}`",
        f"- seeds: `{','.join(str(int(x)) for x in summary_payload['seeds'])}`",
        "",
        "Coverage definition:",
        "",
        "- high-value states: top-10% of valid states by score",
        "- unseen subset: elite states not present in the matched `D_train` for that seed",
        "- metric: `C_q(Q) = Q^{-1} sum_x (1 - (1-q(x))^Q)` over elite unseen states",
        "",
        "Artifacts:",
        "",
        f"- summary plot PDF: `{plot_pdf_rel}`",
        f"- summary plot PNG: `{plot_png_rel}`",
        f"- per-seed metrics CSV: `{summary_payload['per_seed_csv']}`",
        f"- model summary CSV: `{summary_payload['summary_csv']}`",
        f"- pairwise parity-vs-mse summary CSV: `{summary_payload['pairwise_summary_csv']}`",
        f"- job rows CSV: `{summary_payload['job_rows_csv']}`",
        f"- jobs JSON: `{summary_payload['jobs_json']}`",
        f"- data NPZ: `{summary_payload['data_npz']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 15: IBM hardware sampling for seedwise-best IQP coverage.")
    ap.add_argument("--experiment12-dir", type=str, default=str(DEFAULT_EXPERIMENT12_DIR))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--account-name", type=str, default="")
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--optimization-level", type=int, default=1)
    ap.add_argument("--max-execution-time", type=int, default=540)
    ap.add_argument("--budgets", type=str, default="1000,2000,5000")
    ap.add_argument("--elite-frac", type=float, default=ELITE_FRAC)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not HAS_QISKIT:
        raise RuntimeError("Qiskit and qiskit-ibm-runtime are required for Experiment 15.")

    experiment12_dir = Path(args.experiment12_dir).expanduser()
    if not experiment12_dir.is_absolute():
        experiment12_dir = ROOT / experiment12_dir
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    loaded = _load_experiment12_inputs(experiment12_dir)
    exp12_summary = dict(loaded["summary"])
    exp12_data = loaded["data"]

    budgets = tuple(int(x.strip()) for x in str(args.budgets).split(",") if x.strip())
    if not budgets:
        raise ValueError("At least one coverage budget must be provided.")

    n = int(exp12_summary["n"])
    layers = int(exp12_summary["layers"])
    beta = float(exp12_summary["beta"])
    train_m = int(exp12_summary["train_m"])
    seeds = np.asarray(exp12_summary["seeds"], dtype=np.int64)
    shots = int(args.shots)
    p_star = np.asarray(exp12_data["p_star"], dtype=np.float64)
    train_indices = np.asarray(exp12_data["train_indices"], dtype=np.int64)
    model_weights = _build_model_weights_by_key(exp12_data)
    seedwise = _build_seedwise_best_parity(exp12_data)
    seedwise_sigma = np.asarray(seedwise["best_sigma"], dtype=np.float64)
    seedwise_k = np.asarray(seedwise["best_k"], dtype=np.int64)
    seedwise_kl = np.asarray(seedwise["best_kl"], dtype=np.float64)

    p_star_check, support, scores = build_target_distribution_paper(n, beta)
    if not np.allclose(p_star_check, p_star, atol=1e-12):
        raise RuntimeError("Experiment 12 p_star does not match the reconstructed target distribution.")

    elite_mask = topk_mask_by_scores(scores, support, frac=float(args.elite_frac))
    elite_total_count = int(np.sum(elite_mask))

    per_seed_csv = outdir / f"{OUTPUT_STEM}_per_seed_metrics.csv"
    summary_csv = outdir / f"{OUTPUT_STEM}_summary.csv"
    pairwise_summary_csv = outdir / f"{OUTPUT_STEM}_pairwise_summary.csv"
    jobs_json = outdir / f"{OUTPUT_STEM}_jobs.json"
    job_rows_csv = outdir / f"{OUTPUT_STEM}_job_rows.csv"
    data_npz = outdir / f"{OUTPUT_STEM}_data.npz"
    summary_json = outdir / f"{OUTPUT_STEM}_summary.json"
    run_config_json = outdir / "RUN_CONFIG.json"
    circuits_json = outdir / f"{OUTPUT_STEM}_circuits.json"
    readme = outdir / "README.md"
    plot_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    plot_png = outdir / f"{OUTPUT_STEM}.png"

    estimated_executions = int(len(MODEL_KEYS) * int(seeds.size) * shots)
    estimated_quantum_seconds = float(2.0 + 0.00035 * estimated_executions)

    account_name = str(args.account_name).strip() or None
    saved_accounts = sorted(QiskitRuntimeService.saved_accounts().keys())
    account_name_used = str(account_name if account_name is not None else (saved_accounts[0] if saved_accounts else ""))

    if args.dry_run:
        dry_payload = {
            "saved_accounts": saved_accounts,
            "backend_request": str(args.backend),
            "n": n,
            "layers": layers,
            "beta": beta,
            "train_m": train_m,
            "seeds": seeds.tolist(),
            "selection_mode": "seedwise_best_oracle",
            "seedwise_sigma": seedwise_sigma.tolist(),
            "seedwise_k": seedwise_k.tolist(),
            "seedwise_best_kl": seedwise_kl.tolist(),
            "shots": shots,
            "budgets": [int(x) for x in budgets],
            "estimated_executions": estimated_executions,
            "estimated_quantum_seconds": estimated_quantum_seconds,
            "experiment12_dir": _try_rel(experiment12_dir),
            "data_path": _try_rel(Path(loaded["data_path"])),
            "summary_path": _try_rel(Path(loaded["summary_path"])),
        }
        _write_json(summary_json, dry_payload)
        print(json.dumps(dry_payload, indent=2))
        return

    service, saved_accounts = _resolve_service(account_name)
    backend, backend_selection = _choose_backend(service, backend_name=str(args.backend), min_qubits=n)
    backend_name = _backend_name(backend)
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=int(args.optimization_level),
    )
    sampler = SamplerV2(mode=backend)
    sampler.options.max_execution_time = int(args.max_execution_time)

    n_models = len(MODEL_KEYS)
    n_seeds = int(seeds.size)
    n_states = int(2**n)
    n_budgets = len(budgets)
    counts_cube = np.zeros((n_models, n_seeds, n_states), dtype=np.int64)
    qhat_cube = np.zeros((n_models, n_seeds, n_states), dtype=np.float64)
    qideal_cube = np.zeros((n_models, n_seeds, n_states), dtype=np.float64)
    elite_unseen_masks = np.zeros((n_seeds, n_states), dtype=bool)
    exact_coverage_cube = np.zeros((n_models, n_seeds, n_budgets), dtype=np.float64)
    hardware_coverage_cube = np.zeros((n_models, n_seeds, n_budgets), dtype=np.float64)
    exact_kl = np.zeros((n_models, n_seeds), dtype=np.float64)
    hardware_kl = np.zeros((n_models, n_seeds), dtype=np.float64)

    per_seed_rows: List[Dict[str, object]] = []
    job_rows: List[Dict[str, object]] = []
    circuits_rows: List[Dict[str, object]] = []
    jobs_payload: Dict[str, object] = {
        "backend_name": backend_name,
        "saved_accounts": saved_accounts,
        "jobs": [],
    }
    completed_pairs: set[Tuple[int, str]] = set()

    if data_npz.exists():
        try:
            existing = np.load(data_npz, allow_pickle=False)
            if existing["counts_cube"].shape == counts_cube.shape:
                counts_cube[:] = np.asarray(existing["counts_cube"], dtype=np.int64)
                qhat_cube[:] = np.asarray(existing["qhat_cube"], dtype=np.float64)
                qideal_cube[:] = np.asarray(existing["qideal_cube"], dtype=np.float64)
                exact_coverage_cube[:] = np.asarray(existing["exact_coverage_cube"], dtype=np.float64)
                hardware_coverage_cube[:] = np.asarray(existing["hardware_coverage_cube"], dtype=np.float64)
                exact_kl[:] = np.asarray(existing["exact_kl"], dtype=np.float64)
                hardware_kl[:] = np.asarray(existing["hardware_kl"], dtype=np.float64)
                elite_unseen_masks[:] = np.asarray(existing["elite_unseen_masks"], dtype=bool)
        except Exception:
            pass
    if per_seed_csv.exists():
        per_seed_rows = list(_read_csv_rows(per_seed_csv))
        completed_pairs = {(int(row["seed"]), str(row["model_key"])) for row in per_seed_rows}
    if job_rows_csv.exists():
        job_rows = list(_read_csv_rows(job_rows_csv))
    if jobs_json.exists():
        try:
            jobs_payload = json.loads(jobs_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    if circuits_json.exists():
        try:
            circuits_rows = list(json.loads(circuits_json.read_text(encoding="utf-8")).get("circuits", []))
        except Exception:
            pass

    for seed_idx, seed in enumerate(seeds.tolist()):
        idxs_train = np.asarray(train_indices[seed_idx], dtype=np.int64)
        seen_mask = np.zeros(n_states, dtype=bool)
        seen_mask[np.unique(idxs_train)] = True
        elite_unseen_mask = np.asarray(elite_mask & (~seen_mask), dtype=bool)
        elite_unseen_masks[seed_idx] = elite_unseen_mask
        elite_unseen_count = int(np.sum(elite_unseen_mask))
        train_unique_count = int(np.sum(seen_mask))

        print(f"[experiment13] seed={seed} ({seed_idx + 1}/{n_seeds})", flush=True)

        for model_idx, model_key in enumerate(MODEL_KEYS):
            if (int(seed), str(model_key)) in completed_pairs:
                print(f"[experiment13]   skip existing model={model_key} seed={seed}", flush=True)
                continue
            weights = np.asarray(model_weights[model_key][seed_idx], dtype=np.float64)
            qc_ideal = _make_iqp_qiskit_circuit(n=n, layers=layers, weights=weights, measure=False)
            qc_meas = _make_iqp_qiskit_circuit(n=n, layers=layers, weights=weights, measure=True)

            q_exact = np.abs(Statevector.from_instruction(qc_ideal).data) ** 2
            q_exact = np.asarray(q_exact, dtype=np.float64)
            q_exact = q_exact / max(1e-15, float(np.sum(q_exact)))
            qideal_cube[model_idx, seed_idx] = q_exact
            exact_kl[model_idx, seed_idx] = float(_kl_pstar_to_q(p_star, q_exact))
            for budget_idx, budget_q in enumerate(budgets):
                exact_coverage_cube[model_idx, seed_idx, budget_idx] = float(
                    quality_coverage_for_q(q_exact, elite_unseen_mask, int(budget_q))
                )

            isa_qc = pass_manager.run(qc_meas)
            total_ops, twoq_ops = _operation_counts(isa_qc)
            circuit_row = {
                "seed": int(seed),
                "model_key": str(model_key),
                "model_label": MODEL_LABELS[model_key],
                "backend_name": backend_name,
                "raw_depth": int(qc_meas.depth()),
                "transpiled_depth": int(isa_qc.depth()),
                "transpiled_size": int(isa_qc.size()),
                "transpiled_width": int(isa_qc.width()),
                "num_qubits": int(isa_qc.num_qubits),
                "total_ops": int(total_ops),
                "two_qubit_ops": int(twoq_ops),
            }
            if qasm2 is not None:
                try:
                    circuit_row["transpiled_qasm2"] = qasm2.dumps(isa_qc)
                except Exception:
                    circuit_row["transpiled_qasm2"] = ""
            circuits_rows.append(circuit_row)
            _write_json(circuits_json, {"circuits": circuits_rows})

            print(
                f"[experiment13]   model={model_key} seed={seed} depth={isa_qc.depth()} twoq={twoq_ops} shots={shots}",
                flush=True,
            )
            job = sampler.run([isa_qc], shots=shots)
            job_id = job.job_id()
            result = job.result()
            bitstrings = _extract_bitstrings(result[0])
            counts = _bitstrings_to_count_vector(bitstrings, n)
            q_hat = counts.astype(np.float64) / max(1, int(np.sum(counts)))
            counts_cube[model_idx, seed_idx] = counts
            qhat_cube[model_idx, seed_idx] = q_hat
            hardware_kl[model_idx, seed_idx] = float(_kl_pstar_to_q(p_star, q_hat))
            for budget_idx, budget_q in enumerate(budgets):
                hardware_coverage_cube[model_idx, seed_idx, budget_idx] = float(
                    quality_coverage_for_q(q_hat, elite_unseen_mask, int(budget_q))
                )

            job_metrics = {}
            try:
                job_metrics = job.metrics()
            except Exception:
                job_metrics = {}
            jobs_payload["jobs"].append(
                {
                    "seed": int(seed),
                    "model_key": str(model_key),
                    "job_id": str(job_id),
                    "metrics": job_metrics,
                }
            )
            job_row = {
                "seed": int(seed),
                "model_key": str(model_key),
                "model_label": MODEL_LABELS[model_key],
                "backend_name": backend_name,
                "job_id": str(job_id),
                "shots": int(shots),
                "transpiled_depth": int(isa_qc.depth()),
                "transpiled_size": int(isa_qc.size()),
                "two_qubit_ops": int(twoq_ops),
                "quantum_seconds": float(job_metrics.get("usage", {}).get("quantum_seconds", float("nan"))) if isinstance(job_metrics, dict) else float("nan"),
            }
            job_rows.append(job_row)

            row = {
                "seed": int(seed),
                "model_key": str(model_key),
                "model_label": MODEL_LABELS[model_key],
                "backend_name": backend_name,
                "account_name_used": account_name_used,
                "job_id": str(job_id),
                "beta": float(beta),
                "n": int(n),
                "train_m": int(train_m),
                "layers": int(layers),
                "sigma": float(seedwise_sigma[seed_idx]) if model_key == "parity_seedwise_best" else float("nan"),
                "K": int(seedwise_k[seed_idx]) if model_key == "parity_seedwise_best" else -1,
                "elite_frac": float(args.elite_frac),
                "elite_total_count": int(elite_total_count),
                "elite_unseen_count": int(elite_unseen_count),
                "train_unique_count": int(train_unique_count),
                "shots": int(shots),
                "exact_kl": float(exact_kl[model_idx, seed_idx]),
                "hardware_kl": float(hardware_kl[model_idx, seed_idx]),
                "kl_delta_hardware_minus_exact": float(hardware_kl[model_idx, seed_idx] - exact_kl[model_idx, seed_idx]),
                "transpiled_depth": int(isa_qc.depth()),
                "transpiled_size": int(isa_qc.size()),
                "two_qubit_ops": int(twoq_ops),
            }
            for budget_idx, budget_q in enumerate(budgets):
                exact_cov = float(exact_coverage_cube[model_idx, seed_idx, budget_idx])
                hardware_cov = float(hardware_coverage_cube[model_idx, seed_idx, budget_idx])
                row[f"exact_coverage_Q{int(budget_q)}"] = exact_cov
                row[f"hardware_coverage_Q{int(budget_q)}"] = hardware_cov
                row[f"coverage_delta_Q{int(budget_q)}"] = float(hardware_cov - exact_cov)
            per_seed_rows.append(row)
            completed_pairs.add((int(seed), str(model_key)))

            _write_csv(per_seed_csv, per_seed_rows)
            _write_csv(job_rows_csv, job_rows)
            _write_json(jobs_json, jobs_payload)
            np.savez_compressed(
                data_npz,
                seeds=seeds,
                budgets=np.asarray(budgets, dtype=np.int64),
                model_keys=np.asarray(MODEL_KEYS),
                model_labels=np.asarray([MODEL_LABELS[k] for k in MODEL_KEYS]),
                counts_cube=counts_cube,
                qhat_cube=qhat_cube,
                qideal_cube=qideal_cube,
                exact_coverage_cube=exact_coverage_cube,
                hardware_coverage_cube=hardware_coverage_cube,
                exact_kl=exact_kl,
                hardware_kl=hardware_kl,
                elite_mask=np.asarray(elite_mask, dtype=np.int8),
                elite_unseen_masks=np.asarray(elite_unseen_masks, dtype=np.int8),
                train_indices=train_indices,
                p_star=p_star,
                seedwise_sigma=seedwise_sigma,
                seedwise_k=seedwise_k,
                seedwise_best_kl=seedwise_kl,
            )

    summary_rows: List[Dict[str, object]] = []
    pairwise_rows: List[Dict[str, object]] = []
    parity_rows = [row for row in per_seed_rows if row["model_key"] == "parity_seedwise_best"]
    mse_rows = [row for row in per_seed_rows if row["model_key"] == "iqp_mse"]
    parity_by_seed = {int(row["seed"]): row for row in parity_rows}
    mse_by_seed = {int(row["seed"]): row for row in mse_rows}
    common_seeds = sorted(set(parity_by_seed.keys()) & set(mse_by_seed.keys()))

    for model_idx, model_key in enumerate(MODEL_KEYS):
        model_seed_rows = [row for row in per_seed_rows if row["model_key"] == model_key]
        summary_row = {
            "model_key": str(model_key),
            "model_label": MODEL_LABELS[model_key],
            "backend_name": backend_name,
            **{f"exact_kl_{k}": v for k, v in _reduce_seed_stats(exact_kl[model_idx]).items()},
            **{f"hardware_kl_{k}": v for k, v in _reduce_seed_stats(hardware_kl[model_idx]).items()},
        }
        for budget_idx, budget_q in enumerate(budgets):
            exact_stats = _reduce_seed_stats(exact_coverage_cube[model_idx, :, budget_idx])
            hardware_stats = _reduce_seed_stats(hardware_coverage_cube[model_idx, :, budget_idx])
            for key, value in exact_stats.items():
                summary_row[f"exact_coverage_Q{int(budget_q)}_{key}"] = value
            for key, value in hardware_stats.items():
                summary_row[f"hardware_coverage_Q{int(budget_q)}_{key}"] = value
        summary_rows.append(summary_row)

    for budget_q in budgets:
        parity_vals = np.asarray([float(parity_by_seed[int(seed)][f"hardware_coverage_Q{int(budget_q)}"]) for seed in common_seeds], dtype=np.float64)
        mse_vals = np.asarray([float(mse_by_seed[int(seed)][f"hardware_coverage_Q{int(budget_q)}"]) for seed in common_seeds], dtype=np.float64)
        deltas = parity_vals - mse_vals
        pairwise_rows.append(
            {
                "metric": f"hardware_coverage_Q{int(budget_q)}",
                "n_common_seeds": int(len(common_seeds)),
                "parity_mean": float(np.mean(parity_vals)),
                "mse_mean": float(np.mean(mse_vals)),
                "delta_mean": float(np.mean(deltas)),
                "delta_median": float(np.median(deltas)),
                "delta_ci95": float(_ci95_halfwidth(deltas)),
                "parity_seed_wins": int(np.sum(parity_vals > mse_vals)),
                "mse_seed_wins": int(np.sum(parity_vals < mse_vals)),
                "ties": int(np.sum(np.isclose(parity_vals, mse_vals))),
            }
        )
    parity_kl_vals = np.asarray([float(parity_by_seed[int(seed)]["hardware_kl"]) for seed in common_seeds], dtype=np.float64)
    mse_kl_vals = np.asarray([float(mse_by_seed[int(seed)]["hardware_kl"]) for seed in common_seeds], dtype=np.float64)
    kl_deltas = parity_kl_vals - mse_kl_vals
    pairwise_rows.append(
        {
            "metric": "hardware_kl",
            "n_common_seeds": int(len(common_seeds)),
            "parity_mean": float(np.mean(parity_kl_vals)),
            "mse_mean": float(np.mean(mse_kl_vals)),
            "delta_mean": float(np.mean(kl_deltas)),
            "delta_median": float(np.median(kl_deltas)),
            "delta_ci95": float(_ci95_halfwidth(kl_deltas)),
            "parity_seed_wins": int(np.sum(parity_kl_vals < mse_kl_vals)),
            "mse_seed_wins": int(np.sum(parity_kl_vals > mse_kl_vals)),
            "ties": int(np.sum(np.isclose(parity_kl_vals, mse_kl_vals))),
        }
    )

    _write_csv(summary_csv, summary_rows)
    _write_csv(pairwise_summary_csv, pairwise_rows)
    _render_summary_plot(out_pdf=plot_pdf, out_png=plot_png, budgets=budgets, per_seed_rows=per_seed_rows)

    total_quantum_seconds = float(
        np.nansum(
            [
                float(row.get("quantum_seconds", float("nan")))
                for row in job_rows
                if isinstance(row.get("quantum_seconds", float("nan")), (int, float))
            ]
        )
    )
    if not np.isfinite(total_quantum_seconds):
        total_quantum_seconds = float("nan")

    summary_payload = {
        "script": SCRIPT_REL,
        "experiment12_dir": _try_rel(experiment12_dir),
        "experiment12_summary": _try_rel(Path(loaded["summary_path"])),
        "experiment12_data": _try_rel(Path(loaded["data_path"])),
        "backend_name": backend_name,
        "backend_selection": backend_selection,
        "account_name_used": account_name_used,
        "saved_accounts": saved_accounts,
        "beta": float(beta),
        "n": int(n),
        "train_m": int(train_m),
        "layers": int(layers),
        "shots": int(shots),
        "budgets": [int(x) for x in budgets],
        "elite_frac": float(args.elite_frac),
        "elite_total_count": int(elite_total_count),
        "seeds": seeds.tolist(),
        "completed_pairs": sorted([[int(seed), str(model_key)] for seed, model_key in completed_pairs]),
        "n_completed_pairs": int(len(completed_pairs)),
        "n_common_seeds": int(len(common_seeds)),
        "selection_mode": "seedwise_best_oracle",
        "seedwise_sigma": seedwise_sigma.tolist(),
        "seedwise_k": seedwise_k.tolist(),
        "seedwise_best_kl": seedwise_kl.tolist(),
        "estimated_executions": int(estimated_executions),
        "estimated_quantum_seconds": float(estimated_quantum_seconds),
        "observed_quantum_seconds_sum": float(total_quantum_seconds),
        "per_seed_csv": _try_rel(per_seed_csv),
        "summary_csv": _try_rel(summary_csv),
        "pairwise_summary_csv": _try_rel(pairwise_summary_csv),
        "job_rows_csv": _try_rel(job_rows_csv),
        "jobs_json": _try_rel(jobs_json),
        "data_npz": _try_rel(data_npz),
        "circuits_json": _try_rel(circuits_json),
        "plot_pdf": _try_rel(plot_pdf),
        "plot_png": _try_rel(plot_png),
    }
    _write_json(summary_json, summary_payload)
    _write_json(
        run_config_json,
        {
            "script": SCRIPT_REL,
            "outdir": _try_rel(outdir),
            "experiment12_dir": _try_rel(experiment12_dir),
            "backend": str(args.backend),
            "account_name": str(account_name or ""),
            "shots": int(shots),
            "optimization_level": int(args.optimization_level),
            "max_execution_time": int(args.max_execution_time),
            "budgets": [int(x) for x in budgets],
            "elite_frac": float(args.elite_frac),
            "saved_accounts": saved_accounts,
            "summary_json": _try_rel(summary_json),
            "per_seed_csv": _try_rel(per_seed_csv),
            "summary_csv": _try_rel(summary_csv),
            "pairwise_summary_csv": _try_rel(pairwise_summary_csv),
            "job_rows_csv": _try_rel(job_rows_csv),
            "jobs_json": _try_rel(jobs_json),
            "data_npz": _try_rel(data_npz),
            "circuits_json": _try_rel(circuits_json),
            "plot_pdf": _try_rel(plot_pdf),
            "plot_png": _try_rel(plot_png),
            "command": (
                f"python {SCRIPT_REL} --experiment12-dir {str(_try_rel(experiment12_dir))} "
                f"--outdir {str(_try_rel(outdir))} --backend {str(args.backend)} "
                f"--shots {int(shots)} --optimization-level {int(args.optimization_level)} "
                f"--max-execution-time {int(args.max_execution_time)} "
                f"--budgets {','.join(str(int(x)) for x in budgets)} --elite-frac {float(args.elite_frac):g}"
            ),
        },
    )
    _write_readme(
        path=readme,
        summary_payload=summary_payload,
        plot_pdf_rel=_try_rel(plot_pdf),
        plot_png_rel=_try_rel(plot_png),
    )
    write_training_protocol(
        outdir,
        experiment_name="Experiment 15 IBM hardware seedwise-best IQP coverage",
        note=(
            "Load Experiment 12 seedwise-best parity weights and IQP-MSE weights, sample the final circuits on IBM hardware, "
            "and evaluate the same elite-unseen quality-coverage metric used in Experiment 3."
        ),
        source_relpath=SCRIPT_REL,
        metrics_note=(
            "Stores ideal statevector probabilities, hardware counts, hardware q-hat distributions, per-seed KLs, "
            "and quality-coverage metrics at the requested budgets."
        ),
    )
    print(f"[experiment13] wrote {plot_pdf}", flush=True)


if __name__ == "__main__":
    main()
