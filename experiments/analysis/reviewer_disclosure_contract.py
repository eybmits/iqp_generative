"""Canonical reviewer-facing benchmark disclosure shared by analysis scripts."""

from __future__ import annotations

from typing import Any, Dict, List

CANONICAL_BETA_VALUES: List[float] = [round(0.1 * i, 1) for i in range(1, 21)]
CANONICAL_MATCHED_INSTANCE_SEED_IDS: List[int] = list(range(101, 121))


def reviewer_disclosure_contract() -> Dict[str, Any]:
    """Return the canonical matched-instance disclosure plus current repo caveats."""
    return {
        "canonical_matched_instance_protocol": {
            "index": "(beta, s)",
            "beta_values": CANONICAL_BETA_VALUES,
            "seed_ids": CANONICAL_MATCHED_INSTANCE_SEED_IDS,
            "matched_instances_total": len(CANONICAL_BETA_VALUES) * len(CANONICAL_MATCHED_INSTANCE_SEED_IDS),
            "shared_train_dataset_within_instance": True,
            "shared_parity_band_within_instance": True,
            "definition_text": (
                "A matched instance is indexed by (beta, s), with beta in {0.1, 0.2, ..., 2.0} "
                "and s in {1, ..., 20}, yielding 400 matched instances in total."
            ),
        },
        "exact_randomness_stack": {
            "per_instance_order": [
                "sample D_train",
                "sample parity band Omega",
                "initialize each model once",
                "train without restarts unless explicitly disclosed",
            ],
            "current_analysis_seed_formulas": {
                "train_dataset_seed": "s + 7",
                "parity_band_seed": "s + 222",
                "fig3_holdout_seed": "s + 111",
                "fig6_holdout_seed": "holdout_seed + 111",
                "iqp_init_seed": "s + 10000 + 7*K",
                "classical_nnn_fields_parity_init_seed": "s + 30001",
                "classical_dense_fields_xent_init_seed": "s + 30004",
                "transformer_init_seed": "s + 35501",
                "transformer_dataloader_seed": "s + 35512",
                "maxent_init_seed": "s + 36001",
                "restart_policy": "no restarts in the current analysis drivers",
            },
        },
        "training_budget_fairness": {
            "comparison_basis": "fixed optimization budgets per model family; not wall-clock neutral",
            "iqp_parity_mse": {
                "optimizer": "PennyLane AdamOptimizer",
                "learning_rate": 0.05,
                "steps": 600,
                "early_stopping": "none",
                "batch_size": "full-distribution objective",
                "max_objective_evaluations": 600,
            },
            "classical_nnn_fields_parity": {
                "optimizer": "PennyLane AdamOptimizer",
                "learning_rate": 0.05,
                "steps": 600,
                "early_stopping": "none",
                "batch_size": "full-distribution objective",
                "max_objective_evaluations": 600,
            },
            "classical_dense_fields_xent": {
                "optimizer": "PennyLane AdamOptimizer",
                "learning_rate": 0.05,
                "steps": 600,
                "early_stopping": "none",
                "batch_size": "full-distribution objective",
                "max_objective_evaluations": 600,
            },
            "classical_transformer_mle": {
                "optimizer": "torch.optim.Adam",
                "learning_rate": 1e-3,
                "epochs": 600,
                "early_stopping": "none",
                "batch_size": 256,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "max_objective_evaluations": 600,
            },
            "classical_maxent_parity": {
                "optimizer": "torch.optim.Adam",
                "learning_rate": 0.05,
                "steps": 600,
                "early_stopping": "none",
                "batch_size": "full-distribution objective",
                "max_objective_evaluations": 600,
            },
        },
        "restart_policy": {
            "iqp_parity_mse": "single initialization, no restart sweep",
            "classical_nnn_fields_parity": "single initialization, no restart sweep",
            "classical_dense_fields_xent": "single initialization, no restart sweep",
            "classical_transformer_mle": "single initialization, no restart sweep",
            "classical_maxent_parity": "single initialization, no restart sweep",
            "selection_rule": "single trained run per matched instance",
        },
        "model_hyperparameters": {
            "iqp_parity_mse": {
                "parameterization": "1-layer IQP ZZ circuit on cyclic NN+NNN pairs",
                "trainable_parameters_n12_layers1": 24,
                "optimizer": "PennyLane AdamOptimizer",
                "learning_rate": 0.05,
                "steps": 600,
                "early_stopping": "none",
                "restart_policy": "none",
            },
            "classical_nnn_fields_parity": {
                "feature_set": "NN+NNN Ising pair products plus local fields",
                "feature_count_n12": 36,
                "regularization": "none",
                "solver": "first-order gradient descent via PennyLane AdamOptimizer",
                "search_budget": 600,
                "validation_criterion": "not used; fixed budget single run",
            },
            "classical_dense_fields_xent": {
                "feature_set": "dense Ising pair products plus local fields",
                "feature_count_n12": 78,
                "regularization": "none",
                "solver": "first-order gradient descent via PennyLane AdamOptimizer",
                "search_budget": 600,
                "validation_criterion": "not used; fixed budget single run",
            },
            "classical_maxent_parity": {
                "feature_set": "parity-band sufficient statistics",
                "feature_count_default_K512": 512,
                "regularization": "none",
                "solver": "first-order gradient descent via torch.optim.Adam",
                "search_budget": 600,
                "validation_criterion": "not used; fixed budget single run",
            },
            "classical_transformer_mle": {
                "bit_ordering": "most-significant to least-significant bit",
                "d_model": 64,
                "layers": 2,
                "heads": 4,
                "dropout": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
                "epochs": 600,
                "batch_size": 256,
                "early_stopping": "none",
                "final_selected_config": "fixed single config in script defaults",
            },
        },
        "capacity_fairness": {
            "iqp_parity_mse_parameters_n12_layers1": 24,
            "classical_nnn_fields_parity_parameters_n12": 36,
            "classical_dense_fields_xent_parameters_n12": 78,
            "classical_maxent_parity_parameters_default_K512": 512,
            "classical_transformer_mle_parameters_n12_default": 67969,
            "note": (
                "Capacities are disclosed rather than force-matched; all models share the same n, "
                "matched-instance data, and fixed training-budget protocol."
            ),
        },
        "metric_aggregation": {
            "protocol": "compute per matched instance first, then aggregate over all 400 matched instances",
            "no_sample_pooling_before_instance_metrics": True,
            "kl_wins_definition": (
                "KL wins counts matched (beta, seed) instances on which the model achieves the lowest "
                "forward KL among all compared models."
            ),
        },
        "statistics_reporting_protocol": {
            "sweep_summaries": ["mean ± 95% CI", "median + IQR"],
            "paired_tests": ["paired Wilcoxon signed-rank", "Sign test"],
            "paired_test_sample_size": 400,
            "report_effect_direction": True,
            "report_p_value": True,
        },
        "benchmark_constants": {
            "n": 12,
            "support": "even parity only",
            "support_size": 2048,
            "beta_values": CANONICAL_BETA_VALUES,
            "train_m_default": 200,
            "current_repo_good_frac_default": 0.05,
            "reference_band_sigma": 1.0,
            "reference_band_K": 512,
            "fixed_beta_ablation": 0.9,
            "sigma_ablation_values": [0.5, 1.0, 2.0, 3.0],
            "K_ablation_values": [128, 256, 512],
            "budget_values_Q": [1000, 2000, 5000],
        },
        "pre_specified_robustness_axes": {
            "sample_complexity_m": [50, 100, 200, 500, 1000],
            "elite_thresholds": ["top-5%", "top-10%", "top-20%"],
            "parity_band_robustness_over_beta": True,
            "larger_n_pilots": [14, 16, 18],
        },
        "reproducibility_package_contents": [
            "raw per-instance metrics",
            "seed lists",
            "scripts to regenerate Table II and all figures",
            "configs",
        ],
        "artifact_specific_seed_note": (
            "Frozen final figures and curated post-freeze analysis reruns preserve artifact-specific seed "
            "lists such as 42..46 and 101..112. These explicit artifact seeds must not be "
            "conflated with the canonical 20-seed matched-instance benchmark disclosure based on 101..120."
        ),
    }
