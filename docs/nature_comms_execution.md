# Nature-Comms v1 Execution

## One-command run

- Full protocol:
  - `python /Users/superposition/Coding/iqp_generative/test.py --profile nature_comms_v1`
- Smoke protocol:
  - `python /Users/superposition/Coding/iqp_generative/test.py --profile nature_comms_v1 --smoke`

## Produced artifacts

- Loss ablation:
  - `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/39_claim_loss_ablation_nature/`
- Mechanistic visibility intervention:
  - `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/40_claim_visibility_causal_global/`
  - `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/40_claim_visibility_causal_high_value/`
- Fairness report:
  - `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/41_claim_fairness_report/`
- Fair classical baseline matrix (full profile):
  - `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/34_claim_beta_sweep_bestparams/`
- Statistical tables:
  - `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/99_stats_tables/`
- Main figure set:
  - `/Users/superposition/Coding/iqp_generative/outputs/paper_figures_nature_v1/`
  - hashes: `/Users/superposition/Coding/iqp_generative/outputs/paper_figures_nature_v1/main_figure_hashes.sha256`

## Built-in checks

The profile run fails if any expected artifact is missing.
In `--smoke`, the expensive full classical matrix is skipped by design.

## Test commands

- Unit tests:
  - `python -m unittest discover -s /Users/superposition/Coding/iqp_generative/tests -p 'test_*.py'`
- Smoke integration (manual):
  - `python /Users/superposition/Coding/iqp_generative/test.py --profile nature_comms_v1 --smoke`
