# Training Protocol

- protocol version: `analysis-standard-10seeds-all600-v3`
- experiment: `Experiment 7 ablation: Transformer capacity at fixed beta`
- active seed IDs: `111,112,113,114,115,116,117,118,119,120`
- seed count: `10`
- shared training budget: `600`

Active defaults used across the analysis experiment drivers:

- IQP parity: `iqp_steps=600`, `lr=0.05`
- Ising+fields (NN+NNN): `steps=600`, `lr=0.05`
- Dense Ising+fields: `steps=600`, `lr=0.05`
- AR Transformer: `epochs=600`, `lr=1e-3`, `batch_size=256`
- MaxEnt parity: `steps=600`, `lr=0.05`
- restart policy: `single run, no restarts`

Statistics convention:

- use `SD` to report between-seed / between-instance spread
- use `95% CI` when the uncertainty of the mean itself is the quantity of interest
- do not use `SE` as the default error bar in summary figures or tables

Shared randomness order per matched instance:

1. sample `D_train`
2. sample the parity band `Omega` when the model uses parity features
3. initialize each model with its model-specific initialization seed

This file records the active analysis standard going forward. Historical artifacts with
different seed counts or lighter budgets remain legacy snapshots and should not be
interpreted as the current default protocol.

- experiment-specific metric note: The primary outputs are validation NLL and exact forward KL as a function of model capacity.

- note: This capacity ablation uses the shared 10-seed / 600-budget analysis standard.

- source driver: `experiment_7_ablation_transformer_capacity_fixed_beta.py`

- canonical protocol doc: `STANDARD_TRAINING_PROTOCOL.md`
- canonical seed schedule: `docs/benchmark_seed_schedule_10seeds.csv`

