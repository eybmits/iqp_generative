# Training Protocol

- protocol version: `analysis-standard-10seeds-all600-v3`
- experiment: `Experiment 15 IBM hardware seedwise-best IQP coverage`
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

- experiment-specific metric note: Stores ideal statevector probabilities, hardware counts, hardware q-hat distributions, per-seed KLs, and quality-coverage metrics at the requested budgets.

- note: Load Experiment 12 seedwise-best parity weights and IQP-MSE weights, sample the final circuits on IBM hardware, and evaluate the same elite-unseen quality-coverage metric used in Experiment 3.

- source driver: `experiment_15_ibm_hardware_seedwise_best_coverage.py`

- canonical protocol doc: `STANDARD_TRAINING_PROTOCOL.md`
- canonical seed schedule: `docs/benchmark_seed_schedule_10seeds.csv`

