# Training Protocol

- protocol version: `analysis-standard-10seeds-all600-v3`
- experiment: `Experiment 16 seedwise-best hardware recovery curve`
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

- experiment-specific metric note: Stores mean/std recovery fraction R(Q) over the elite-unseen state set for target, uniform, spectral completion, and IQP simulation/hardware curves.

- note: Aggregate recovery curves over the completed Experiment 15 seedwise-best parity hardware runs, and report mean and standard deviation across seeds for simulation and hardware.

- source driver: `experiment_16_seedwise_best_hardware_recovery_curve.py`

- canonical protocol doc: `STANDARD_TRAINING_PROTOCOL.md`
- canonical seed schedule: `docs/benchmark_seed_schedule_10seeds.csv`

