# Standard Training Protocol

- protocol version: `analysis-standard-10seeds-all600-v1`
- active seed IDs: `101,102,103,104,105,106,107,108,109,110`
- seed count: `10`
- shared training budget: `600`

This is the active default protocol for new analysis experiments in this repository.

Shared defaults:

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

The seed schedule is stored in `docs/benchmark_seed_schedule_10seeds.csv`.
Each training script that uses this protocol should also write a local `TRAINING_PROTOCOL.md`
into its output directory so the run artifact is self-describing.

Historical artifacts with `5`, `20`, or other seed counts are retained as legacy snapshots.
They do not override this active default.
