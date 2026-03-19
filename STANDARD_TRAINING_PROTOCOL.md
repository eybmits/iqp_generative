# Standard Training Protocol

- protocol version: `analysis-standard-10seeds-all600-v3`
- active seed IDs: `111,112,113,114,115,116,117,118,119,120`
- seed count: `10`
- shared training budget: `600`

This is the active default protocol for new analysis experiments in this repository.

Final figure policy:

- all statistical summary experiments in the final package use `10` matched seeds by default
- fixed-case illustrative panels are the explicit exception
- in the current final set, the exceptions are:
  - `Experiment 1` panels `(a)` and `(b)` as fixed-`beta`, fixed-seed illustrations
  - `Experiment 4`, which is a fixed-case recovery illustration driven from a saved single-beta NPZ

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

The active 10-seed window was shifted to `111..120` so that the final
reporting package uses a single fixed matched interval and fully excludes the
known unstable draft-seed case at `110`.
