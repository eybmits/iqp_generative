# Training Protocol

- protocol version: `analysis-standard-10seeds-all600-v1`
- active seed IDs: `101,102,103,104,105,106,107,108,109,110`
- seed count: `10`
- shared training budget: `600`

This is the active default protocol for new analysis artifacts under `outputs/analysis/`.

Shared defaults:

- IQP parity: `iqp_steps=600`, `lr=0.05`
- Ising+fields (NN+NNN): `steps=600`, `lr=0.05`
- Dense Ising+fields: `steps=600`, `lr=0.05`
- AR Transformer: `epochs=600`, `lr=1e-3`, `batch_size=256`
- MaxEnt parity: `steps=600`, `lr=0.05`
- restart policy: `single run, no restarts`

Each newly generated experiment directory should include its own local `TRAINING_PROTOCOL.md`.
Older subdirectories with different seed counts or budgets remain legacy snapshots.
