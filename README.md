# iqp_generative

IQP generative experiments with shared core code and clean output/data separation.

**Structure**
- `iqp_generative/` shared core implementation
- `experiments/` runnable experiment scripts
- `outputs/` generated figures/results (gitignored)
- `data/` datasets and caches (gitignored)
- `docs/` notes and drafts

**Install**
1. `pip install -r requirements.txt`
2. `pip install -r requirements-emnist.txt` (only needed for EMNIST)

**Run Experiments**
1. Full validation core (paper target)
```bash
python -m iqp_generative.core --outdir outputs/exp00_full_validation_paper --target-family paper
```
2. Full validation core (IQP-hard target)
```bash
python -m iqp_generative.core --outdir outputs/exp00_full_validation_iqp --target-family iqp_hard --n 12 --target-iqp-seed 123
```
3. Experiment 1: main plots
```bash
python experiments/exp01_main_plots.py
```
4. Experiment 2: budget-law scatter
```bash
python experiments/exp02_budget_law.py
```
5. Experiment 3: visibility min-vis
```bash
python experiments/exp03_visibility_minvis.py
```
6. Experiment 4: finite-shot noise scaling
```bash
python experiments/exp04_noise_scaling.py
```
7. Experiment 5: discovery axis sweep
```bash
python experiments/exp05_discovery_axis.py
```
8. Experiment 6: EMNIST
```bash
python experiments/exp06_emnist.py --split byclass --include-test --even-parity-only --include-fields
```

**Outputs**
- Defaults go to `outputs/exp00_full_validation*` and `outputs/exp01_*` ... `outputs/exp06_*`.
- You can override with `--outdir` on each script.
