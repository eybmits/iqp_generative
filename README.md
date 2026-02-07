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

**Run Experiments**
1. Full validation core (`paper_even` target)
```bash
python -m iqp_generative.core --outdir outputs/exp00_full_validation_paper_even --target-family paper_even
```
2. Experiment 1: main plots
```bash
python experiments/exp01_main_plots.py
```
3. Experiment 2: budget-law scatter
```bash
python experiments/exp02_budget_law.py
```
4. Experiment 3: visibility min-vis
```bash
python experiments/exp03_visibility_minvis.py
```
5. Experiment 4: finite-shot noise scaling
```bash
python experiments/exp04_noise_scaling.py
```
6. Experiment 5: discovery axis sweep
```bash
python experiments/exp05_discovery_axis.py
```
7. Experiment 7: claim report
```bash
python experiments/exp07_claim_report.py --mode run+analyze
```

**Outputs**
- Defaults go to `outputs/exp00_full_validation*` and `outputs/exp01_*` ... `outputs/exp07_*`.
- You can override with `--outdir` on each script.

**Current Final Package**
- `outputs/paper_even_final/` contains the cleaned, paper-even-only result bundle.
