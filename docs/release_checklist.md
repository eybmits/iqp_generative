# Release Checklist

## 1) Environment and tests
- Install deps: `pip install -r requirements.txt`
- Run tests: `pytest -q`
- Expected status: all tests pass (current baseline: `16 passed, 1 skipped`)

## 2) Public entry points
- Verify claim runners exist and run from `experiments/claims/` (`claim01` ... `claim11`)
- Prefer claim runners over direct legacy script calls for reproducibility

## 3) Reproducibility artifacts
- Keep run docs alongside major figure outputs (example: `RUN_DOC_q1000_no_iqp_mse_pareto_front.md`)
- Ensure each key figure has corresponding CSV + summary JSON

## 4) Repository hygiene
- Remove local caches/artifacts before commit:
  - `__pycache__/`, `*.pyc`, `.DS_Store`, `tmp/`
- Confirm `.gitignore` covers local artifacts and outputs

## 5) Documentation consistency
- `README.md` quickstart and claim list are current
- `docs/reproducibility_runbook.md` reflects primary protocol and workflow
- `experiments/claims/README.md` and `experiments/legacy/README.md` match current claim count

## 6) Pre-publish review
- `git status` contains only intentional changes
- No accidental file rewrites in unrelated modules
- Optional: run one smoke claim (e.g., `make claim07`) after merge

## 7) Versioning
- Use descriptive commit messages by scope:
  - `docs: ...`
  - `exp46: ...`
  - `claims: ...`
- Tag release only after figures + summary tables are frozen
