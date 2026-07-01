# Refactor Progress Log

Per §8, one entry per phase. Each phase must end with tests green before the next begins.

---

## Phase A — Diagnostic ✅ (complete)

- **Deliverable:** [`reports/refactor/00_diagnostic.md`](00_diagnostic.md) — codebase, config,
  artifact and test inventories; 10-gap→module mapping; refactor risk register (R1–R10).
- **Baseline established (regression anchor):** `uv run pytest -q` → **all tests pass, exit 0**
  (full suite on the offline synthetic adapter; session-scoped fixtures run the pipeline once).
- **Non-negotiables (§10) confirmed present** in current code and their guarding tests.
- **No source edited.** Only `reports/refactor/{00_diagnostic,progress}.md` written.
- **Key decisions taken in the diagnostic:**
  - `evaluation/metrics.py` → **replaced** by `evaluation/discrimination.py` (re-export shim
    during transition to avoid breaking `pipeline`/`cli`/tests — risk R2).
  - Sample design (§5.1) is **backward-compatible**: falls back to today's temporal/stratified
    split when DPD/origination columns are absent (risk R1).
  - Default challenger = `sklearn.GradientBoostingClassifier` (no new heavy dep); SHAP optional
    with `LinearExplainer`/coefficient fallback (risk R4).
  - Add `configs/german_credit.yaml` in Phase B to host fairness/reject examples; keep
    `home_credit.yaml` as the `run` default (risk R10).

### Commit
- _pending_ (repo is not currently a git repository; `git init` recommended before Phase B so
  each phase can be pinned to a commit hash per DoD §9.13).

---

## Phase B — Foundations rebuild ⏳ (not started)
Config schema (§4) → governance metadata (§5.9) → sample design + DoD (§5.1) → regression test.

## Phase C — Evaluation upgrade ⏳
## Phase D — Validation upgrade ⏳
## Phase E — Fairness & Monitoring ⏳
## Phase F — Governance & MDD finalisation ⏳
## Phase G — Green everything ⏳
