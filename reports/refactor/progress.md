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

## Phase B — Foundations rebuild ✅ (complete)

Config schema (§4) → governance metadata (§5.9) → sample design + DoD (§5.1) → regression test.

**Delivered:**
1. **Config schema (§4):** 9 new pydantic blocks in `config.py` (`sample_design`,
   `reject_inference`, `discrimination`, `benchmark`, `calibration_extended`, `fairness`,
   `explainability`, `monitoring_extended`, `governance`) + defaults in `base.yaml`, all
   fail-fast validated. Added optional column-name fields to `sample_design` to close the
   §4 underspecification (§5.1 needs to *name* the DPD/status/origination columns).
2. **Governance metadata (§5.9):** new `governance/metadata.py` — model card with git SHA,
   package/interpreter versions, deterministic dataset + artifact SHA-256 hashes, seed,
   timestamp; auto-detected assumptions/limitations (KGB when reject inference off; missing
   fairness when no protected attrs). Emits `artifacts/model_card.json`. Volatile provenance
   kept **out** of `model.json`'s version hash (risk R6) → model version stays deterministic.
3. **Sample design + DoD (§5.1):** new `data/definition_of_default.py` (panel cure/re-default
   resolution, flat DPD/status construction, cohort assignment, exclusions, seasoning,
   `sample_design.json`) and rebuilt `data/split.py` to run sample design then carve. **Backward
   compatible** (risk R1): on the flat synthetic/Home-Credit frames sample design is a no-op on
   the modelling matrix — splits byte-identical to baseline (verified by `test_regression_baseline`).
4. **Wiring + config:** governance + sample-design summaries added to the payload/MDD; added
   `configs/german_credit.yaml` (hosts fairness/reject examples, risk R10).

**Tests (all green):** `test_definition_of_default.py`, `test_sample_design.py`,
`test_governance_metadata.py`, `test_regression_baseline.py`, and expanded `test_config.py`.
- Full suite: **96 passed** (was 68); ruff + `ruff format` + mypy clean.
- Coverage: **95% overall**; `data/definition_of_default.py` 97%, `governance/metadata.py` 98%
  (meets the ≥95% domain-module gate for the new modules).

**Non-negotiables (§10) preserved:** no change to binning/WoE/selection/scorecard math/leakage
guards; splits unchanged on default configs; deterministic model version retained.

### Commit
- Baseline: `3db26a9`  ·  Phase B: `d670dba`.

## Phase C — Evaluation upgrade ✅ (complete)

Discrimination with uncertainty (§5.3) → extended calibration backtest (§5.5) → MDD ch 8, 10.
Closes gaps #4 (no uncertainty) and #10 (no optimism correction), part of #5 (calibration).

**Delivered:**
- **`evaluation/discrimination.py` (§5.3):** AUC/Gini/KS with seeded bootstrap CIs (BCa /
  percentile / basic), McClish partial AUC, Somers' D, decile lift & cumulative gains, and
  **.632+ optimism correction** (Efron & Gong 1983) refitting the reportable logit form on
  each bootstrap OOB sample. Emits `discrimination.json`. `metrics.py` is now a thin shim
  re-exporting `auc/gini/ks` (single source of truth) — no bare Gini without a CI beside it.
- **`evaluation/calibration.py` (§5.5):** Brier, ECE, Hosmer-Lemeshow, reliability curve, and
  **per-grade Jeffreys backtest** with binomial **traffic-light** (green/yellow/red) + grade
  HHI. Emits `calibration_backtest.json` + `reports/figures/reliability_curve.png`.
- **MDD:** new sections 8a (discrimination + CIs + optimism) and 8b (calibration traffic light).
- **Pipeline:** both computed on the fitted model; artifacts hashed into the model card;
  bootstrap seeded so `model.json` version stays deterministic (verified by regression test).

**Tests:** `test_discrimination_ci.py` (CI covers analytic AUC≈0.94 on two Gaussians; BCa/
percentile/basic; seeded determinism; optimism ≤ apparent) and `test_calibration_traffic_light.py`
(all-green on calibrated, red on 2× miscalibrated; Jeffreys/binomial bounds).
- Full suite: **107 passed**. ruff + format + mypy clean.
- Coverage: **95% overall**; `evaluation/discrimination.py` 95%, `evaluation/calibration.py` 97%
  (meets the ≥95% `evaluation/` gate).

**Note (risk R3):** suite runtime rose to ~3.5 min from bootstrap work; kept in check by small
iteration counts in the test configs. Will revisit if it grows further.

### Commit
- Phase C: `<pending>`.

## Phase D — Validation upgrade ⏳
## Phase E — Fairness & Monitoring ⏳
## Phase F — Governance & MDD finalisation ⏳
## Phase G — Green everything ⏳
