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
- Phase C: `865cf97`.

## Phase D — Validation upgrade ✅ (complete)

Reject inference (§5.2) → champion vs challenger (§5.4) → explainability (§5.8) → MDD ch 3, 11, 13.
Closes gaps #2 (reject inference), #3 (benchmark), #8 (explainability).

**Delivered:**
- **`data/reject_inference.py` (§5.2):** parceling, reweighting, fuzzy augmentation on the WoE
  design; `reject_inference_sensitivity.json` (coef + Gini shift vs KGB). Off by default;
  pipeline runs it only when `enabled` + reject data supplied (frozen WoE transform, no refit).
- **`evaluation/benchmark.py` (§5.4):** GBM/RF/xgboost challenger, **DeLong test** (fast
  Sun & Xu 2014) for correlated AUCs, OOT Gini±CI for both models, under-specified verdict
  (gap > 0.03 AND DeLong p < 0.05) surfaced in MDD + CLI. `benchmark.json`.
- **`evaluation/explainability.py` (§5.8):** exact native **linear SHAP** for the reportable
  model (β·(woe−E[woe]), no dependency at serve time) + `shap.TreeExplainer` for the challenger
  (impurity fallback if shap absent — risk R4). Interpretability parity (Jaccard top-K).
  `global_importance.json` + `shap_summary.png`. Added `shap==0.51` dependency.
- **Serving:** `POST /explain` returns points reasons + SHAP reasons + points/SHAP agreement;
  `woe_means` persisted so the endpoint needs no SHAP library.
- **MDD:** sections 3 (reject inference / KGB), 11 (champion-challenger + DeLong verdict),
  13 (explainability + parity).

**Tests:** `test_reject_inference.py` (parceling recovers the population intercept far closer
than KGB under outcome-correlated selection — the real RI effect, confirmed empirically),
`test_benchmark_delong.py` (DeLong detects a real difference, null z≈0; challenger beats a
linear model on an XOR interaction → under-specified verdict), `test_explainability_parity.py`
(exact linear SHAP, additivity, direction, agreement), plus `/explain` API parity tests.

**Finding worth recording:** KGB slope coefficients are *consistent* even under strong
selection — selection on the outcome shifts only the intercept/base rate. So reject inference's
robust, demonstrable win is base-rate/intercept recovery, matching the Banasik & Crook
literature the module cites. Tests assert that (not a slope-recovery claim that theory doesn't
support).

- Full suite green: **132 passed**; ruff + format + mypy clean.
- Coverage: **95% overall**; all domain modules ≥95% (reject_inference 99%, benchmark 99%,
  explainability 96%, discrimination 95%, calibration 97%, calibration_checks 100%).
- Runtime ~4 min (challenger fit + SHAP); trimmed challenger `n_estimators`/`shap_sample_size`
  in test configs to hold the line (risk R3).

### Commit
- Phase D: `<pending>`.

## Phase E — Fairness & Monitoring ✅ (complete)

Fairness (§5.6) → multi-period monitoring (§5.7) → MDD ch 12, 14.
Closes gaps #6 (single-snapshot monitoring) and #7 (no fairness testing).

**Delivered:**
- **`evaluation/fairness.py` (§5.6):** AIR (80% rule), SMD, SPD, EOD per protected attribute
  + mutual-information proxy scan. Numeric protected attrs dichotomised (age at 25), categorical
  by minority level. **Build-failing** on AIR < alert unless `acknowledge_failure`. `fairness.json`.
- **`monitoring/runlog.py` (§5.7):** SQLite/JSONL append-only run-log; PSI/CSI **trend** (OLS
  slope over periods flags rising drift before any single breach); **AvE backtest** per grade
  (Jeffreys traffic light); **score migration matrix** (off-diagonal mass).
- **`monitoring/monitor.py`:** `run_monitoring(..., period_id=)` appends to the run-log;
  new `run_monitoring_report` writes `monitoring_trend_report.json`.
- **CLI:** `monitor --period-id`; new `monitor-report` command.
- **Pipeline:** fairness computed on the full scored population (favourable = score ≥ median);
  `configs/german_credit.yaml` acknowledges the age-AIR breach (demo data, R5).
- **MDD:** section 12 (fairness + proxy scan + verdict); monitoring plan (14) covered by §10 +
  run-log location/cadence.

**Tests:** `test_fairness.py` (AIR flags planted disparate impact; proxy scan identifies the
planted proxy; build-fails when unacknowledged; masks) and `test_monitoring_runlog.py`
(sqlite+jsonl store; three-period rising trend flagged though each period is under alert; AvE
red on a miscalibrated grade; migration off-diagonal).

- Full suite green (145 passed); ruff + format + mypy clean. fairness 97%, monitor 100%,
  runlog 97%; TOTAL 94% (≥85% gate; monitoring/ is outside the ≥95% domain list).

### Commit
- Phase E: `cfba152`.

## Phase F — Governance & MDD finalisation ✅ (complete)

Full assumptions/limitations register wired → MDD restructured into a per-chapter package →
deterministic single-command regeneration → README updated.

**Delivered:**
- **`reporting/` package** replacing the single `reporting.py`: one module per MDD chapter
  under `mdd_sections/` (**16 chapters**, §6), an `MddContext` built purely from serialized
  artifacts, an orchestrator writing per-chapter `.md`/`.html` + `index.html` + a combined
  document.
- **Deterministic (DoD §9.10):** removed `datetime.now()` from the MDD; the provenance line
  uses the payload's frozen `created_at`/`version`. **Regeneration is byte-identical** (test).
- **Limitations & assumptions register (ch 15)** sourced from `model_card.json` (KGB entry
  auto-appears when reject inference is disabled).
- **`scorecard report`** CLI regenerates the MDD from artifacts with no retraining
  (`regenerate_mdd` via `load_context`).
- **README** updated: pipeline diagram, artifacts table, all CLI commands, `/explain`, the
  full config block table, and the new architecture tree.

**Tests:** `test_reporting_mdd.py` (16 chapters registered 1..16; all chapter files written;
byte-identical regeneration; KGB entry in ch15). `test_pipeline_e2e` still green (combined
document filename preserved).

- Full suite green: **149 passed**; ruff + format + mypy clean.
- Coverage: **TOTAL 95%**; features/ 95.8%, all evaluation modules ≥95%, DoD 97%, reject 99%
  (all §7 domain gates met). Reporting chapters are outside the ≥95% list (overall ≥85% applies).
- Runtime ~7 min (challenger+SHAP+bootstrap across several full pipeline runs) — flagged for
  Phase G trimming (risk R3).

### Commit
- Phase F: `<pending>`.

## Phase G — Green everything ⏳
