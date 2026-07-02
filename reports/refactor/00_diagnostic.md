# PHASE A — DIAGNOSTIC

> Refactor: *Elevating Credit PD Scorecard Repo to Comprehensive Production Standard.*
> This document is produced **before any source edit** (per §0 / §2). It inventories the
> current repo, judges every module/config/artifact/test, maps the 10 gaps in §1 to the new
> modules in §5, and opens a risk register for the refactor itself.

Status of baseline at diagnosis time: single-fit WoE logistic scorecard, German-Credit-shaped
synthetic adapter + Home Credit CSV adapter, Typer CLI (`run/evaluate/monitor/serve`), FastAPI
`/score`, deterministic `model.json` artifact + auto-generated MDD (10 sections). Test suite
~55 tests, session-scoped fixtures run the pipeline once on the synthetic adapter.

---

## 1. Codebase inventory

Judgment legend: **KEEP** (as-is, minor edits allowed) · **EXTEND** (public API stays, add
surface) · **REBUILD** (interface stays, internals materially change) · **DEMOLISH**
(remove/replace outright).

### `src/creditscorecard/` — top level

| Path | What it does | Judgment |
|---|---|---|
| `__init__.py` | Package version + docstring on orientation convention. | KEEP |
| `config.py` | Pydantic models for all config blocks; deep-merge base+named YAML; fail-fast. | **EXTEND** — add 8 new config blocks (§4): `sample_design`, `reject_inference`, `discrimination`, `benchmark`, `calibration_extended`, `fairness`, `explainability`, `monitoring_extended`, `governance`. No existing model removed. |
| `logging.py` | `configure_logging` + `get_logger`. | KEEP |
| `pipeline.py` | The end-to-end `run` orchestration; builds payload/tables; calls MDD. | **REBUILD** — must plug in sample-design, discrimination-with-CI, extended calibration backtest, benchmark, fairness, explainability, governance, and richer payload. Orchestration shape stays; body grows substantially and payload schema expands. |
| `reasons.py` | Points-shortfall adverse-action reason codes. | KEEP (EXTEND consumer) — reused as the "points-based" side of the SHAP-vs-points interpretability parity (§5.8). |
| `registry.py` | Serialize/load `model.json`; deterministic content hash; MLflow guard. | **EXTEND** — must also hash artifacts for governance (§5.9), persist the many new `artifacts/*.json`, and record dataset/git/env provenance. Core hash/version logic KEEP. |
| `reporting.py` | Single-file MDD generator (10 sections) from payload+tables+figures. | **REBUILD** — becomes `reporting/` package with one module per MDD chapter (§6, 16 chapters). Deterministic-regeneration requirement (DoD §9.10) tightened. |
| `scoring.py` | Artifact-only scorer (`ScoringModel`) used by pipeline + API. | **EXTEND** — add SHAP-based `/explain` support (§5.8) alongside existing points reasons. Core scoring KEEP (parity-critical). |
| `cli.py` | Typer CLI: `run/evaluate/monitor/serve`. | **EXTEND** — `monitor` gains `--period-id` and writes to run-log; add `monitor-report`; `serve` must expose `/explain`. Existing commands keep their contract. |

### `src/creditscorecard/data/`

| Path | What it does | Judgment |
|---|---|---|
| `adapters.py` | csv + synthetic (German-Credit-shaped) loaders; attaches synthetic dates; validates binary target. | **EXTEND** — synthetic generator must optionally emit DPD/status/origination columns so §5.1 default-definition + §5.2 reject-inference synthetic fixtures are exercisable. csv/synthetic dispatch KEEP. |
| `schema.py` | Pandera contract: binary non-null target, non-empty frame. | **EXTEND** — add optional columns for sample design (DPD, status, origination date) validated when present. |
| `split.py` | Temporal-or-stratified train/test/OOT split. | **REBUILD** — becomes the consumer of the new Sample Design module (§5.1): performance-window resolution, seasoning filter, exclusions, cohort assignment feed the split. `SplitData` dataclass shape KEEP. |
| `definition_of_default.py` | *(does not exist)* | **ADD** (§5.1) |
| `reject_inference.py` | *(does not exist)* | **ADD** (§5.2) |

### `src/creditscorecard/features/`

| Path | What it does | Judgment |
|---|---|---|
| `binning.py` | OptBinning wrapper → frozen serializable specs; pure-pandas transform; Missing/Other codes. | KEEP / EXTEND — spec explicitly says extend, do not rewrite. |
| `woe.py` | WoE=ln(%Good/%Bad), IV, zero-cell stabilisation; frozen lookup transform. | KEEP / EXTEND — transform logic correct; API may extend. |
| `selection.py` | IV filter, iterative VIF, forward selection; auditable trail. | KEEP — used verbatim; may add bootstrap-coefficient hook consumers. |

### `src/creditscorecard/model/`

| Path | What it does | Judgment |
|---|---|---|
| `train.py` | statsmodels Logit (reportable) + sklearn parity + sign enforcement. | **REBUILD** — add bootstrap coefficient CIs and pass a benchmark/optimism hook (§5.4, §5.3). Single-fit final model + sign check + parity KEEP (non-negotiable §10). |
| `calibrate.py` | Single additive intercept shift to anchor mean PD. | **REBUILD (light)** — anchor math KEEP; the *testing* of calibration moves to new `evaluation/calibration.py` (§5.5). Keep this module focused on setting the anchor. |
| `scorecard.py` | Factor/Offset/PDO points math + Master Scale. | KEEP — formulas correct (constraint 9). Master Scale may gain HHI/traffic-light metadata consumers. |

### `src/creditscorecard/evaluation/`

| Path | What it does | Judgment |
|---|---|---|
| `metrics.py` | AUC/Gini/KS point estimates, metrics table. | **DEMOLISH → replaced by** `discrimination.py` (§5.3). Thin point-estimate-only module violates "no Gini without a bootstrap CI" (§3 DEMOLISH rule). Keep the pure AUC/Gini/KS helpers by moving them into the new module. |
| `curves.py` | ROC/CAP/calibration/score-dist figures + Hosmer-Lemeshow. | **EXTEND** — add reliability curve, SHAP summary, vintage seasoning, migration-matrix figures; HL moves alongside Brier/ECE in `calibration.py`. |
| `stability.py` | Frozen-reference PSI/CSI + HHI. | KEEP / EXTEND — extend to CSI history, migration matrix, AvE (§5.7). Frozen-bin mechanism is non-negotiable (§10, §11). |
| `calibration_checks.py` | MAPE-by-grade, anchor gap, curve-shape band. | **EXTEND / absorb** — folded into the richer `evaluation/calibration.py` (§5.5) which adds Brier, ECE, Jeffreys per-grade backtest, traffic lights, HHI. Existing checks retained as a subset. |
| `discrimination.py` | *(does not exist)* | **ADD** (§5.3) |
| `calibration.py` | *(does not exist — distinct from model/calibrate.py)* | **ADD** (§5.5) |
| `benchmark.py` | *(does not exist)* | **ADD** (§5.4) |
| `fairness.py` | *(does not exist)* | **ADD** (§5.6) |
| `explainability.py` | *(does not exist)* | **ADD** (§5.8) |

### `src/creditscorecard/monitoring/`

| Path | What it does | Judgment |
|---|---|---|
| `monitor.py` | One-shot PSI/CSI/HHI snapshot → `monitoring_report.json`; escalation. | **REBUILD** — writes to a run-log store, adds AvE backtest, vintage analysis, migration matrix (§5.7). Single-snapshot output that does not compare against a stored historical log is on the §3 DEMOLISH list. |
| `runlog.py` | *(does not exist)* | **ADD** (§5.7) |

### `src/creditscorecard/governance/` — *(directory does not exist)*

| Path | Judgment |
|---|---|
| `metadata.py` | **ADD** (§5.9) — model card, provenance hashes, assumptions/limitations register. |

### `src/creditscorecard/reporting/` — *(currently a single `reporting.py`)*

| Path | Judgment |
|---|---|
| `reporting/mdd_sections/` | **ADD** — one module per MDD chapter (§6). Convert `reporting.py` → package. |

### `app/`

| Path | What it does | Judgment |
|---|---|---|
| `api.py` | FastAPI `/score`, `/batch-score`, `/model-info`, `/health`. | **EXTEND** — add `POST /explain` (§5.8). Existing endpoints KEEP. |

---

## 2. Config inventory (`configs/base.yaml`, `configs/*.yaml`)

Existing keys — all **KEEP** (no removals; the refactor adds dimensions, §10):

| Block | Keys | Judgment |
|---|---|---|
| top | `seed` | keep |
| `data` | `adapter, path, target, date_column` | keep (add DPD/status/origination column names used by `sample_design`) |
| `split` | `test_size, oot_size, stratify` | keep (now consumed downstream of sample design) |
| `binning` | `min_bin_pct, monotonic_trend, max_n_bins` | keep |
| `selection` | `iv_min, iv_suspicious, vif_threshold, forward_metric, cv_folds` | keep |
| `model` | `engine, enforce_sign_check, sign_overrides, parity_tol` | keep |
| `calibration` | `anchor_default_rate` | keep |
| `scaling` | `pdo, target_score, target_odds, rating_grades, round_points` | keep |
| `monitoring` | `psi_bins, psi_warn, psi_alert` | keep (superset lives in `monitoring_extended`) |
| `validation` | `gini_min, ks_min, hhi_max, mape_max, anchor_gap_max, curve_shape_n_se` | keep |
| `tracking` | `mlflow_enabled, mlflow_uri` | keep |
| `paths` | `artifacts_dir, reports_dir, data_dir` | keep |

New blocks to **ADD** (verbatim from §4), each validated by a new pydantic model with the same
fail-fast contract as existing blocks:

- `sample_design` (observation/performance windows, `default_definition`, `exclusions`, `cohort_key`, `minimum_seasoning_months`)
- `reject_inference` (`enabled`, `reject_data_path`, `method`, `bad_rate_multiplier`, `parcels`)
- `discrimination` (`bootstrap_iterations`, `bootstrap_method`, `confidence_level`, `compute_partial_auc`, `partial_auc_range`, `compute_somers_d`)
- `benchmark` (`enabled`, `challenger`, `challenger_params`, `delong_test`, `interpretability_parity`)
- `calibration_extended` (`hosmer_lemeshow_groups`, `compute_brier`, `compute_ece`, `reliability_curve_bins`, `per_grade_backtest`)
- `fairness` (`enabled`, `protected_attributes`, `favourable_outcome`, `metrics`, `air_threshold_warn`, `air_threshold_alert`, `proxy_scan`, and — added — `acknowledge_failure` per §5.6 build-failing rule)
- `explainability` (`shap_enabled`, `shap_sample_size`, `reason_codes_top_n`, `global_importance_method`, `interpretability_parity_top_k`)
- `monitoring_extended` (`runlog_backend`, `runlog_path`, `cadence`, `ave_backtest_enabled`, `vintage_analysis_enabled`, `psi_history_min_periods`)
- `governance` (`model_id, model_name, model_tier, model_purpose, owner, developer, validator, approval_date, next_review_date, intended_use, known_limitations, assumptions`) + config-level `benchmark` verdict threshold (`gini_gap`, default 0.03) referenced in §5.4.

**Config discrepancy flagged:** the current CLI/API default is `configs/home_credit.yaml`; the
project memory references a `configs/german_credit.yaml` that does not exist in the tree. The
synthetic adapter emits German-Credit-shaped columns (`checking_status`, `age_years`, …,
target `default`) while `home_credit.yaml` targets `TARGET`. For §5.6 fairness (which the spec
anchors on German Credit's `age`/`foreign_worker`) and for reproducible offline tests, a
`configs/german_credit.yaml` should be **added** so the fairness/reject-inference examples have
a home. Recommend adding it in Phase B rather than renaming `home_credit.yaml`.

---

## 3. Artifact inventory (`artifacts/`)

Current serialized outputs:

| Artifact | Produced by | Change under refactor |
|---|---|---|
| `artifacts/model.json` | `registry.save_artifacts` | **EXTEND scope** — gains governance provenance (git SHA, hashes, versions), sample-design summary, richer model block (bootstrap coef CIs). Format (single JSON) and deterministic-hash mechanism KEEP. |
| `artifacts/tables/iv.csv`, `selection_iv.csv`, `metrics.csv`, `master_scale.csv` | `pipeline._build_tables` | KEEP names/format; `metrics.csv` gains CI columns (from §5.3). |
| `artifacts/tables/woe_*.csv` (per selected feature) | `pipeline._woe_detail` | KEEP. |
| `reports/monitoring_report.json` | `monitor._save_report` | **REBUILD** — superseded by the run-log store (`monitoring/runlog.db` default) + `monitor-report` trend artifact. Single-snapshot JSON retained only as a per-run convenience. |

New artifacts to **ADD** (one per §5 module, each governance-stamped, each sourcing an MDD chapter):

- `artifacts/sample_design.json` (§5.1)
- `artifacts/reject_inference_sensitivity.json` (§5.2, when enabled)
- `artifacts/discrimination.json` (§5.3)
- `artifacts/benchmark.json` (§5.4)
- `artifacts/calibration_backtest.json` (§5.5) + `reports/figures/reliability_curve.png`
- `artifacts/fairness.json` (§5.6)
- `artifacts/global_importance.json` (§5.8) + `reports/figures/shap_summary.png`
- `artifacts/model_card.json` (§5.9)
- `monitoring/runlog.db` (or `.jsonl`) + trend-report artifact (§5.7)
- Vintage seasoning + score-migration figures (§5.7)

Figures currently produced: `roc_curve.png`, `cap_curve.png`, `calibration.png`,
`score_distribution.png` — KEEP; add reliability, SHAP summary, vintage, migration figures.

---

## 4. Test inventory (`tests/`)

| Test file | Tests | Judgment after refactor |
|---|---|---|
| `conftest.py` | session fixtures: `config`, `dataset`, `split`, `fitted`, `pipeline_payload` | **EXTEND** — `split` fixture now flows through sample design; add fixtures for reject/protected-attribute synthetic frames. |
| `test_binning.py` | monotonic WoE, missing/other codes, frozen transform, min-bin | KEEP. |
| `test_woe_iv.py` | orientation, exact WoE/IV, manual-formula match, sorted IV | KEEP. |
| `test_selection.py` | IV filter, iterative VIF, forward select | KEEP. |
| `test_scorecard_scaling.py` | Factor/Offset exact, points formula, score↔PD consistency, master-scale monotonic | KEEP (non-negotiable formulas). |
| `test_config.py` | merge, fail-fast on bad split/iv/psi order, csv-needs-path | **EXTEND** — add fail-fast cases for the 9 new config blocks. |
| `test_leakage.py` | binning/WoE frozen on test/OOT, transform-before-fit raises | KEEP (non-negotiable §10). |
| `test_stability_psi.py` | zero PSI identical, shift raises PSI, frozen edges, CSI, guard, roundtrip | KEEP / EXTEND for history & migration. |
| `test_monitoring.py` | status thresholds, stable/escalate | **REBUILD** — now targets the run-log store + trend (`test_monitoring_runlog.py`, §7). |
| `test_validation_metrics.py` | HHI, MAPE, anchor gap, curve shape, summary present | KEEP / EXTEND — traffic-light + Jeffreys per-grade move to `test_calibration_traffic_light.py`. |
| `test_pipeline_e2e.py` | artifacts present, payload contract, perf reasonable, reproducible, frozen ref persisted | **EXTEND** — payload contract grows; becomes/【feeds】`test_regression_baseline.py` (§7) for cross-run numeric stability. |
| `test_registry.py` | deterministic version, changes with content, missing raises, numpy roundtrip, mlflow guard | KEEP / EXTEND for governance hashes (`test_governance_metadata.py`). |
| `test_cli.py` | run/evaluate/monitor | **EXTEND** — add `monitor --period-id` and `monitor-report`. |
| `test_scoring_parity.py` | artifact==payload, API score==pipeline, health/info, batch, 422 | KEEP / EXTEND — add `/explain` parity. |

New test files to **ADD** (§7): `test_definition_of_default.py`, `test_sample_design.py`,
`test_reject_inference.py`, `test_discrimination_ci.py`, `test_benchmark_delong.py`,
`test_calibration_traffic_light.py`, `test_fairness.py`, `test_monitoring_runlog.py`,
`test_explainability_parity.py`, `test_governance_metadata.py`, `test_regression_baseline.py`.

Coverage target: ≥85% overall; ≥95% on `features/`, `model/`, `evaluation/`,
`data/definition_of_default.py`, `data/reject_inference.py`.

---

## 5. Gap-to-module mapping

| # | Gap (§1) | Closing module(s) | Spec |
|---|---|---|---|
| 1 | Sample design trivialised (no DoD / performance window / vintage / exclusions) | `data/definition_of_default.py`, rebuilt `data/split.py` | §5.1 |
| 2 | No reject inference (KGB-only bias) | `data/reject_inference.py` | §5.2 |
| 3 | No champion–challenger benchmark | `evaluation/benchmark.py` | §5.4 |
| 4 | Discrimination metrics have no uncertainty | `evaluation/discrimination.py` (replaces `metrics.py`) | §5.3 |
| 5 | Calibration under-tested (HL only) | `evaluation/calibration.py` (Brier/ECE/Jeffreys/traffic-light/HHI) | §5.5 |
| 6 | Monitoring is a single snapshot | `monitoring/runlog.py` + rebuilt `monitoring/monitor.py` | §5.7 |
| 7 | No fairness testing (AIR/SMD/SPD/EOD/proxy) | `evaluation/fairness.py` | §5.6 |
| 8 | Explainability is points-only (no SHAP, no parity) | `evaluation/explainability.py` + `scoring.py`/`api.py` | §5.8 |
| 9 | No governance metadata (tier/id/owner/limitations) | `governance/metadata.py` + `model_card.json` | §5.9 |
| 10 | No optimism correction / coefficient CIs | `.632+` in `discrimination.py`; bootstrap CIs in rebuilt `model/train.py` | §5.3, §5.4 |

All 10 gaps become first-class modules; the MDD (§6, 16 chapters) consumes their artifacts.

---

## 6. Risk register for the refactor itself

| # | Risk | Where it bites | Detection / mitigation |
|---|---|---|---|
| R1 | Rebuilding `split.py` around sample design changes split sizes → shifts every downstream number and breaks `test_pipeline_e2e` reproducibility. | pipeline, all perf tests | Keep sample-design **opt-in / backward-compatible** (when no DPD/origination columns, fall back to current temporal/stratified split). Regression test `test_regression_baseline.py` pins key numbers across two runs. |
| R2 | Replacing `metrics.py` with `discrimination.py` breaks imports in `pipeline.py`, `cli.py evaluate`, `test_validation_metrics`, `test_pipeline_e2e`. | many | Provide the same `compute_metrics`/`metrics_table` API from the new module (re-export shim) during transition; migrate call sites in one commit. |
| R3 | Bootstrap CIs / SHAP / GBM challenger add heavy compute → test suite blows past its ~60s budget. | CI, dev loop | Small `bootstrap_iterations` and `shap_sample_size` in test config; gate expensive paths behind config flags; seed all RNGs. |
| R4 | New deps (shap, xgboost?) not in `pyproject`/`uv.lock`; offline/Windows wheels may be unavailable. | build, Docker | Prefer `sklearn.ensemble.GradientBoostingClassifier` (already available) as default challenger; SHAP optional-dep + fallback to `LinearExplainer`/coefficient importance; document in limitations if unavailable. |
| R5 | Fairness build-failing assertion (§5.6) could make `run` fail on the default dataset. | CLI `run`, CI | `fairness.enabled` false unless `protected_attributes` set; `acknowledge_failure` escape hatch; German Credit config opts in explicitly. |
| R6 | Determinism regressions: bootstrap/GBM/SHAP introduce RNG. | DoD §9.10 byte-identical MDD | Every new RNG seeded from `config.seed`; MDD regeneration reads artifacts only (no refit); `test_regression_baseline` + `test_governance_metadata` hash-stability guard. |
| R7 | `reporting.py` → `reporting/` package conversion breaks the lazy import in `pipeline.py` and `test_pipeline_e2e`. | MDD generation | Keep `generate_mdd` entry point signature; add chapters incrementally; snapshot-test the index. |
| R8 | Reject inference & sample design need columns the synthetic/German data lack. | §5.1, §5.2 | Extend synthetic adapter to emit DPD/status/origination + a reject pool; where real reject data is absent, ship documented stub + limitations-register entry (§0, §5.2). |
| R9 | Coverage gate ≥95% on domain modules is demanding for stochastic code. | CI gate | Design modules with pure, deterministic cores (seeded) and thin I/O shells; test the cores directly. |
| R10 | Config default mismatch (`home_credit` vs missing `german_credit`) causes confusion for fairness examples. | config, docs | Add `configs/german_credit.yaml` in Phase B; keep `home_credit.yaml` default for `run`. |

---

## Baseline established

- Test suite: run recorded in Phase B progress log (regression anchor for R1/R2/R6).
- One command: `uv run scorecard run --config configs/home_credit.yaml` (synthetic adapter offline).
- Non-negotiables carried from §10 confirmed present in code: no-leakage (frozen binning/WoE,
  guarded by `test_leakage`), single-fit final model + sign check + sklearn parity
  (`model/train.py`), frozen PSI bins (`evaluation/stability.py`), WoE=ln(%Good/%Bad)
  (`features/woe.py`), IV thresholds + iterative VIF (`features/selection.py`), exact
  Factor/Offset/PDO (`model/scorecard.py`), artifacts-are-the-model (`registry.py`/`scoring.py`),
  reproducibility (`_set_seed`, deterministic hash). **No new module may weaken these (§10).**

---

## Next step

Phase A is complete. Phase B begins with the config schema (§4) + governance metadata (§5.9) +
sample design (§5.1), each landing with tests green before the next, per §8. No source has been
edited yet.

---

## Post-implementation reconciliation (Phases B–G)

The plan above was executed as written. Deviations worth recording so this document matches
the final code state (DoD §9.1):

- **`evaluation/metrics.py`** was **kept as a thin re-export shim** of `auc/gini/ks` from the
  new `discrimination.py` (not deleted) so `pipeline`/`cli`/tests keep a single import surface
  (risk R2). `discrimination.py` is the CI-bearing reportable source.
- **`reporting.py` → `reporting/` package** with `mdd_sections/` (16 chapter modules) + a
  deterministic orchestrator; `datetime.now()` removed from the MDD so regeneration is
  byte-identical (DoD §9.10). Added `scorecard report` to regenerate from artifacts.
- **`sample_design`** config gained optional column-name fields (`dpd_column`, `status_column`,
  `origination_date_column`, …) — §4's skeleton did not name the DPD/status columns that §5.1
  requires; seasoning is keyed off the *explicit* origination column so it never perturbs the
  legacy split on flat data (risk R1).
- **SHAP** is installed as a real dependency (`shap==0.51`) and used for the challenger; the
  reportable model uses exact native linear SHAP (no serve-time dependency). The impurity
  fallback (risk R4) remains for environments without shap.
- **Open items** are tracked in [`known_gaps.md`](known_gaps.md): Docker build not runnable in
  this environment (Dockerfile present, endpoints verified via TestClient), `woe.py` per-file
  coverage (features/ package aggregate ≥95%), and a few algorithmic constants left in code.

Per-phase commit hashes are in [`progress.md`](progress.md).
