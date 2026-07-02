# Credit PD Scorecard

Production-grade, reproducible **Probability-of-Default (PD) application
scorecard**. One command develops, evaluates, calibrates, scales, monitors, and
serves a monotonic Weight-of-Evidence (WoE) logistic scorecard, and auto-writes
a validator-ready **Model Development Document (MDD)**.

Built to SR 11-7 expectations: every non-negotiable modeling constraint is
enforced in code **and** guarded by a test.

---

## The one command

```bash
uv run scorecard run --config configs/home_credit.yaml
```

This runs the full development pipeline:

```
ingest → validate (pandera) → sample design (default definition / cohorts / exclusions /
seasoning) → temporal split → monotonic binning (OptBinning) → WoE/IV → IV filter →
iterative VIF → forward selection → statsmodels Logit → sign check + sklearn parity →
calibration → scorecard scaling → Master Scale → discrimination with bootstrap CIs + .632+
optimism → calibration backtest (Brier/ECE/Jeffreys traffic light) → champion-vs-challenger
(DeLong) → explainability (SHAP) → fairness (AIR/proxy) → freeze PSI/CSI reference →
governance model card → persist artifacts → generate 16-chapter MDD + figures
```

Outputs:

| Path | Contents |
| --- | --- |
| `artifacts/model.json` | The complete, serialized model (the artifacts *are* the model). |
| `artifacts/model_card.json` | Governance metadata: git SHA, hashes, versions, assumptions/limitations register. |
| `artifacts/sample_design.json` | Cohorts, exclusions, default definition, dev sizes, per-cohort base rate. |
| `artifacts/discrimination.json` | AUC/Gini/KS with bootstrap CIs, partial AUC, Somers' D, .632+ optimism. |
| `artifacts/calibration_backtest.json` | Brier/ECE/HL + per-grade Jeffreys traffic light + grade HHI. |
| `artifacts/benchmark.json` | Champion vs challenger: OOT Gini±CI, DeLong test, verdict. |
| `artifacts/global_importance.json` | SHAP global importance + interpretability parity. |
| `artifacts/fairness.json` | AIR / SMD / SPD / EOD per protected attribute + proxy scan. |
| `artifacts/tables/*.csv` | IV, per-characteristic WoE, master scale, metrics. |
| `reports/figures/*.png` | ROC, CAP, calibration, score distribution, reliability curve, SHAP summary. |
| `reports/mdd/*.{md,html}` | Model Development Document — 16 per-chapter files + `index.html` + combined document. |

## Other commands

```bash
uv run scorecard evaluate --config configs/home_credit.yaml            # re-run eval from artifacts
uv run scorecard report   --config configs/home_credit.yaml            # regenerate the MDD from artifacts (deterministic)
uv run scorecard monitor  --config configs/home_credit.yaml --new-data new.csv --period-id 2025-Q1   # PSI/CSI + run-log
uv run scorecard monitor-report --config configs/home_credit.yaml      # PSI/CSI trend report over logged periods
uv run scorecard serve                                                 # FastAPI on :8000 (/score + /explain)
```

## Make targets

```
make bootstrap   # uv sync + pre-commit install
make lint        # ruff check + ruff format --check + mypy
make test        # pytest with coverage (fails < 85%)
make run         # the one command
make serve       # FastAPI app
make all         # bootstrap -> lint -> test -> run
```

## Docker

```bash
docker build -t creditscorecard .
docker compose up            # builds a baked-in model and serves the API on :8000
docker compose --profile mlflow up   # also start an MLflow server on :5000
```

---

## Serving

The API loads **only** the serialized artifacts (no training libraries at
inference).

| Endpoint | Description |
| --- | --- |
| `POST /score` | One applicant → `{score, pd, rating_grade, points_breakdown, reason_codes}`. |
| `POST /explain` | Score + points reason codes **+ SHAP-based rationale + points/SHAP agreement** (§5.8). |
| `POST /batch-score` | JSON list of applicants. |
| `GET /model-info` | Feature list, scaling constants, model version/hash. |
| `GET /health` | Liveness + loaded model version. |

The reportable model's SHAP values are **exact and closed-form** (linear in WoE), so
`/explain` adds no heavy dependency to the serving container.

```bash
curl -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"features": {"NAME_CONTRACT_TYPE":"Cash loans","AMT_INCOME_TOTAL":180000,"AMT_CREDIT":500000, ...}}'
```

**Reason codes** are adverse-action style: the characteristics where the
applicant lost the most points versus the best attainable bin.

---

## Core conventions (orientation consistency chain)

The single most important correctness property. All parts are mutually
consistent and asserted by tests:

1. Target `default`: **`1 = Bad` (the modeled event)**, `0 = Good`.
2. `statsmodels.Logit` estimates **`P(Bad)`** directly.
3. WoE is fixed to **`WoE = ln(%Good / %Bad)`** → positive WoE = better applicant.
4. Therefore every WoE coefficient must be **negative** (sign check enforces this).
5. Points: **`points_i = -(WoE_i·β_i + α/n)·Factor + Offset/n`**, `TotalScore = Σ points_i`.
   The leading minus makes score rise with Good:Bad odds; **higher score ⇔ lower PD**.

Scaling: `Factor = PDO/ln(2)`, `Offset = TargetScore − Factor·ln(TargetOdds)`
(TargetOdds is Good:Bad at TargetScore).

Other enforced constraints: transformers `fit` on **train only** (leakage
guard); monotonic WoE bins with explicit Missing/Special bins; IV filter drops
`IV<0.02` and *flags* `IV>0.5` for leakage review; iterative VIF; single final
model fit; **frozen** PSI/CSI reference bins reused verbatim at monitoring time.

---

## Configuration

Config is layered: `configs/base.yaml` provides defaults; a named file
(e.g. `configs/home_credit.yaml`) is deep-merged on top. Invalid config fails
fast with a clear message. Key sections:

| Section | Keys |
| --- | --- |
| `data` | `adapter` (csv\|synthetic), `path`, `target`, `date_column` |
| `split` | `test_size`, `oot_size`, `stratify` |
| `binning` | `min_bin_pct`, `monotonic_trend`, `max_n_bins` |
| `selection` | `iv_min`, `iv_suspicious`, `vif_threshold`, `forward_metric`, `cv_folds` |
| `model` | `engine`, `enforce_sign_check`, `sign_overrides`, `parity_tol` |
| `calibration` | `anchor_default_rate` (null = train base rate) |
| `scaling` | `pdo`, `target_score`, `target_odds`, `rating_grades`, `round_points` |
| `monitoring` | `psi_bins`, `psi_warn`, `psi_alert` |
| `sample_design` | default definition (DPD/cure/re-default), exclusions, `cohort_key`, seasoning, DPD/status/origination columns |
| `reject_inference` | `enabled`, `reject_data_path`, `method`, `bad_rate_multiplier`, `parcels` |
| `discrimination` | `bootstrap_iterations`, `bootstrap_method`, `confidence_level`, partial-AUC, Somers' D |
| `benchmark` | `challenger`, `challenger_params`, `delong_test`, verdict thresholds |
| `calibration_extended` | Brier/ECE toggles, `hosmer_lemeshow_groups`, per-grade Jeffreys + traffic light |
| `fairness` | `protected_attributes`, AIR warn/alert thresholds, `proxy_scan`, `acknowledge_failure` |
| `explainability` | `shap_enabled`, `shap_sample_size`, `reason_codes_top_n`, parity top-K |
| `monitoring_extended` | `runlog_backend` (sqlite\|jsonl), `runlog_path`, `cadence`, trend min periods |
| `governance` | `model_id`, `model_name`, `model_tier`, owner/developer/validator, review dates |
| `tracking` | `mlflow_enabled`, `mlflow_uri` |

`configs/german_credit.yaml` is the offline demo (synthetic German-Credit-shaped data) that
exercises fairness (`age_years`, `foreign_worker`) and is the home for reject-inference examples.

---

## Dataset

The default config trains on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
Kaggle dataset via the generic CSV adapter — no adapter-specific code needed:

```yaml
# configs/home_credit.yaml
data:
  adapter: csv
  path: data/raw/home_credit/application_train.csv
  target: TARGET                # 1 = Bad/default, 0 = Good (already coded this way)
  date_column: application_date # synthetic dates attached; enables temporal OOT
```

Only `application_train.csv` (307k applications, one row per loan, binary
`TARGET`) is used — it's the single flat table this pipeline's WoE/binning
scorecard is built for. The other tables in the Kaggle bundle
(`bureau*.csv`, `previous_application.csv`, `installments_payments.csv`,
`POS_CASH_balance.csv`, `credit_card_balance.csv`) are relational, keyed by
`SK_ID_PREV`/`SK_ID_BUREAU`, and would need a separate join/aggregation step
before they could feed this pipeline; they are not wired in.

To point at a different dataset, add a new `configs/<name>.yaml` with the same
`data:` block (adjust `path`/`target`/`date_column`) — every entry point
(`cli.py`, `app/api.py`, `Makefile`, tests) reads the config path from one
place, so swapping datasets never means hunting down hardcoded paths per file.
The pipeline auto-detects numeric vs categorical columns, bins them
monotonically, and applies the same WoE/selection/scaling machinery. Ensure
the target is coded `1 = Bad`; adjust the pandera contract in
`src/creditscorecard/data/schema.py` if you want strict range checks on your
columns.

---

## Architecture

```
src/creditscorecard/
  config.py          pydantic-settings config (YAML load/validate/merge) — 20 blocks
  logging.py         structured logging
  data/              adapters (csv|synthetic), pandera schema, split;
                     definition_of_default (Basel default/cure/cohorts/exclusions), reject_inference
  features/          binning (OptBinning, frozen specs), WoE/IV, selection (IV/VIF/forward)
  model/             train (Logit + sign check + sklearn parity), calibrate, scorecard scaling
  evaluation/        discrimination (CIs + .632+), calibration (Brier/ECE/Jeffreys),
                     benchmark (challenger + DeLong), fairness (AIR/proxy), explainability (SHAP),
                     curves, stability (PSI/CSI frozen bins), metrics (shim)
  monitoring/        monitor (PSI/CSI + escalation), runlog (multi-period store/trend/AvE/migration)
  governance/        metadata (model card: git SHA, hashes, assumptions/limitations register)
  registry.py        artifact save/load + content-hash version (+ optional MLflow)
  scoring.py         artifact-only scorer + native linear-SHAP explain (shared by pipeline & API)
  reasons.py         adverse-action reason codes
  reporting/         MDD package — one module per chapter under mdd_sections/ (16 chapters), deterministic
  pipeline.py        orchestrates the one command
  cli.py             Typer CLI: run / evaluate / report / monitor / monitor-report / serve
app/api.py           FastAPI service (artifacts only): /score, /explain, /batch-score, /model-info, /health
tests/               constraint guards + e2e (≥85% overall; ≥95% on domain modules)
```

See [`reports/refactor/00_diagnostic.md`](reports/refactor/00_diagnostic.md) and
[`reports/refactor/progress.md`](reports/refactor/progress.md) for the full refactor rationale,
module specifications, and per-phase commit history.

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).
