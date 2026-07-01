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
ingest → validate (pandera) → temporal split → monotonic binning (OptBinning)
→ WoE/IV → IV filter → iterative VIF → forward selection → statsmodels Logit
→ sign check + sklearn parity → calibration → scorecard scaling → Master Scale
→ evaluate (train/test/OOT) → freeze PSI/CSI reference → persist artifacts
→ generate MDD + figures
```

Outputs:

| Path | Contents |
| --- | --- |
| `artifacts/model.json` | The complete, serialized model (the artifacts *are* the model). |
| `artifacts/tables/*.csv` | IV, per-characteristic WoE, master scale, metrics. |
| `reports/figures/*.png` | ROC, CAP, calibration (Hosmer-Lemeshow), score distribution. |
| `reports/mdd/*.{md,html}` | Model Development Document. |

## Other commands

```bash
uv run scorecard evaluate --config configs/home_credit.yaml          # re-run eval from artifacts
uv run scorecard monitor  --config configs/home_credit.yaml --new-data new.csv   # PSI/CSI vs frozen ref
uv run scorecard serve                                                 # FastAPI on :8000
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
| `POST /batch-score` | JSON list of applicants. |
| `GET /model-info` | Feature list, scaling constants, model version/hash. |
| `GET /health` | Liveness + loaded model version. |

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
| `tracking` | `mlflow_enabled`, `mlflow_uri` |

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
  config.py          pydantic-settings config (YAML load/validate/merge)
  logging.py         structured logging
  data/              adapters (csv|synthetic), pandera schema, temporal split
  features/          binning (OptBinning, frozen specs), WoE/IV, selection (IV/VIF/forward)
  model/             train (Logit + sign check + sklearn parity), calibrate, scorecard scaling
  evaluation/        metrics (AUC/Gini/KS), curves, stability (PSI/CSI frozen bins)
  monitoring/        PSI/CSI runner with warn/alert escalation
  registry.py        artifact save/load + content-hash version (+ optional MLflow)
  scoring.py         artifact-only scorer (shared by pipeline & API)
  reasons.py         adverse-action reason codes
  reporting.py       MDD (markdown + HTML)
  pipeline.py        orchestrates the one command
  cli.py             Typer CLI: run / evaluate / monitor / serve
app/api.py           FastAPI service (artifacts only)
tests/               constraint guards + e2e (≥85% coverage)
```

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).
