.PHONY: bootstrap lint test run evaluate serve monitor docker all clean

CONFIG ?= configs/home_credit.yaml

bootstrap:
	uv sync --extra dev
	uv run pre-commit install || true

lint:
	uv run ruff check src tests app
	uv run ruff format --check src tests app
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

run:
	uv run scorecard run --config $(CONFIG)

evaluate:
	uv run scorecard evaluate --config $(CONFIG)

monitor:
	uv run scorecard monitor --config $(CONFIG) --new-data $(NEW_DATA)

serve:
	uv run scorecard serve

docker:
	docker build -t creditscorecard:latest .

all: bootstrap lint test run

clean:
	rm -rf artifacts/* reports/figures/* reports/mdd/* .pytest_cache .mypy_cache .ruff_cache
