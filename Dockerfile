# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# uv from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies first (cached across code changes).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Application code and config.
COPY configs ./configs
COPY app ./app

# Bake a model into the image so the API is ready to serve.
# Uses the synthetic adapter (configs/base.yaml default): real data under
# data/raw/ is excluded from the build context by .dockerignore. Serving only
# reads artifacts/model.json, so this doesn't need to match the dataset used
# for a "real" training run (see configs/home_credit.yaml).
RUN uv run scorecard run --config configs/base.yaml

EXPOSE 8000
CMD ["uv", "run", "scorecard", "serve", "--host", "0.0.0.0", "--port", "8000"]
