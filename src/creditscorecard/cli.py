"""Typer CLI: ``run``, ``evaluate``, ``monitor``, ``serve``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

from creditscorecard.config import load_config
from creditscorecard.logging import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="Credit PD scorecard CLI.")
logger = get_logger(__name__)

CONFIG_ENV = "CREDITSCORECARD_CONFIG"
DEFAULT_CONFIG = "configs/home_credit.yaml"


@app.command()
def run(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="YAML config path."),
) -> None:
    """Run the full development pipeline (the one command)."""
    configure_logging()
    from creditscorecard.pipeline import run_pipeline

    cfg = load_config(config)
    result = run_pipeline(cfg)
    typer.echo("\nPerformance:\n" + result.metrics.to_string(index=False))
    typer.echo(f"\nArtifacts: {result.artifacts_path}")
    typer.echo(f"MDD:       {result.mdd_paths['html']}")
    typer.echo(f"Version:   {result.payload['version']}")


@app.command()
def evaluate(config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c")) -> None:
    """Re-run evaluation from saved artifacts (no refitting)."""
    configure_logging()
    import pandas as pd

    from creditscorecard.data.adapters import load_dataset
    from creditscorecard.data.schema import validate_dataframe
    from creditscorecard.data.split import split_data
    from creditscorecard.evaluation.metrics import metrics_table
    from creditscorecard.scoring import ScoringModel

    cfg = load_config(config)
    model = ScoringModel.from_config(cfg)
    df = validate_dataframe(load_dataset(cfg), cfg)
    split = split_data(df, cfg)
    target = cfg.data.target

    probs = {}
    for name, frame in {"train": split.train, "test": split.test, "oot": split.oot}.items():
        scored = model.score_frame(frame)
        probs[name] = (frame[target].to_numpy(), scored["pd"].to_numpy())
    table = metrics_table(probs)
    typer.echo(f"Model version: {model.version}\n")
    typer.echo(table.to_string(index=False))
    _ = pd  # keep import explicit for clarity


@app.command()
def monitor(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    new_data: str = typer.Option(..., "--new-data", help="CSV of new applicants."),
) -> None:
    """Compute PSI/CSI of new data vs the frozen reference."""
    configure_logging()
    from creditscorecard.monitoring.monitor import run_monitoring

    cfg = load_config(config)
    report = run_monitoring(cfg, new_data)
    typer.echo(f"Model version: {report.version}  ·  n={report.n_new}")
    typer.echo(f"Score PSI: {report.psi:.4f}  [{report.psi_status}]")
    typer.echo(f"Grade HHI: {report.hhi:.4f}  [{report.hhi_status}]")
    typer.echo("CSI by characteristic:")
    for feat, val in sorted(report.csi.items(), key=lambda kv: kv[1], reverse=True):
        typer.echo(f"  {feat:<28} {val:.4f}  [{report.csi_status[feat]}]")
    typer.echo(f"\nEscalate: {report.escalate}")


@app.command()
def serve(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Launch the FastAPI scoring service (loads artifacts only)."""
    configure_logging()
    import uvicorn

    os.environ[CONFIG_ENV] = config
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from app.api import app as fastapi_app

    uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    app()
