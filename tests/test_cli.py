"""CLI integration: run -> evaluate -> monitor via a temp file-based config."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from creditscorecard.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def cli_config(tmp_path_factory, dataset):
    tmp = tmp_path_factory.mktemp("cli")
    art = (tmp / "artifacts").as_posix()
    rep = (tmp / "reports").as_posix()
    dat = (tmp / "data").as_posix()
    cfg = tmp / "cli.yaml"
    cfg.write_text(
        "data:\n"
        "  adapter: synthetic\n"
        "  date_column: application_date\n"
        f"paths:\n  artifacts_dir: {art}\n  reports_dir: {rep}\n  data_dir: {dat}\n",
        encoding="utf-8",
    )
    new_csv = tmp / "new.csv"
    dataset.to_csv(new_csv, index=False)
    return cfg, new_csv


def test_cli_run(cli_config):
    cfg, _ = cli_config
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "Version:" in result.output


def test_cli_evaluate(cli_config):
    cfg, _ = cli_config
    result = runner.invoke(app, ["evaluate", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "Model version:" in result.output


def test_cli_monitor(cli_config):
    cfg, new_csv = cli_config
    result = runner.invoke(app, ["monitor", "--config", str(cfg), "--new-data", str(new_csv)])
    assert result.exit_code == 0, result.output
    assert "Score PSI:" in result.output
