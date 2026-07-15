"""Tests for the ``ds`` command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

import ds
from ds import cli

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "project"


def test_version_command_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == ds.__version__


def test_no_command_errors() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_slugify() -> None:
    assert cli._slugify("Customer Churn") == "customer_churn"
    assert cli._slugify("time-series demo") == "time_series_demo"


def test_new_scaffolds_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Run as if invoked from a repo root whose template lives at the real path.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "TEMPLATE_DIR", TEMPLATE_DIR)

    assert cli.main(["new", "My Analysis", "-d", "a demo"]) == 0

    project = tmp_path / "projects" / "my_analysis"
    assert (project / "pipeline.py").is_file()
    assert (project / "tests" / "test_pipeline.py").is_file()
    assert (project / "notebooks").is_dir()
    assert "a demo" in (project / "README.md").read_text()
    assert "Created" in capsys.readouterr().out


def test_new_refuses_existing_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "TEMPLATE_DIR", TEMPLATE_DIR)
    (tmp_path / "projects" / "taken").mkdir(parents=True)

    assert cli.main(["new", "taken"]) == 1


def test_new_without_template_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # no templates/ here
    assert cli.main(["new", "whatever"]) == 1
    assert "Template not found" in capsys.readouterr().out
