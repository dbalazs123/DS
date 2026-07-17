"""Tests for the ``ds`` command-line interface."""

from __future__ import annotations

import subprocess
import sys
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


def test_slugify_strips_path_and_punctuation() -> None:
    # No separators survive, so a slug is always a single safe path segment.
    assert cli._slugify("../evil") == "evil"
    assert cli._slugify("a/b.c") == "a_b_c"
    assert cli._slugify("my.project") == "my_project"
    assert cli._slugify("   ") == ""


def test_new_rejects_name_without_usable_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "TEMPLATE_DIR", TEMPLATE_DIR)
    assert cli.main(["new", "!!!"]) == 1
    assert not (tmp_path / "projects").exists()


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

    # The stub keeps the shape every real pipeline keeps: an injectable
    # settings parameter, and the run instructions ds new itself prints.
    stub = (project / "pipeline.py").read_text()
    assert "def run(output_dir: Path, settings: Settings | None = None)" in stub
    assert "ds run my_analysis" in stub
    assert "ds run my_analysis" in (project / "README.md").read_text()
    assert "settings=settings" in (project / "tests" / "test_pipeline.py").read_text()


def test_new_empty_description_leaves_no_dangling_dash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "TEMPLATE_DIR", TEMPLATE_DIR)

    assert cli.main(["new", "Bare"]) == 0

    project = tmp_path / "projects" / "bare"
    first_line = (project / "pipeline.py").read_text().splitlines()[0]
    assert first_line == '"""Bare'
    assert "—" not in (project / "README.md").read_text().splitlines()[0]


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


def _make_project(projects_dir: Path, name: str) -> Path:
    """Create a minimal runnable project directory under ``projects_dir``."""
    project = projects_dir / name
    project.mkdir(parents=True)
    (project / "pipeline.py").write_text("print('ran')\n")
    return project


def test_list_projects_only_returns_runnable_dirs(tmp_path: Path) -> None:
    _make_project(tmp_path, "beta")
    _make_project(tmp_path, "alpha")
    (tmp_path / "no_pipeline").mkdir()  # a dir without pipeline.py is not runnable
    (tmp_path / "loose.py").write_text("")  # a stray file is not a project

    found = cli._list_projects(tmp_path)

    assert [p.name for p in found] == ["alpha", "beta"]  # sorted, runnable only


def test_list_projects_missing_dir_is_empty(tmp_path: Path) -> None:
    assert cli._list_projects(tmp_path / "nope") == []


def test_find_project_matches_literal_slug_and_underscored(tmp_path: Path) -> None:
    _make_project(tmp_path, "customer_churn")
    _make_project(tmp_path, "_example")

    # Literal, spaced and hyphenated forms all slugify to the same project.
    assert cli._find_project("customer_churn", tmp_path).name == "customer_churn"
    assert cli._find_project("Customer Churn", tmp_path).name == "customer_churn"
    assert cli._find_project("customer-churn", tmp_path).name == "customer_churn"
    # A leading-underscore dir (like _example) is reachable via its slug.
    assert cli._find_project("_example", tmp_path).name == "_example"
    assert cli._find_project("example", tmp_path).name == "_example"


def test_find_project_rejects_traversal_and_misses(tmp_path: Path) -> None:
    _make_project(tmp_path, "customer_churn")

    # Never builds a path from the name, so a traversal attempt just matches nothing.
    assert cli._find_project("../evil", tmp_path) is None
    assert cli._find_project("does_not_exist", tmp_path) is None


def test_run_executes_matched_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    projects = tmp_path / "projects"
    _make_project(projects, "customer_churn")
    monkeypatch.setattr(cli, "PROJECTS_DIR", projects)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    # A fuzzy name resolves to the slugged directory and runs its pipeline.py.
    assert cli.main(["run", "Customer Churn"]) == 0
    assert calls == [[sys.executable, str(projects / "customer_churn" / "pipeline.py")]]
    assert "Running" in capsys.readouterr().out


def test_run_propagates_pipeline_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    projects = tmp_path / "projects"
    _make_project(projects, "demo")
    monkeypatch.setattr(cli, "PROJECTS_DIR", projects)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, check=False: subprocess.CompletedProcess(cmd, 3),
    )

    assert cli.main(["run", "demo"]) == 3


def test_run_unknown_project_lists_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    projects = tmp_path / "projects"
    _make_project(projects, "alpha")
    _make_project(projects, "beta")
    monkeypatch.setattr(cli, "PROJECTS_DIR", projects)

    assert cli.main(["run", "ghost"]) == 1
    out = capsys.readouterr().out
    assert "No project matching 'ghost'" in out
    assert "alpha" in out and "beta" in out


def test_run_no_projects_reports_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    projects = tmp_path / "projects"  # never created
    monkeypatch.setattr(cli, "PROJECTS_DIR", projects)

    assert cli.main(["run", "anything"]) == 1
    assert "no runnable projects" in capsys.readouterr().out
