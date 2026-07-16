"""Command-line interface for the DS workspace.

A small ``ds`` command wrapping common workspace tasks. It ships with the
package, so once ``ds`` is installed you can run ``ds --help`` to see what's
available. The ``new`` and ``run`` subcommands are project-aware and therefore
work from a checkout of the DS repository: ``new`` needs ``templates/`` and
Copier, and ``run`` resolves projects under ``projects/``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ds import __version__

_NON_SLUG = re.compile(r"[^0-9a-z]+")

# Resolved relative to the current working directory: `ds new` is meant to be
# run from the repository root, the same as the raw `copier copy` command.
TEMPLATE_DIR = Path("templates/project")
PROJECTS_DIR = Path("projects")


def _slugify(name: str) -> str:
    """Convert a project name into a directory- and import-safe slug.

    Any run of characters that isn't a lowercase letter or digit collapses to a
    single underscore, so the result is always a safe single path segment (no
    ``/``, ``.`` or other separators that could escape ``projects/``). Returns
    an empty string when ``name`` has no usable characters.
    """
    return _NON_SLUG.sub("_", name.strip().lower()).strip("_")


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    """Scaffold a new project under ``projects/`` from the Copier template."""
    if not TEMPLATE_DIR.is_dir():
        print(f"Template not found at '{TEMPLATE_DIR}'. Run `ds new` from the DS repo root.")
        return 1

    try:
        from copier import run_copy
    except ImportError:
        print("Copier is required to scaffold projects. Install dev tools with `uv sync`.")
        return 1

    slug = _slugify(args.name)
    if not slug:
        print(f"'{args.name}' has no usable characters for a project name.")
        return 1

    destination = PROJECTS_DIR / slug
    if destination.exists():
        print(f"'{destination}' already exists — choose another name or remove it first.")
        return 1

    run_copy(
        str(TEMPLATE_DIR),
        str(destination),
        data={
            "project_name": args.name,
            "project_slug": slug,
            "description": args.description or "",
        },
        defaults=True,
        quiet=True,
    )
    print(f"Created {destination}/ — run it with `ds run {slug}`")
    return 0


def _list_projects(projects_dir: Path) -> list[Path]:
    """Return the runnable project directories (those with a ``pipeline.py``).

    Sorted by directory name for stable, human-friendly listings.
    """
    if not projects_dir.is_dir():
        return []
    return sorted(
        (p for p in projects_dir.iterdir() if p.is_dir() and (p / "pipeline.py").is_file()),
        key=lambda p: p.name,
    )


def _find_project(name: str, projects_dir: Path) -> Path | None:
    """Resolve ``name`` to an existing project directory, or ``None``.

    The lookup never builds a path from ``name``; it enumerates the real
    directories under ``projects/`` and selects one whose name matches ``name``
    either literally or after slugification (so ``"Customer Churn"``,
    ``customer_churn`` and ``customer-churn`` all resolve to the same project,
    and ``_example`` is reachable as ``example``). Because only existing
    entries are ever selected, a traversal attempt like ``../evil`` simply
    matches nothing — the same slug discipline that keeps ``ds new`` inside
    ``projects/``.
    """
    candidates = {name, _slugify(name)}
    for project in _list_projects(projects_dir):
        if project.name in candidates or _slugify(project.name) in candidates:
            return project
    return None


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a project's ``pipeline.py`` by name, from the ``projects/`` area."""
    project = _find_project(args.name, PROJECTS_DIR)
    if project is None:
        available = _list_projects(PROJECTS_DIR)
        if available:
            names = ", ".join(p.name for p in available)
            print(f"No project matching '{args.name}'. Available projects: {names}")
        else:
            print(f"No project matching '{args.name}'; no runnable projects under {PROJECTS_DIR}/.")
        return 1

    pipeline = project / "pipeline.py"
    print(f"Running {pipeline}")
    completed = subprocess.run([sys.executable, str(pipeline)], check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``ds`` argument parser."""
    parser = argparse.ArgumentParser(prog="ds", description="DS workspace command-line tools.")
    parser.add_argument("--version", action="version", version=f"ds {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the installed ds version.")
    version_parser.set_defaults(func=_cmd_version)

    new_parser = subparsers.add_parser("new", help="Scaffold a new project under projects/.")
    new_parser.add_argument("name", help="Project name; also used to derive the directory slug.")
    new_parser.add_argument(
        "-d", "--description", default=None, help="One-line description of the project."
    )
    new_parser.set_defaults(func=_cmd_new)

    run_parser = subparsers.add_parser(
        "run", help="Run a project's pipeline.py by name (resolved under projects/)."
    )
    run_parser.add_argument(
        "name", help="Project name or slug; matched against the directories under projects/."
    )
    run_parser.set_defaults(func=_cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ds`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
