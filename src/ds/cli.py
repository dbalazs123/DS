"""Command-line interface for the DS workspace.

A small ``ds`` command wrapping common workspace tasks. It ships with the
package, so once ``ds`` is installed you can run ``ds --help`` to see what's
available. The ``new`` subcommand scaffolds a project and therefore only works
from a checkout of the DS repository (it needs ``templates/`` and Copier).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ds import __version__

# Resolved relative to the current working directory: `ds new` is meant to be
# run from the repository root, the same as the raw `copier copy` command.
TEMPLATE_DIR = Path("templates/project")
PROJECTS_DIR = Path("projects")


def _slugify(name: str) -> str:
    """Convert a project name into a directory- and import-safe slug."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


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
    print(f"Created {destination}/ — run it with `uv run python {destination}/pipeline.py`")
    return 0


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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ds`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
