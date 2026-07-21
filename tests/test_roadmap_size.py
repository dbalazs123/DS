"""Enforce size budgets on the always-read docs so they can't silently re-bloat.

``ROADMAP.md`` and ``CLAUDE.md`` are both read into context at the start of most
sessions, so their size is a recurring token cost. They are deliberately kept
small and live-only:

- ``ROADMAP.md`` holds just the *live* roadmap (the "Where things stand" table,
  the demand queue, and the working agreement); completed plan-of-record
  entries, per-project friction backlogs, and settled-decision rationales belong
  in ``ROADMAP_ARCHIVE.md``.
- ``CLAUDE.md`` holds durable guidance; per-project narrative and resolved
  history belong in ``ROADMAP_ARCHIVE.md`` / ``CHANGELOG.md``.

These tests turn that convention into an enforced invariant: appending a new
friction backlog (~140 lines) to the live roadmap — the exact drift that grew it
to ~1,440 lines before — or growing ``CLAUDE.md`` with per-project narrative
trips the budget and fails CI, pointing the author at the archive. Raising a
``*_MAX_LINES`` ceiling is therefore a deliberate decision, made in the same
change, not an accident.
"""

from __future__ import annotations

from pathlib import Path

# Headroom above the current live size (~80 lines) for the capability table and
# demand queue to breathe, but well below the size a single appended backlog
# would push the file to. See the module docstring before raising this.
ROADMAP_MAX_LINES = 120

# Headroom above the current size (~250 lines) for the engineering-notes and
# conventions lists to grow, but not so much that per-project narrative can
# accrete unnoticed. See the module docstring before raising this.
CLAUDE_MD_MAX_LINES = 300

REPO_ROOT = Path(__file__).resolve().parents[1]


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_roadmap_stays_within_line_budget() -> None:
    """The live roadmap must stay small; history goes to ROADMAP_ARCHIVE.md."""
    line_count = _line_count(REPO_ROOT / "ROADMAP.md")
    assert line_count <= ROADMAP_MAX_LINES, (
        f"ROADMAP.md has {line_count} lines, over the {ROADMAP_MAX_LINES}-line "
        "budget. Completed plan-of-record entries, friction backlogs, and "
        "settled-decision rationales belong in ROADMAP_ARCHIVE.md — move the "
        "new history there. Raise ROADMAP_MAX_LINES only as a deliberate "
        "decision (see this test's module docstring)."
    )


def test_claude_md_stays_within_line_budget() -> None:
    """CLAUDE.md is auto-loaded every session; keep it durable-guidance-only."""
    line_count = _line_count(REPO_ROOT / "CLAUDE.md")
    assert line_count <= CLAUDE_MD_MAX_LINES, (
        f"CLAUDE.md has {line_count} lines, over the {CLAUDE_MD_MAX_LINES}-line "
        "budget. Per-project narrative and resolved history belong in "
        "ROADMAP_ARCHIVE.md / CHANGELOG.md, not here. Raise CLAUDE_MD_MAX_LINES "
        "only as a deliberate decision (see this test's module docstring)."
    )


def test_roadmap_archive_exists() -> None:
    """The archive the budgets assume must be present and non-empty."""
    archive = REPO_ROOT / "ROADMAP_ARCHIVE.md"
    assert archive.is_file(), "ROADMAP_ARCHIVE.md is missing (see ROADMAP.md)."
    assert archive.read_text(encoding="utf-8").strip(), "ROADMAP_ARCHIVE.md is empty."
