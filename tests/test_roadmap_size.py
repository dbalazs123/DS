"""Enforce the ROADMAP.md size budget so history can't silently re-bloat it.

``ROADMAP.md`` is read into context at the start of most sessions, so its size
is a recurring token cost. The file is deliberately kept to the *live* roadmap
(the "Where things stand" table, the demand queue, and the working agreement);
completed plan-of-record entries, per-project friction backlogs, and
settled-decision rationales belong in ``ROADMAP_ARCHIVE.md`` instead.

This test turns that convention into an enforced invariant: appending a new
friction backlog (~140 lines) to the live file — the exact drift that grew it to
~1,440 lines before — trips the budget and fails CI, pointing the author at the
archive. Raising ``ROADMAP_MAX_LINES`` is therefore a deliberate decision, made
in the same change, not an accident.
"""

from __future__ import annotations

from pathlib import Path

# Headroom above the current live size (~70 lines) for the capability table and
# demand queue to breathe, but well below the size a single appended backlog
# would push the file to. See the module docstring before raising this.
ROADMAP_MAX_LINES = 120

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_roadmap_stays_within_line_budget() -> None:
    """The live roadmap must stay small; history goes to ROADMAP_ARCHIVE.md."""
    roadmap = REPO_ROOT / "ROADMAP.md"
    line_count = len(roadmap.read_text(encoding="utf-8").splitlines())
    assert line_count <= ROADMAP_MAX_LINES, (
        f"ROADMAP.md has {line_count} lines, over the {ROADMAP_MAX_LINES}-line "
        "budget. Completed plan-of-record entries, friction backlogs, and "
        "settled-decision rationales belong in ROADMAP_ARCHIVE.md — move the "
        "new history there. Raise ROADMAP_MAX_LINES only as a deliberate "
        "decision (see this test's module docstring)."
    )


def test_roadmap_archive_exists() -> None:
    """The archive the budget assumes must be present and non-empty."""
    archive = REPO_ROOT / "ROADMAP_ARCHIVE.md"
    assert archive.is_file(), "ROADMAP_ARCHIVE.md is missing (see ROADMAP.md)."
    assert archive.read_text(encoding="utf-8").strip(), "ROADMAP_ARCHIVE.md is empty."
