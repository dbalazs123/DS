---
name: roadmap-honesty
description: >-
  Keep the DS toolkit's roadmap and docs honest and correctly split: append
  history (friction backlogs, completed plan-of-record entries) to
  ROADMAP_ARCHIVE.md continuing the item numbering, keep only the live parts in
  ROADMAP.md under its CI size budget, and sync CLAUDE.md / README.md /
  CHANGELOG.md / docs in the same change. Use this whenever a change alters
  structure, tooling, or the public API and the docs must follow: finishing a
  demand loop, recording friction, closing or striking an item, adding a project
  or a library surface, or any "update the roadmap / keep the docs honest"
  request. Reach for it even when the user just says "note this down" — the
  ROADMAP↔ARCHIVE split is enforced by a CI size gate and is easy to violate by
  appending narrative to the wrong file. Do NOT use it for editing unrelated prose
  docs that have nothing to do with the roadmap/state.
---

# Roadmap honesty

This repo keeps two roadmap files on purpose, and a CI test enforces the split.
Getting history into the wrong file re-inflates the always-read file and fails
CI. The rule is simple: **`ROADMAP.md` is live-only; all history goes to
`ROADMAP_ARCHIVE.md`.**

## The two files

- **`ROADMAP.md`** — small, read in full every session. Holds *only*: "Where
  things stand" (the capability-per-stage table + project roster), the demand
  queue (next up + deprioritized-until-pulled), and the working agreement.
  `tests/test_roadmap_size.py` fails CI if it grows past its line budget — so it
  must stay lean.
- **`ROADMAP_ARCHIVE.md`** — the durable record, not read whole. Holds: the goal
  evaluation, the completed plan of record (`P1…PN`), the per-project friction
  backlogs (**items numbered 1…N**, referenced by number from project code and
  `CHANGELOG.md`), and the settled-decision rationales that `CLAUDE.md`'s
  engineering notes point to.

## The write-side rule (this is the process, not a one-off)

When a demand loop completes or a decision settles:

1. **Append the new history to `ROADMAP_ARCHIVE.md`**, never to `ROADMAP.md`:
   - a `## Friction backlog (from projects/<slug>)` section, **continuing the item
     numbering** from the last item in the archive (grep for the highest `^N.`
     entry — do not restart at 1). Mark each item `served` (built this loop, with
     the function it became), `struck`/`parked` (with the revisit trigger), or
     recorded-and-done-inline. Mirror the format of the existing sections.
   - a completed plan-of-record `P<N>` entry describing what shipped and the
     honest result.
   - Update the archive's intro line that states the `P1…PN` and `items 1…N`
     ranges.
2. **Update only the small live parts of `ROADMAP.md`**: the "Where things stand"
   stage row if a surface changed, the project roster line/count, and the demand
   queue (mark the loop done, refresh the deprioritized list). **Do not** append
   per-loop narrative here — that reflex is exactly what re-bloats the file.
3. **Do not grow `CLAUDE.md`'s roadmap section with narrative** either — same
   re-inflation risk. It points *to* the archive; keep it a pointer.

## Same-change doc sync

Documentation must never drift ahead of (or behind) reality — update these in the
**same** change that altered structure, tooling, or the API:

- **`CLAUDE.md`** — the project list in the repo-tree comment, and the `P`/item
  ranges in the roadmap section if you closed items.
- **`README.md`** — the `projects/` listing line, and any changed command.
- **`CHANGELOG.md`** — an `[Unreleased] › Added`/`Changed`/`Fixed` entry, keeping
  the Keep-a-Changelog format. Reference friction items by number.
- **`docs/guide.md`** — only if a public library surface changed (a new function,
  a new keyword). The top-level API surface is pinned by
  `tests/test_public_api.py`; docs describe stage imports, not a flat surface.

## Verify

```bash
make test                       # runs test_roadmap_size.py (budget) + test_public_api.py (surface)
uv run mkdocs build --strict    # only if you touched docs/
```

If `test_roadmap_size.py` fails, `ROADMAP.md` grew too large — move the prose you
just added into `ROADMAP_ARCHIVE.md` and leave only the live delta behind.
