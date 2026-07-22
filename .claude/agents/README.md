# Subagents

Subagents (`.claude/agents/*.md`) carve a **read-heavy, well-scoped** slice off
the main thread and run it in isolated context. This repo uses them sparingly and
on purpose: the demand loop's spine is sequential value judgment (the aliasing
bar, the fit/apply split, roadmap honesty), which wants continuous context and
must stay in the main thread. Only delegate work that is genuinely fan-out and
does not carry that judgment.

Current roster:

- **`dataset-scout`** — read-only selection reconnaissance for the demand loop
  (Step 2 / Step 10). Recommends a dataset slate; never builds or picks.

## Choosing a subagent's `model:`

The `model:` field is the **only** place model choice is actually enforceable —
there is no per-step model knob for the main conversation (that is the user's
`/model` / session default). So "Opus for judgment, cheaper models for simple
work" only has teeth on work you delegate. Pick the field with that in mind:

| The agent's work is… | `model:` | Why |
|----------------------|----------|-----|
| Carrying real judgment whose *mistake misroutes the loop* — the dataset **selection** shape, anything applying the aliasing bar or roadmap discipline | `inherit` | Getting the shape wrong sends the whole loop down the wrong path; the main thread re-vets, but a weak recommendation wastes that vetting. Inherit = the session's model (Opus in practice). |
| Read-only **search / collection** with little judgment — grep sweeps, gathering file lists, transcribing findings the main thread will interpret | `sonnet` | Cheaper and faster fan-out; the main thread supplies the judgment, so the agent only has to be *thorough*, not *wise*. |
| High-volume trivial **classification / extraction** with a checkable result | `haiku` | Only when the task is genuinely mechanical and the output is easy to validate. Rare here — most "simple" work in this repo is already a deterministic script. |

Two rules that keep this honest:

- **Don't wrap a deterministic script in an agent to downgrade it.** If the work
  is "run `consumer_matrix.py` and report numbers," just run the script — a Haiku
  agent around it adds a round-trip and buys nothing.
- **Never delegate judgment purely to make it cheaper.** If a mistake in the
  slice would violate the ordering rule or the aliasing bar in a way the main
  thread might not catch, it belongs in the main thread on the session model, not
  in a downgraded agent.

`dataset-scout` is deliberately `inherit`: its selection reasoning sits in the
first row (a wrong *shape* misroutes the loop). Downgrading it to `sonnet` is
*defensible* because the main thread confirms the pick via `AskUserQuestion` —
but the default here favors a strong recommendation over a cheap one, since the
shape decision is the highest-leverage judgment in the loop.

## Keep this roster honest

When you add a subagent, add its one-line entry above, set `tools:` to the
minimum it needs (read-only agents get no `Edit`/`Write`/`Agent`), and choose
`model:` from the table with an explicit reason if it is not `inherit`.
