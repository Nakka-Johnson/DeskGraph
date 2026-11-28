# DeskGraph: guidance for Claude Code

## What this is
Derivatives pricing and risk platform with an AI RFQ copilot.

## Read before working
- docs/WHEN_IN_DOUBT.md: rules that never bend, scope, conventions and units,
  where code goes, done checklist. Check it whenever unsure.
- docs/ARCHITECTURE.md: blueprint, build plan with step status, decision log,
  open questions. Build only to Agreed decisions; never build on Proposed or
  Open ones without asking.
- docs/SPEC.md: requirements and "done when" per component.
- docs/adr/: full record of major decisions.
If documents disagree, the latest Agreed entry in the decision log wins; fix the
other document in the same change. Record any new decision as Proposed and ask.

## Rules
- src/deskgraph/core is a pure library: no I/O, no network, no globals.
- Instruments and market inputs are frozen dataclasses with validation.
- Every random number comes from a seeded numpy Generator passed in via config.
- The LLM never produces a number. Prices, greeks and risk come from tools,
  and the number-check guardrail enforces it.
- No quote leaves the system without a human approval record.
- Every API response includes lineage: snapshot_id, engine version, git SHA, seed.
- Type everything; mypy --strict must pass. Use ruff for lint and format.
- Write tests before pricing code: closed-form checks, parity identities,
  Hypothesis properties.
- No network in unit tests; use fixtures in tests/fixtures/.
- Explain non-obvious maths in docstrings with the formula and units.

## Commands
- uv sync --all-extras
- uv run ruff check . && uv run ruff format --check .
- uv run mypy
- uv run pytest
- uv run pytest benchmarks --benchmark-only
- uv run python -m evals.run --suite fast

## Workflow
Plan first, then tests, then code. Keep changes small and focused.
Ask before adding a dependency.
