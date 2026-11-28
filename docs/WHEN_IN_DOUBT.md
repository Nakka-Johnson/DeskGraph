# When in doubt

Read this before starting anything you're unsure about: a new feature, a new
dependency, a design choice, a change of direction. If this page doesn't settle
it, check the decision log in [ARCHITECTURE.md](ARCHITECTURE.md#decision-log). If
that doesn't settle it either, stop and decide explicitly (see the end of this page).

## The one sentence

> DeskGraph prices and risk-manages equity derivatives correctly and reproducibly,
> and lets AI handle the routine desk work, but **every number comes from the
> pricer and every quote needs a human approval.**

If a change doesn't serve that sentence, it doesn't belong in the project.

## Which document answers what

| Question | Document |
|---|---|
| What is this and how do I run it? | [README.md](../README.md) |
| What must each component do, and when is it done? | [SPEC.md](SPEC.md) |
| How does it fit together, what's the plan, what did we decide and why? | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Am I about to go the wrong way? | This page |
| The full reasoning behind one big decision | [adr/](adr/) |
| Rules for Claude Code | [CLAUDE.md](../CLAUDE.md) |

**If two documents disagree**, the most recent *Agreed* entry in the
ARCHITECTURE decision log wins. Fix the other document in the same change. Never
leave the conflict in place.

## Rules that never bend

1. **The LLM never produces a number.** Prices, greeks, risk and P&L come from tool
   outputs. The number-check guardrail blocks any answer containing a number that
   isn't in a tool output, a retrieved chunk or the user's own request.
2. **No quote leaves without a human approval record.** The AI can create a
   *pending* quote; it can never approve one. There is no `approve` MCP tool.
3. **`src/deskgraph/core` is pure.** No I/O, no network, no clock, no logging,
   no mutable globals. `tests/test_core_purity.py` enforces it.
4. **Randomness is seeded and passed in.** Every random number comes from a
   `numpy.random.Generator` built from a seed in the config. The same inputs and
   seed always give the same price.
5. **Every result carries lineage**: `snapshot_id`, engine version, git SHA, seed,
   path count, timestamp. A result without lineage is a bug.
6. **Market data is immutable.** A `snapshot_id` is a content hash; the same id
   always means the same inputs. Snapshots are never edited, only superseded.
7. **Layers only depend downwards**: AI → services → risk → graph → core → data
   types. The AI reaches the system only through MCP tools over the API.
8. **Tests come before pricing code**: closed forms, parity identities, Hypothesis
   properties.
9. **Every reported metric is produced by a script in the repo.** No hand-typed
   numbers in the README.
10. **No real client data.** Public option chains, public rates, public term sheets.

## Scope

**In scope**

- Equity derivatives on listed underlyings (SPY first).
- Three products: European, barrier, autocallable.
- Black-Scholes, Monte Carlo (GBM), SVI surface, bump greeks, dependency graph,
  scenarios, P&L explain.
- RFQ copilot, MCP tools, a hand-written agent harness, guardrails, RAG over
  public term sheets, evals.
- FastAPI service, Postgres, Docker Compose, Terraform on AWS, GitHub Actions.

**Out of scope** (say no, or park it under *Stretch* in the SPEC)

- Price prediction, backtesting, trading signals or strategies.
- Other asset classes (rates, FX, credit, commodities) and other products.
- Low-latency or real-time trading, order routing, execution.
- A web UI. The interfaces are the API, a CLI chat and MCP.
- Real authentication or user management. Roles are demo-level (see ARCHITECTURE).
- Real client data or real sending of quotes to anyone.

**Stretch only, after M4**: Kafka live risk, Dupire local vol, Ray batches, the
local-model comparison write-up.

## Tie-breakers

When two goals pull against each other, the higher one wins:

1. **Correctness**: the number is right and tested against something independent.
2. **Reproducibility**: the same inputs give the same output, with lineage to prove it.
3. **Safety of the AI path**: guardrails, approval, audit.
4. **Simplicity**: the smallest design that meets the SPEC.
5. **Speed**: only after it's correct, and only with a benchmark to show it.
6. **More features**: last.

## Where does this code go?

| If the code… | It goes in |
|---|---|
| is maths on in-memory values (pricing, greeks, SVI) | `src/deskgraph/core/` |
| reads or writes files, Parquet, Postgres, or calls Yahoo/FRED | `src/deskgraph/data/` |
| caches and invalidates derived values | `src/deskgraph/graph/` |
| aggregates positions, ladders, scenarios, P&L explain | `src/deskgraph/risk/` |
| handles HTTP, jobs, idempotency, lineage stamping | `src/deskgraph/api/` |
| talks to an LLM, MCP, RAG or guardrails | `src/deskgraph/ai/` |
| runs on a schedule outside the app (daily capture) | `scripts/` |
| measures quality of AI behaviour | `evals/` |
| measures speed | `benchmarks/` |
| checks correctness | `tests/` (mirror the `src/` layout) |

If it seems to belong in two places, it's two pieces of code.

## Before you add…

**A dependency.** Ask first (CLAUDE.md). Is there one already in the stack that
does it? Is it maintained and typed? Add it with `uv add`, never by hand, and note
it in the decision log.

**A feature.** Which user story or SPEC component does it serve? If none, stop.
If it's in a later milestone, park it.

**An API endpoint or MCP tool.** Does it return lineage? Does it validate input
with Pydantic? Which roles may call it? Could it let the AI approve or send a
quote? (It must not.)

**A data source.** Is it public and free? Is the raw data saved before cleaning?
Can tests run on a saved fixture with no network?

**A change to a convention** (units, time, levels). Record it in the decision
log first, then update [Conventions](#conventions-and-units) here, then change the
code, in one change.

## Conventions and units

| Quantity | Convention |
|---|---|
| Volatility | Decimal per year: 0.20 means 20% |
| Interest rate, dividend yield | Decimal, continuously compounded, per year |
| Time | Year fractions in M1. When dates arrive: ACT/365 for pricing |
| Monte Carlo time step | Daily = 1/252 of a year (trading days) |
| Vanilla and barrier strikes and barrier levels | Absolute prices (e.g. strike 720.0) |
| Autocallable levels (autocall, knock-in) | Fractions of initial spot S₀ (e.g. 1.0, 0.6); S₀ = `initial_spot`, or spot at pricing when None (D-21) |
| Vanilla price | Per one unit of underlying |
| Autocallable price | Per unit notional (1.0 = par) |
| Delta | Change in price per 1.00 move in spot |
| Gamma | Change in delta per 1.00 move in spot |
| Vega | Per 1.00 change in vol (×0.01 for "per vol point") |
| Theta | Per year (÷365 for "per calendar day") |
| Rho | Per 1.00 change in rate (×0.0001 for "per bp") |
| Bump sizes | Spot 1% relative, vol 1 point, rate 1 bp, time 1 day; time and vol bumps shrink to half the time to the first date ahead and half the vol when smaller (P-21) |
| `snapshot_id` | `sha256:<hex>` of the canonical serialisation of the inputs |

Every docstring that returns one of these states its unit.

## Warning signs: you may be going the wrong way if…

- the model's text contains a number that no tool returned;
- you're writing code in `core/` that needs a file, a URL, the time or a logger;
- a test only passes with a particular random seed, or fails one run in ten;
- you're tuning tolerances until a test goes green instead of finding out why it fails;
- a result can't be reproduced from its lineage alone;
- you're adding a product, asset class or UI that isn't in the SPEC;
- you're building M3 or M4 work while M1 or M2 isn't done;
- a dependency appeared that nobody agreed;
- you can't explain a piece of code without reading it;
- the README shows a number no script produced;
- a change touches many layers at once for a small feature;
- the daily chain capture hasn't run for days (M3 depends on it).

## Done checklist for every change

- [ ] Serves a user story or SPEC component in the current milestone.
- [ ] Tests written first; they fail without the change and pass with it.
- [ ] `uv run ruff check . && uv run ruff format --check .` clean.
- [ ] `uv run mypy` clean (strict).
- [ ] `uv run pytest` green.
- [ ] Units stated in docstrings; maths explained with the formula.
- [ ] New results carry lineage.
- [ ] No new dependency without agreement; decision log updated if anything was decided.
- [ ] Docs that the change makes wrong are fixed in the same change.
- [ ] Diff read line by line; small commit on a feature branch.

## When something isn't covered here

1. Write the question down in one sentence.
2. List the options (two or three), with what each costs.
3. Pick using the tie-breakers above.
4. Add it to the [decision log](ARCHITECTURE.md#decision-log) as *Proposed*; it
   becomes *Agreed* once the project owner confirms.
5. If it's a big decision (it shapes a layer or is hard to undo), write an ADR.

## Glossary

| Term | Meaning |
|---|---|
| RFQ | Request for quote: a client asks the desk for a price on a product |
| Quote | A priced offer created from an RFQ; pending until a human approves it |
| Autocallable | Note that redeems early with coupons if the underlying is at or above the autocall level on an observation date |
| Knock-in | Barrier below which the investor takes the underlying's loss at maturity |
| Barrier option | Option that starts (knock-in) or stops (knock-out) existing when spot crosses a level |
| Snapshot | Immutable, content-hashed set of market inputs for one moment |
| Lineage | Everything needed to reproduce a result: snapshot, engine version, git SHA, seed, paths, time |
| Greeks | Price sensitivities: delta, gamma, vega, theta, rho (and vanna, a spot-vol cross term) |
| P&L explain | Splitting a day's change in value into greek terms, plus what's left unexplained |
| SVI | Stochastic volatility inspired: a five-parameter formula for implied total variance across strikes |
| Total variance | vol² × T; SVI fits this rather than vol |
| Butterfly / calendar arbitrage | Smile shapes that imply negative probabilities / variance falling with expiry |
| Common random numbers | Reusing the same random draws for base and bumped prices so MC noise cancels in greeks |
| Antithetic variates | Pairing each random path with its mirror image to reduce variance |
| Control variate | A quantity with a known expectation used to correct MC noise |
| Dirty flag | Marker that a cached value is stale because an input it depends on changed |
| MCP | Model Context Protocol: the standard interface the AI uses to call DeskGraph's tools |
| Agent harness | The loop that lets a model plan, call tools, check results and answer, with limits |
| Number check | Guardrail that blocks answers containing numbers not found in tool outputs |
| RAG | Retrieval-augmented generation: answer using retrieved term-sheet passages, with citations |
| KID | Key information document: a public, standard summary of a structured product |
| ADR | Architecture decision record: one page on one decision, its context and consequences |
