# DeskGraph

**A small front-office quant platform for equity derivatives.** A tested pricing
library, immutable market data and a dependency-graph risk engine, with an AI
copilot that turns sales requests for quotes into priced, human-approved quotes.

> **Every number comes from the pricer, never from the model. Every quote needs a
> human approval.**

This README describes the finished system. The [Status](#status) table shows what
is built today, and [Results](#results) stays empty until a script in the repo has
measured each number.

---

## Contents

[What it does](#what-it-does) · [How it's built](#how-its-built) ·
[Features](#features) · [Quick start](#quick-start) · [Usage](#usage) ·
[Configuration](#configuration) · [Market data](#market-data) ·
[Testing and quality](#testing-and-quality) · [Evals](#evals) ·
[Results](#results) · [Project layout](#project-layout) · [Status](#status) ·
[Documentation](#documentation) · [Limitations](#limitations) ·
[References](#references)

---

## What it does

DeskGraph models an equity structured-products desk at small scale. Sales receive
requests for quotes (RFQs), structurers price them, traders manage the risk and the
daily P&L. AI handles the routine parts; people approve anything that leaves the desk.

**Sales: from a chat message to a pending quote**

```text
sales> Price me a 1y autocallable on SPY, 60% knock-in, quarterly observations.
copilot> What coupon per quarter, and what notional?
sales> 2.5% a quarter, 1 million.
copilot> Priced with Monte Carlo on snapshot sha256:… (seed and paths in the quote's
         lineage). Quote Q-… created and waiting for approval by a trader or desk head.
```

**Trader: a scenario on the live book**

```text
trader> What's my vega by expiry if vol drops 5 points?
copilot> (table of vega by expiry, every figure taken from the run_scenario tool output)
```

**Desk head: why did P&L move?**

```text
desk_head> Why did P&L move yesterday?
copilot> (P&L split into delta, gamma, vega, theta and vanna, plus the unexplained
          remainder, from the pnl_explain tool)
```

If the copilot's answer contains a number that didn't come from a tool, a retrieved
document or the user's own message, the answer is blocked.

---

## How it's built

Six layers, each depending only on the ones below. The AI reaches the system only
through MCP tools that call the HTTP API, so it can't skip validation, role checks
or lineage.

```text
AI layer        MCP server · agent harness · RFQ copilot · RAG · guardrails
   │ MCP tool calls → HTTP
Services        FastAPI: price, risk, scenarios, P&L explain, quotes, approvals, lineage
Risk engine     greeks ladders · spot × vol scenarios · P&L explain
Dependency graph  input and price nodes · dirty-flag invalidation · scenario contexts
Quant core      instruments · Black-Scholes · Monte Carlo · SVI · greeks   (pure, no I/O)
Market data     immutable content-hashed snapshots · Parquet · Postgres
```

Across every layer: evals gate CI, Langfuse and OpenTelemetry trace every call,
Docker and Terraform deliver it.

Full blueprint and every design decision: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Features

**Pricing core** (`deskgraph.core`, pure library)

- Products: European options, barrier options (down/up, in/out, daily monitoring),
  autocallables (memory coupons, daily or at-maturity knock-in).
- Black-Scholes-Merton with dividend yield: price, delta, gamma, vega, theta, rho.
- Monte Carlo on exact GBM: seeded and reproducible, antithetic variates, control
  variate for Europeans, O(paths) memory for path-dependent products.
- Bump-and-revalue greeks with common random numbers.
- Raw SVI volatility surface per expiry, butterfly and calendar arbitrage checks,
  interpolation in total variance.

**Market data**

- Daily capture of listed option chains; raw data saved before any cleaning.
- Cleaning (zero bids, wide spreads, stale quotes), mid prices, implied vols by Brent's method.
- Rate curve from FRED; dividend yield from put-call parity or flat.
- Immutable snapshots: `snapshot_id` is a content hash, so the same id always means
  the same inputs.

**Risk**

- Dependency graph that recomputes only what changed, with scenario contexts that
  never touch base values.
- Greeks ladders by expiry and strike; spot × vol scenario grids.
- P&L explain between two snapshots, per position and per book, with vanna and an
  unexplained remainder.

**Services**

- FastAPI with OpenAPI docs; background jobs for batch work; idempotency keys on POSTs.
- Lineage on every response: snapshot, engine version, git SHA, seed, paths, time.
- Quote lifecycle with human approval and an append-only audit log.

**AI copilot**

- RFQ parsing from free text into a typed request, business-rule checks, follow-up
  questions for missing fields.
- MCP server with seven tools, each validated and limited by desk role.
- Hand-written agent harness: step cap, timeouts, retries, typed final answers,
  full tracing.
- Guardrails: number check, PII masking, prompt-injection screening, per-role tool lists.
- Term-sheet extraction from public KIDs and hybrid RAG (BM25 + embeddings,
  reciprocal rank fusion, reranker) with citations.
- Model routing: small model for parsing, larger model for reasoning, local model
  for comparison.

---

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) (it installs the pinned Python
version itself). Docker for the API and database. AWS credentials for the copilot.

```bash
git clone <repo-url> deskgraph && cd deskgraph
uv sync --all-extras
uv run pytest                      # the pricing library works with nothing else
```

Full local stack:

```bash
cp .env.example .env               # fill in keys (see Configuration)
docker compose up -d               # API on http://localhost:8000, Postgres with pgvector
uv run python -m deskgraph.ai.chat --role sales
```

API docs: <http://localhost:8000/docs>.

---

## Usage

### As a Python library

```python
from deskgraph.core import black_scholes, monte_carlo
from deskgraph.core.instruments import Autocallable, European, Monitoring, OptionType
from deskgraph.core.market import MarketInputs
from deskgraph.core.monte_carlo import MCConfig

mkt = MarketInputs(spot=42.0, rate=0.10, div_yield=0.0, vol=0.20)
call = European(OptionType.CALL, strike=40.0, expiry=0.5)

black_scholes.price(call, mkt)  # 4.76, Hull's textbook example
black_scholes.greeks(call, mkt)  # delta, gamma, vega, theta, rho

result = monte_carlo.price(call, mkt, MCConfig(n_paths=200_000, seed=7))
result.price, result.std_error  # same seed, same numbers, every time

note = Autocallable(
    expiry=1.0,
    observation_times=(0.25, 0.5, 0.75, 1.0),
    autocall_level=1.0,  # fractions of initial spot
    coupon=0.025,  # per quarter, memory
    knock_in_level=0.6,
    knock_in_monitoring=Monitoring.AT_MATURITY,
)
monte_carlo.price(note, mkt, MCConfig(n_paths=200_000, seed=7))
mkt.snapshot_id  # "sha256:…" for lineage
```

### Over HTTP

```bash
curl -s -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -H "X-Desk-Role: structurer" \
  -H "Idempotency-Key: 3f1c9a52-…" \
  -d '{
        "instrument": {"type": "european", "option_type": "call", "strike": 720.0, "expiry": 0.5},
        "snapshot_id": "sha256:…",
        "method": "monte_carlo",
        "mc": {"n_paths": 200000, "seed": 42}
      }'
```

```json
{
  "price": "<number>",
  "std_error": "<number>",
  "lineage": {
    "snapshot_id": "sha256:…",
    "engine_version": "0.1.0",
    "git_sha": "…",
    "method": "monte_carlo",
    "seed": 42,
    "n_paths": 200000,
    "created_at": "…",
    "request_id": "…"
  }
}
```

| Endpoint | Purpose |
|---|---|
| `POST /price` | Price one instrument (Black-Scholes or Monte Carlo) |
| `POST /risk` | Greeks for an instrument or a book |
| `POST /scenario` | Spot and vol shifts on the book through the dependency graph |
| `GET /pnl-explain` | P&L explain between two snapshots |
| `GET /snapshots/{id}` | A stored market snapshot |
| `POST /quotes` | Create a pending quote (idempotent) |
| `POST /quotes/{id}/approve` | Human approval or rejection; never available to the AI |

### Copilot in the terminal

```bash
uv run python -m deskgraph.ai.chat --role sales        # or structurer, trader, desk_head
```

The role decides which tools the copilot may use.

### Approving a quote

```bash
curl -s -X POST http://localhost:8000/quotes/<quote-id>/approve \
  -H "Content-Type: application/json" -H "X-Desk-Role: desk_head" \
  -d '{"approver": "<your name>", "decision": "approve", "comment": "within limits"}'
```

### From an MCP client (Claude Desktop, Claude Code)

```json
{
  "mcpServers": {
    "deskgraph": {
      "command": "uv",
      "args": ["run", "python", "-m", "deskgraph.ai.mcp_server"],
      "env": { "DESKGRAPH_API_URL": "http://localhost:8000", "DESKGRAPH_ROLE": "trader" }
    }
  }
}
```

Tools: `price_instrument`, `compute_greeks`, `run_scenario`, `pnl_explain`,
`get_snapshot`, `search_termsheets`, `create_quote`.

---

## Configuration

Set in `.env` locally (git-ignored; copy `.env.example`), or Secrets Manager on AWS.

| Variable | Used by | Purpose |
|---|---|---|
| `DESKGRAPH_DATABASE_URL` | API | Postgres connection string |
| `DESKGRAPH_API_URL` | MCP server, chat | Where the API runs |
| `DESKGRAPH_ROLE` | MCP server | Desk role for the session |
| `AWS_REGION`, `AWS_PROFILE` | AI layer | Bedrock access |
| `DESKGRAPH_MODEL_SMALL` | AI layer | Model for parsing and extraction |
| `DESKGRAPH_MODEL_LARGE` | AI layer | Model for agent reasoning |
| `OLLAMA_HOST`, `DESKGRAPH_MODEL_LOCAL` | AI layer | Local comparison model |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | AI layer | Tracing |
| `FRED_API_KEY` | Data loaders | Rates from FRED |
| `GIT_SHA` | API | Set at build time for lineage |

---

## Market data

Option chains can't be downloaded for past dates, so history is recorded once a
trading day:

```bash
uv run scripts/capture_chain.py              # SPY by default; or: ... SPY QQQ
```

It writes `data/raw/chains/<TICKER>/<date>/chain.parquet` and `meta.json` (spot,
timestamps, row counts, a sha256 of the chain), skips days already captured, and
never leaves half-written data. It carries its own dependencies and doesn't touch
the project environment. `data/raw/` is git-ignored; back it up separately.

Schedule it on Windows (weekdays, 15:45 New York time = 20:45 UK):

```powershell
$a = New-ScheduledTaskAction -Execute "uv.exe" -Argument "run scripts\capture_chain.py" -WorkingDirectory "<repo path>"
$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 20:45
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "DeskGraph chain capture" -Action $a -Trigger $t -Settings $s
```

Snapshots built from these raw files are immutable and identified by a content hash.

---

## Testing and quality

```bash
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run mypy                                           # strict typing
uv run pytest                                         # unit, property and integration tests
uv run pytest benchmarks --benchmark-only             # speed and accuracy benchmarks
uv run python -m evals.run --suite fast               # AI evals (full: --suite full)
```

What the tests guarantee:

- Black-Scholes matches textbook values; put-call parity holds; analytic greeks
  match finite differences.
- Monte Carlo agrees with Black-Scholes within 3 standard errors; knock-in plus
  knock-out equals the vanilla; autocallable edge cases are exact.
- Pricing laws hold for random inputs (Hypothesis): bounds, monotonicity in vol and spot.
- The core library does no I/O, reads no clock and keeps no global state (checked
  on every module).
- Any API result can be reproduced exactly from its lineage.
- The AI can't approve quotes, and a made-up number in an answer is blocked.

CI (GitHub Actions) runs lint, types, tests and the fast eval suite on every pull
request; benchmarks are tracked; the full eval suite runs nightly.

---

## Evals

| Dataset | Size | Measures |
|---|---|---|
| `rfq_cases.jsonl` | 200 synthetic RFQs, some with missing fields | Field accuracy, correct follow-up questions |
| `termsheets/` | Hand-labelled public KIDs | Field accuracy per field |
| `rag_questions.jsonl` | 100 questions with gold chunks | recall@10, MRR |
| `answers_labelled.jsonl` | 50 hand-labelled answers | LLM-judge agreement with human labels, faithfulness |
| `agent_tasks.jsonl` | 50 tasks with expected tool calls and numbers | Task success, numeric grounding |
| `injection_cases/` | 30 documents with planted instructions | Successful injections |

Results are written to `evals/results/*.json` and summarised below.

---

## Results

Every number in this section is produced by a script in the repo; the script is
named next to it. Empty cells haven't been measured yet.

**Pricing and performance** (`benchmarks/`)

| Metric | Target | Result | Script |
|---|---|---|---|
| MC error vs Black-Scholes (European) | within 3 standard errors | | `benchmarks/` |
| MC throughput at 1M paths | tracked | | `benchmarks/` |
| Vectorised vs Python-loop speedup | tracked | | `benchmarks/` |
| Incremental vs full revaluation speedup | tracked | | `benchmarks/` |

**Volatility and risk**

| Metric | Target | Result | Script |
|---|---|---|---|
| SVI fit RMSE per expiry, liquid strikes | < 0.5 vol points | | |
| Arbitrage violations after fit | 0, or each flagged | | |
| Unexplained P&L, vanilla book, normal days | < 5% | | |

**AI copilot** (`evals/`)

| Metric | Target | Result | Script |
|---|---|---|---|
| RFQ field accuracy | set before first run | | `evals.run` |
| Numeric grounding | 100% | | `evals.run` |
| Agent task success | set before first run | | `evals.run` |
| Successful prompt injections | 0 | | `evals.run` |
| RAG recall@10 | set before first run | | `evals.run` |
| Term-sheet extraction accuracy | set before first run | | `evals.run` |
| p95 latency, cost per query | tracked | | Langfuse export |

---

## Project layout

```text
deskgraph/
├── README.md
├── CLAUDE.md                  # rules for Claude Code
├── pyproject.toml, uv.lock    # dependencies and tool config
├── docs/
│   ├── SPEC.md                # what each component must do, and when it's done
│   ├── ARCHITECTURE.md        # blueprint, build plan, decision log
│   ├── WHEN_IN_DOUBT.md       # rules, scope, conventions, done checklist
│   └── adr/                   # one page per major decision
├── scripts/capture_chain.py   # daily option chain capture
├── src/deskgraph/
│   ├── core/                  # pure quant library
│   ├── data/                  # loaders, cleaning, snapshots
│   ├── graph/                 # dependency graph
│   ├── risk/                  # ladders, scenarios, P&L explain
│   ├── api/                   # FastAPI service, lineage, quotes
│   └── ai/                    # MCP server, harness, RFQ copilot, RAG, guardrails
├── tests/                     # unit, property, integration
├── evals/                     # datasets, runner, results
├── benchmarks/                # speed and accuracy
├── infra/terraform/           # AWS deployment
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Status

| Milestone | Dates | Scope | State |
|---|---|---|---|
| M1 Pricing core | 2026-11-26 → 2026-12-09 | Black-Scholes, Monte Carlo, greeks, CI, benchmarks | **In progress**: project setup, CI, market inputs, instruments, Black-Scholes and Monte Carlo done; daily capture running |
| M2 AI copilot | 2026-12-10 → 2026-12-30 | API, quotes and approval, MCP, harness, RFQ copilot, guardrails, evals | Not started |
| M3 Data, vol and risk | 2026-12-31 → 2027-01-13 | Snapshots, SVI, dependency graph, risk, P&L explain, extraction, RAG | Not started |
| M4 Polish | 2027-01-14 → 2027-01-20 | Docker, Terraform deploy, ADRs, results, demo | Not started |

Step-by-step progress: [docs/ARCHITECTURE.md § Build plan](docs/ARCHITECTURE.md#12-build-plan).

---

## Documentation

| Document | Read it for |
|---|---|
| [docs/SPEC.md](docs/SPEC.md) | Requirements and "done when" for every component |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How it fits together, the plan, every decision and why |
| [docs/WHEN_IN_DOUBT.md](docs/WHEN_IN_DOUBT.md) | Rules that never bend, scope, conventions, checklists |
| [docs/adr/](docs/adr/) | Full record of each major decision |

---

## Limitations

- **Not for real trading or investment decisions.** Prices are model outputs on
  delayed, unofficial public data.
- Equity derivatives only; three products; GBM dynamics with an SVI surface (no
  stochastic or local vol unless the stretch goal lands).
- Yahoo Finance data is unofficial and delayed; history exists only from the first
  capture day (2026-11-25).
- Roles are demo-level identities, not real authentication.
- No real client data is used anywhere.

---

## License

Not chosen yet.

## References

- [AI at BNP Paribas: Chat2Trade turns client RFQs from chat and email into structured requests](https://group.bnpparibas/en/news/ai-at-bnp-paribas-1-producing-reliable-quotes-faster-to-help-global-markets-clients-in-trading)
- [FIA, Nov 2026: Generative AI moves from experiment to execution in capital markets](https://www.fia.org/marketvoice/articles/generative-ai-moves-experiment-execution-capital-markets)
- [Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
- Hull, *Options, Futures, and Other Derivatives*: textbook values used in tests.
- Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*: SVI and the butterfly condition.
