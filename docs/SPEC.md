# DeskGraph project spec

Nov 26, 2026

DeskGraph is a small front-office quant platform in Python: a pricing library, versioned market data and a dependency-graph risk engine, with an AI copilot that turns sales RFQs into priced, human-approved quotes. Build it in four milestones over about eight weeks; each one is demo-able and produces real numbers for your CV.

## What it is and why

DeskGraph mimics an equity structured-products desk. Sales receive requests for quotes (RFQs) by chat and email, structurers price them, and traders manage the risk and the daily P&L. The project rebuilds that loop at small scale, with AI doing the routine parts and humans approving anything that leaves the desk.

### Three user stories

1. **Sales:** "Price me a 1y autocallable on SPY, 60% knock-in, quarterly observations." The copilot parses it into a typed trade request, asks for any missing field, prices it through tools, and holds the quote until a human approves it. Every step is audit-logged.
2. **Trader:** "What's my vega by expiry if vol drops 5 points?" The agent builds that scenario on the dependency graph and answers with numbers that came from the pricer, never from the model.
3. **Desk head:** "Why did P&L move yesterday?" P&L explain splits the change into delta, gamma, vega and theta, and shows what is left unexplained.

### What it proves for the UBS JD

| JD line | Where DeskGraph shows it |
|---|---|
| Quantitative models and analytics | Pricing library, SVI surface, risk engine, P&L explain |
| Production changes to large complex systems | Typed, tested, CI-gated code with lineage on every result |
| AI-powered systems from idea to production | RFQ copilot, MCP tools, agent harness, evals |
| Sound judgment with AI | Number check, human approval, audit log, injection tests |
| Partnering with Trading, Structuring and Sales | User stories built around their workflows |
| numpy, polars, scipy; harnesses; MCP | Named directly in the stack |

### Non-goals

- No price prediction, backtesting or trading strategies: that is buy-side work, not this role.
- No real client data: public option chains, public rates and public term sheets only.
- One asset class (equity derivatives) and three products (European, barrier, autocallable), done properly.
- Not low-latency or real-time trading; correctness and reproducibility come first.

## Architecture

Six layers, each depending only on the ones below it, so the maths can be tested on its own and the AI can never bypass it.

**The AI layer only calls tools; every number comes from the quant core.**

```
┌──────────────────────────────────────────────────────────────┐   ┌───────────────────────────┐
│ AI layer                                                     │   │ Across every layer        │
│ MCP server · agent harness · RFQ copilot · RAG · guardrails  │   │                           │
│ · approvals                                                  │   │ Evals in CI               │
└──────────────────────────────┬───────────────────────────────┘   │   datasets and metrics    │
                               │ MCP tool calls                    │   gate every merge        │
┌──────────────────────────────▼───────────────────────────────┐   │                           │
│ Services                                                     │   │ Observability             │
│ FastAPI pricing, risk and quote API · async jobs · lineage   │   │   Langfuse traces         │
│ on every result                                              │   │   OpenTelemetry, logs     │
└──────────────────────────────┬───────────────────────────────┘   │                           │
┌──────────────────────────────▼───────────────────────────────┐   │ Infra                     │
│ Risk engine                                                  │   │   Docker, Terraform       │
│ Greeks ladders · spot x vol scenarios · daily P&L explain ·  │   │   AWS, GitHub Actions     │
│ batch runs                                                   │   │                           │
└──────────────────────────────┬───────────────────────────────┘   │                           │
┌──────────────────────────────▼───────────────────────────────┐   │                           │
│ Dependency graph                                             │   │                           │
│ Nodes for inputs and prices · dirty-flag invalidation ·      │   │                           │
│ scenario contexts                                            │   │                           │
└──────────────────────────────┬───────────────────────────────┘   │                           │
┌──────────────────────────────▼───────────────────────────────┐   │                           │
│ Quant core (pure library)                                    │   │                           │
│ Instruments · Black-Scholes · Monte Carlo · SVI surface ·    │   │                           │
│ greeks                                                       │   │                           │
└──────────────────────────────┬───────────────────────────────┘   │                           │
┌──────────────────────────────▼───────────────────────────────┐   │                           │
│ Market data                                                  │   │                           │
│ Immutable snapshots · option chains, rates, KIDs · Postgres  │   │                           │
│ and Parquet                                                  │   │                           │
└──────────────────────────────────────────────────────────────┘   └───────────────────────────┘
```

*DeskGraph architecture · 6 layers, 3 cross-cutting concerns*

The AI layer reaches the system only through MCP tools; services call the risk engine and graph; all maths lives in the pure quant core, on top of immutable snapshots. Evals, observability and infra apply to every layer.

## Repo layout and stack

One Python package with a pure quant core at the bottom and everything with I/O above it, so the maths can be tested without a network or a database.

```
deskgraph/
├── CLAUDE.md                  # conventions for Claude Code
├── pyproject.toml             # uv, ruff, mypy --strict, pytest
├── docs/
│   ├── SPEC.md                # this document, exported as Markdown
│   └── adr/                   # one page per design decision
├── src/deskgraph/
│   ├── core/                  # pure quant library: no I/O, no network
│   │   ├── market.py          # MarketInputs, rate curve
│   │   ├── instruments.py     # European, Barrier, Autocallable
│   │   ├── black_scholes.py   # closed form + analytic greeks
│   │   ├── monte_carlo.py     # GBM engine, variance reduction
│   │   ├── greeks.py          # bump-and-revalue, common random numbers
│   │   └── vol/svi.py         # SVI fit + arbitrage checks
│   ├── data/                  # loaders, cleaning, snapshot store
│   ├── graph/                 # dependency graph, scenario contexts
│   ├── risk/                  # ladders, scenario grids, P&L explain
│   ├── api/                   # FastAPI app, jobs, lineage
│   └── ai/
│       ├── mcp_server.py      # tools over the API
│       ├── harness.py         # agent loop, step cap, retries
│       ├── rfq.py             # RFQ schema, parser, validation
│       ├── termsheets.py      # PDF to typed instrument
│       ├── rag/               # chunking, BM25 + dense, reranker
│       └── guardrails.py      # number check, PII mask, injection filter
├── evals/                     # datasets, runners, results
├── benchmarks/                # pytest-benchmark suites
├── tests/                     # unit, property (Hypothesis), integration
├── infra/terraform/           # AWS deploy
└── .github/workflows/ci.yml   # lint, types, tests, evals, benchmarks
```

| Area | Choice | Why |
|---|---|---|
| Language and env | Python 3.12, uv | Fast installs and a lockfile CI can trust |
| Numerics | numpy, scipy, polars | Vectorised Monte Carlo; scipy.optimize for SVI; polars for option chains |
| Types and schemas | Pydantic v2, mypy --strict | Typed RFQs and API payloads; catches errors before runtime |
| Testing | pytest, Hypothesis, pytest-benchmark | Unit tests, property tests for pricing laws, tracked speed |
| API | FastAPI | OpenAPI docs for free; async jobs |
| Storage | PostgreSQL + pgvector, Parquet | Snapshots, audit log and embeddings in one database; raw chains as files |
| LLMs | AWS Bedrock (Claude), plus one local model via Ollama | Your existing Bedrock work; local model shows the "data can't leave" trade-off |
| Agents | Own harness + official MCP Python SDK | The JD names harnesses and MCP; your own loop shows you understand it |
| Retrieval | rank-bm25, an embedding model, a cross-encoder reranker | Hybrid search catches exact identifiers that embeddings blur |
| Observability | Langfuse, OpenTelemetry, JSON logs | Every LLM call and tool call traced with cost and latency |
| Delivery | Docker Compose, Terraform (AWS ECS + RDS), GitHub Actions | Runs locally in one command; deploys like a real service |
| Data | yfinance option chains, FRED rates, issuer KIDs and final terms | Free and public; yfinance is unofficial and delayed, fine for a portfolio project |

## Component specs

Each component lists what to build, the design choices that matter, the tests that say it's done, and what an interviewer will ask you about it.

### 1. Market data and snapshots

**Build:** a `MarketSnapshot` holding spot, a rate curve, dividend yield and a cleaned option chain, plus loaders for yfinance chains and FRED rates.

- Snapshots are immutable and content-hashed: `snapshot_id` is a hash of the data, so the same id always means the same inputs.
- Store raw chains as Parquet and snapshot metadata in Postgres.
- Cleaning: drop zero bids, very wide spreads and stale quotes; use mid prices; back out implied vols with Brent's method on Black-Scholes.
- Rate curve: simple bootstrap or interpolation from FRED SOFR and Treasury yields; dividend yield implied from put-call parity or set flat.

**Done when:** re-pricing from a `snapshot_id` gives identical results; loader tests run on a saved fixture with no network.

**Be ready to explain:** why immutable snapshots matter for audit and model validation, and how you cleaned the chain.

### 2. Pricing core

**Build:** instruments, market inputs, Black-Scholes prices and greeks, a Monte Carlo engine, and bump-and-revalue greeks.

- Instruments and market inputs are frozen dataclasses with validation; they hold no pricing logic.
- Black-Scholes-Merton closed form with a dividend yield; analytic delta, gamma, vega, theta and rho, with units written in the docstrings.
- Monte Carlo simulates exact GBM in log space with a seeded `numpy.random.Generator`, so every price is reproducible from its config.
- Antithetic variates pair each path with its mirror; compute standard errors on pair means, since the two paths of a pair are not independent.
- Control variate for Europeans: the discounted terminal spot, whose expectation is known exactly.
- Barriers: step paths daily and keep a running knock flag, so memory is O(paths), not O(paths x steps).
- Autocallable (per unit notional): on each observation date, redeem early with accumulated coupons if the underlying is at or above the autocall level; otherwise at maturity repay notional above the knock-in level, or take the underlying's loss below it.
- Greeks by central bumps with common random numbers: every revaluation reuses the same seed, so the noise cancels.

**Done when:**

- MC prices are within 3 standard errors of Black-Scholes for calls and puts.
- Hull's textbook example matches: S=42, K=40, r=10%, vol=20%, T=0.5 gives call 4.76 and put 0.81.
- Put-call parity holds; analytic greeks match central finite differences.
- Barrier knock-in plus knock-out equals the vanilla price.
- Autocallable edge cases are exact: autocall level 0 pays (1 + coupon) discounted to the first date; never called with knock-in 0 pays the discounted notional.
- Hypothesis properties: call bounded by spot, price rising with vol, call rising and put falling with spot.

**Be ready to explain:** why antithetic errors use pairs, why the control variate cuts variance, discrete versus continuous barrier monitoring, how the autocallable pays.

### 3. Volatility surface (SVI)

**Build:** a raw SVI fit per expiry, arbitrage checks, and interpolation in total variance across expiries.

Raw SVI models total implied variance w = vol² x T as a function of log-moneyness k = log(K/F):

$$
w(k) = a + b\left(\rho\,(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)
$$

- Fit with weighted least squares in scipy.optimize: bounds b ≥ 0, |ρ| < 1, σ > 0, several starting points, weights from vega or bid-ask width.
- Butterfly check: the density function g(k) must stay non-negative on a strike grid.

$$
g(k) = \left(1 - \frac{k\,w'}{2w}\right)^2 - \frac{w'^2}{4}\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{w''}{2}
$$

- Calendar check: total variance must not fall as expiry rises at the same k.
- In M1 the Monte Carlo uses the surface vol at each product's strike and expiry; Dupire local vol for autocallables is a stretch goal.

**Done when:** RMSE under 0.5 vol points per expiry on liquid strikes, zero violations after the fit (or each one flagged), and a plot of fit against market for each expiry.

**Be ready to explain:** why you fit total variance, what butterfly and calendar arbitrage mean, why SVI rather than a spline.

### 4. Dependency graph

**Build:** a small graph where inputs (spot, surface, curve) and derived values (forwards, prices, greeks, book totals) are nodes.

- A node's compute function records which nodes it reads, so edges come from real dependencies.
- `set(node, value)` marks every downstream node dirty; `get(node)` recomputes only if dirty and caches the result.
- Scenario contexts: `with graph.scenario(spot={"SPY": 1.05})` overrides inputs in a child layer, leaves the base untouched, and caches per scenario.
- Single-threaded in M3; note thread safety as a later decision in an ADR.

**Done when:** bumping one underlying recomputes only its positions (assert recompute counts in tests); incremental against full revaluation is benchmarked; scenarios never change base values.

**Be ready to explain:** invalidation versus recompute, what goes in a cache key, and why bank platforms such as SecDB and Athena are built this way.

### 5. Risk engine

**Build:** per-position greeks through graph bumps, ladders by expiry and strike, spot x vol scenario grids, and P&L explain between two snapshots.

$$
\Delta V \approx \delta\,\Delta S + \tfrac{1}{2}\,\Gamma\,\Delta S^2 + \nu\,\Delta\sigma + \Theta\,\Delta t
$$

- Unexplained P&L = actual change in value minus the explained terms, per position and for the book.
- Add vanna (spot x vol cross term) once the basics work, and measure how much unexplained it removes.
- Test book: about 1,000 synthetic positions, mostly vanillas plus a few barriers and autocallables.

**Done when:** unexplained is under 5% on normal days for the vanilla book, and large-move days are logged with the reason.

**Be ready to explain:** why unexplained grows on big moves and what each cross term captures.

### 6. Services and lineage

**Build:** a FastAPI service over the core and risk engine.

- Endpoints: `POST /price`, `POST /risk`, `POST /scenario`, `GET /pnl-explain`, `GET /snapshots/{id}`, `POST /quotes`, `POST /quotes/{id}/approve`.
- Every response carries lineage: `snapshot_id`, engine version, git SHA, Monte Carlo seed and path count, timestamp.
- Batch work runs as background jobs; POST requests take an idempotency key.

**Done when:** any past response can be reproduced from its lineage alone, and OpenAPI docs are generated.

**Be ready to explain:** why lineage matters to model risk and audit, and how idempotency prevents double quotes.

### 7. AI layer

**Build:** an MCP server, your own agent harness, the RFQ copilot, term-sheet extraction, hybrid RAG and guardrails.

- **MCP server** (official Python SDK) exposing tools over the API: `price_instrument`, `compute_greeks`, `run_scenario`, `pnl_explain`, `get_snapshot`, `search_termsheets`, `create_quote`. Every tool validates input with Pydantic and checks the caller's desk role.
- **Agent harness:** plan, call a tool, observe, verify, answer. Step cap (about 8), timeouts, retries with backoff, a typed final-answer schema, and a full trace in Langfuse.
- **RFQ copilot:** free text to a Pydantic `RFQ` (underlying, product, notional, tenor, barrier and autocall levels, coupon, observation frequency). Business rules check it (knock-in below autocall level, positive tenor), missing fields trigger a follow-up question, then it prices through tools and creates a pending quote. Nothing is sent without a human approval record.
- **Term-sheet extraction:** public KIDs and final terms PDFs to text and tables (pdfplumber), then to a typed instrument, with low-confidence fields flagged for review.
- **Hybrid RAG** over term sheets: chunk by section, BM25 plus dense embeddings merged by reciprocal rank fusion, a cross-encoder reranker, citations on every answer, and a metadata filter by role.
- **Guardrails:** a number check (every number in a final answer must appear in a tool output or a retrieved chunk, or the answer is blocked); PII masked before model calls; retrieved text screened for injected instructions; a per-role tool allow-list.
- **Model routing:** a small model for parsing and extraction, a larger one for reasoning, and a local Ollama model as a comparison.

**Done when:** the eval thresholds in component 8 pass, and every quote in the database has an approval record.

**Be ready to explain:** why the model never computes numbers, how you detect prompt injection, and how you decided the routing.

### 8. Evals

**Build:** versioned eval datasets and a runner that gates CI.

| Dataset | Size | Measures |
|---|---|---|
| `rfq_cases.jsonl` | 200 synthetic RFQs, varied phrasing, some with missing fields | Field accuracy, correct follow-up questions |
| `termsheets/` | 50 to 100 hand-labelled public KIDs | Field accuracy per field |
| `rag_questions.jsonl` | 100 questions with gold chunk ids | recall@10, MRR |
| `answers_labelled.jsonl` | 50 answers you label by hand | LLM-judge agreement with you, faithfulness |
| `agent_tasks.jsonl` | 50 tasks with expected tool calls and numbers | Task success, numeric grounding (must be 100%) |
| `injection_cases/` | 30 documents with planted instructions | Successful injections (must be 0) |

- A fast subset runs on every pull request and blocks the merge on a drop above 2%; the full suite runs nightly.
- Results go to `evals/results/*.json` and a table in the README, with p95 latency and cost per query from Langfuse.

**Be ready to explain:** how you built the labelled sets, and why you checked the LLM judge against your own labels.

### 9. Observability, delivery and quality

- Langfuse traces every model and tool call; OpenTelemetry covers the API; logs are JSON with request ids.
- Docker Compose brings up the API, Postgres with pgvector and Langfuse in one command.
- Terraform deploys to AWS: ECS Fargate for the API, RDS for Postgres, Secrets Manager for keys.
- GitHub Actions runs ruff, mypy --strict, unit and property tests, the fast eval suite, and benchmarks with a regression threshold.
- Write one ADR per major decision: ADRs themselves, a pure core, Monte Carlo design, snapshots and lineage, the dependency graph, SVI, your own harness against a framework, the number-check guardrail, model routing.

## Build plan

Build the pricing core first, then the AI copilot the team lead asked about, then data, volatility and risk, then polish. Each milestone ends with something you can demo and numbers you can put on the CV.

| Milestone | Weeks | Build (component numbers above) | Done when | CV numbers it unlocks |
|---|---|---|---|---|
| M1 Pricing core | 1 to 2 | 2, plus the CI skeleton and benchmarks | Every pricing test passes in CI; benchmark results saved as JSON | MC error against closed form, MC speed, loop against vectorised speedup |
| M2 AI copilot | 3 to 5 | 6 (price, quote, approve endpoints), 7 (MCP, harness, RFQ copilot, guardrails), 8 (RFQ and agent evals), Langfuse | Eval table in the README; every quote has an approval record | RFQ accuracy, numeric grounding, task success, injection rate, p95 latency, cost per query |
| M3 Data, vol and risk | 6 to 7 | 1, 3, 4, 5, plus term-sheet extraction and RAG | Recompute counts tested; SVI fits plotted; P&L explain runs on two real snapshots | SVI RMSE, incremental speedup, unexplained %, recall@10, extraction accuracy |
| M4 Polish | 8 | 9: Terraform deploy, ADRs, README with diagram and results, demo GIF | A stranger can clone and run it from the README | None new |
| Stretch | Later | Kafka live risk, Dupire local vol, Ray batches, local model comparison | Optional | Optional |

If the UBS round comes before week 8, demo M1 plus the RFQ copilot from M2.

Target values for every metric are in *Document* <!-- TODO: link missing in the source PDF -->. Put a number on your CV only once you have measured it.

## Building it with Claude Code

Export this doc as Markdown into `docs/SPEC.md`, add the CLAUDE.md below, then work through the prompts one at a time.

### Workflow for every task

1. Plan first: ask Claude Code for a plan and agree it before any code.
2. Tests first, then the implementation until they pass.
3. Run ruff, mypy and pytest locally.
4. Read every diff yourself before committing; you will be asked about any line.
5. Small commits, one feature per branch and pull request, and CI green before merging.

### Keep it yours

- Write the ADRs in your own words; Claude Code can draft, you edit.
- After each component, explain it out loud for 10 minutes without notes. If you can't, don't move on.
- Commit steadily across the weeks; a real history is evidence, one giant commit is a red flag.
- Only put a metric on the CV if a script in the repo produced it.

### CLAUDE.md to paste into the repo root

See [`/CLAUDE.md`](../CLAUDE.md).

### Starter prompts

**M1 Pricing core**

1. "Read docs/SPEC.md, section Pricing core. Propose the module layout and the tests you'll write first. Don't write code yet."
2. "Write tests for black_scholes.py first: Hull's example (S=42, K=40, r=0.10, vol=0.20, T=0.5 gives call 4.76 and put 0.81), put-call parity, and greeks against central finite differences. Then implement until they pass."
3. "Implement monte_carlo.py per the spec: seeded generator, antithetic variates with errors on pair means, a control variate for Europeans, a barrier step loop with a knock flag, and the autocallable on its observation grid. Tests first: within 3 standard errors of Black-Scholes, barrier in-out parity, autocallable edge cases."
4. "Add Hypothesis property tests for the pricing laws in the spec, then a pytest-benchmark suite that saves MC error and runtime at 1M paths to benchmarks/results.json."

**M2 AI copilot**

5. "Define the RFQ Pydantic model and write 20 labelled RFQs in evals/data/rfq_cases.jsonl, some with missing fields. Write the field-accuracy eval before the parser."
6. "Build the RFQ parser with structured output on Bedrock, the validation rules, and follow-up questions for missing fields. Run the eval and report accuracy per field."
7. "Build the FastAPI price, quote and approve endpoints with lineage on every response, then an MCP server exposing price_instrument, compute_greeks and create_quote."
8. "Build the agent harness: at most 8 steps, validated tool calls, retries, a typed final answer and Langfuse tracing. Add the number-check guardrail with a test proving a made-up number is blocked."

**M3 Data, vol and risk**

9. "Add a snapshot loader using one saved SPY chain as a fixture, then the SVI fit per expiry with butterfly and calendar checks. Plot each fit against the market."
10. "Implement the dependency graph with dirty-flag invalidation and scenario contexts. Assert recompute counts in tests and benchmark incremental against full revaluation."
11. "Implement P&L explain between two snapshots and report unexplained P&L per position and for the book."

**M4 Polish**

12. "Write the Terraform for ECS and RDS, fill the README results table from the evals and benchmarks, and draft any missing ADRs for me to edit."

## Sources

- [AI at BNP Paribas: Chat2Trade turns client RFQs from chat and email into structured requests](https://group.bnpparibas/en/news/ai-at-bnp-paribas-1-producing-reliable-quotes-faster-to-help-global-markets-clients-in-trading): a real precedent for the RFQ copilot.
- [FIA, Nov 2026: Generative AI moves from experiment to execution in capital markets](https://www.fia.org/marketvoice/articles/generative-ai-moves-experiment-execution-capital-markets): banks keep agents at recommending actions, and find LLMs unreliable on numbers, which is why DeskGraph's numbers come only from tools.
- [Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol): the protocol behind the tool server.
