# DeskGraph architecture and build plan

This is the blueprint. [SPEC.md](SPEC.md) says *what* each component must do and
when it's done; this document says *how* the pieces fit, *in what order* we build
them, and *what we decided and why*. For quick rules and conventions, see
[WHEN_IN_DOUBT.md](WHEN_IN_DOUBT.md).

Every decision here has a status:

- **Agreed**: confirmed by the project owner; build to it.
- **Proposed**: recommended but not yet confirmed; don't build anything hard to undo on it.
- **Open**: needs a decision before the work that depends on it starts.

Contents: [System context](#1-system-context) ·
[Layers](#2-layers-and-the-dependency-rule) ·
[Modules](#3-module-map) ·
[Domain model](#4-domain-model) ·
[Flows](#5-key-flows) ·
[Storage](#6-storage) ·
[Lineage](#7-lineage-and-reproducibility) ·
[AI layer](#8-ai-layer) ·
[Security](#9-trust-boundaries-and-security) ·
[Testing and CI](#10-testing-and-ci) ·
[Deployment](#11-deployment) ·
[Build plan](#12-build-plan) ·
[Decision log](#decision-log) ·
[Open questions](#open-questions-and-proposals) ·
[Risks](#risks)

---

## 1. System context

```
          People                                  External services
 ┌────────────────────────┐                ┌─────────────────────────────────┐
 │ Sales       (RFQs)     │                │ Yahoo Finance  option chains    │
 │ Structurer  (pricing)  │                │ FRED           rates            │
 │ Trader      (risk)     │──── CLI chat ──│ AWS Bedrock    Claude models    │
 │ Desk head   (P&L,      │     MCP        │ Ollama         local model      │
 │             approvals) │     HTTP API   │ Langfuse       traces           │
 └───────────┬────────────┘                │ Issuer sites   public KIDs      │
             │                             └───────────────┬─────────────────┘
             ▼                                             │
 ┌─────────────────────────────────────────────────────────▼──────────────┐
 │                               DeskGraph                                │
 │  AI layer → Services → Risk engine → Dependency graph → Quant core     │
 │                           ↑                                            │
 │                     Market data (snapshots, Parquet, Postgres)         │
 └────────────────────────────────────────────────────────────────────────┘
```

**Actors and what they may do** (Proposed, see [P-08](#open-questions-and-proposals)):

| Role | Typical request | Can create quotes | Can approve quotes |
|---|---|---|---|
| `sales` | "Price me a 1y autocallable on SPY…" | Yes (pending) | No |
| `structurer` | "What coupon makes this par?" | Yes (pending) | No |
| `trader` | "Vega by expiry if vol drops 5 points?" | No | Yes |
| `desk_head` | "Why did P&L move yesterday?" | No | Yes |
| AI agent | Acts *for* one of the roles above, with that role's tool list | Yes (pending, via tool) | **Never** |

---

## 2. Layers and the dependency rule

Six layers. Each may import only from layers below it (**Agreed**, D-01). The AI
layer doesn't import the lower layers at all: it calls the HTTP API through MCP
tools (**Agreed**, D-05), so validation, role checks and lineage can't be bypassed.

```
 ┌───────────────────────────────────────────────┐
 │ ai/        MCP server, harness, RFQ copilot,  │  calls the API over HTTP only
 │            RAG, guardrails, model routing     │
 ├───────────────────────────────────────────────┤
 │ api/       FastAPI, jobs, idempotency,        │
 │            lineage stamping, audit log        │
 ├───────────────────────────────────────────────┤
 │ risk/      book, ladders, scenario grids,     │
 │            P&L explain                        │
 ├───────────────────────────────────────────────┤
 │ graph/     nodes, dirty flags, scenarios      │
 ├───────────────────────────────────────────────┤
 │ core/      instruments, market inputs, BS,    │  pure: no I/O (enforced by test)
 │            MC, greeks, SVI                    │
 ├───────────────────────────────────────────────┤
 │ data/      loaders, cleaning, snapshot store  │  does I/O; builds core inputs
 └───────────────────────────────────────────────┘
```

`data/` sits beside `core/` rather than strictly below it: `data/` produces
`core` types (snapshots, market inputs) and does all the I/O, while `core/`
never imports `data/`.

**Allowed imports** (Proposed to enforce by test, [P-09](#open-questions-and-proposals)):

| Package | May import from `deskgraph.` |
|---|---|
| `core` | `core` only |
| `data` | `core` |
| `graph` | `core` |
| `risk` | `graph`, `core`, `data` |
| `api` | `risk`, `graph`, `core`, `data` |
| `ai` | nothing from the lower layers; HTTP client to `api` and its own schemas |

---

## 3. Module map

| Module | Responsibility | Main public interface | Milestone |
|---|---|---|---|
| `core/market.py` | Market inputs; discounting | `MarketInputs(spot, rate, div_yield, vol)`, `.discount_factor(t)`, `.snapshot_id` | M1 |
| `core/instruments.py` | Product definitions and validation | `OptionType`, `European`, `BarrierType`, `Barrier`, `Monitoring`, `Autocallable` | M1 |
| `core/results.py` | Shared result types | `Greeks(delta, gamma, vega, theta, rho)` | M1 |
| `core/black_scholes.py` | Closed-form prices and greeks | `price(inst, mkt) -> float`, `price_vec(is_call, spot, strike, expiry, rate, div_yield, vol) -> array`, `greeks(inst, mkt) -> Greeks` (vectorised greeks added when the risk engine needs them, M3) | M1 |
| `core/monte_carlo.py` | GBM paths, variance reduction, payoffs | `MCConfig`, `MCResult`, `price(inst, mkt, cfg) -> MCResult` | M1 |
| `core/greeks.py` | Bump-and-revalue with common random numbers | `bump_greeks(inst, mkt, cfg) -> Greeks` | M1 |
| `core/vol/svi.py` | SVI fit per expiry, arbitrage checks, interpolation | `fit_svi(...)`, `butterfly_violations(...)`, `calendar_violations(...)`, `VolSurface` | M3 |
| `data/chains.py` | Load raw captured chains, clean, implied vols | `load_raw_chain(ticker, date)`, `clean_chain(df)` | M3 |
| `data/rates.py` | FRED loaders, rate curve construction | `load_rates(date)`, `build_curve(...)` | M3 |
| `data/snapshots.py` | Build, hash, store and load snapshots | `MarketSnapshot`, `SnapshotStore.save/get` | M3 (minimal hash in M1) |
| `graph/` | Dependency graph and scenario contexts | `Graph.node`, `.set`, `.get`, `.scenario(...)` | M3 |
| `risk/book.py` | Positions and books | `Position`, `Book` | M3 |
| `risk/ladders.py`, `risk/scenarios.py` | Greeks by expiry/strike; spot × vol grids | `ladder(book, by=...)`, `spot_vol_grid(...)` | M3 |
| `risk/pnl_explain.py` | P&L explain between two snapshots | `explain(book, snap_t0, snap_t1) -> PnLExplain` | M3 |
| `api/app.py` | FastAPI app and routes | `/price`, `/risk`, `/scenario`, `/pnl-explain`, `/snapshots/{id}`, `/quotes`, `/quotes/{id}/approve` | M2 (price, quotes), M3 (rest) |
| `api/lineage.py` | Lineage model and stamping | `Lineage` | M2 |
| `api/quotes.py` | Quote lifecycle, idempotency, audit | `QuoteService` | M2 |
| `ai/mcp_server.py` | MCP tools over the API | 7 tools (see §8) | M2 (3 tools), M3 (rest) |
| `ai/harness.py` | Agent loop with limits and tracing | `run_agent(task, role) -> FinalAnswer` | M2 |
| `ai/rfq.py` | RFQ schema, parser, rules, follow-ups | `RFQ`, `parse_rfq(text) -> RFQ \| FollowUp` | M2 |
| `ai/guardrails.py` | Number check, PII mask, injection screen, allow-list | `check_numbers(...)`, `mask_pii(...)`, `screen_chunk(...)` | M2 |
| `ai/chat.py` | CLI chat for a chosen role | `python -m deskgraph.ai.chat --role sales` | M2 (Proposed, P-07) |
| `ai/termsheets.py` | PDF to typed instrument with confidence | `extract(pdf_path) -> Extraction` | M3 |
| `ai/rag/` | Chunking, BM25 + dense, RRF, reranker | `search(query, role) -> list[Chunk]` | M3 |
| `scripts/capture_chain.py` | Daily raw chain capture (standalone) | `uv run scripts/capture_chain.py [TICKERS]` | Built |
| `evals/` | Datasets and runner | `python -m evals.run --suite fast\|full` | M2 |
| `benchmarks/` | Speed and accuracy benchmarks | `pytest benchmarks --benchmark-only` | M1 |

---

## 4. Domain model

All core types are frozen dataclasses with validation in `__post_init__`
(**Agreed**). API and AI types are Pydantic v2 models that convert to and from
core types at the API boundary.

### Core (M1)

```text
MarketInputs      spot > 0, rate, div_yield, vol > 0          (all finite)
                  discount_factor(t) = exp(-rate · t)
                  snapshot_id = "sha256:" + hash(canonical fields)

European          option_type: CALL | PUT, strike > 0, expiry > 0
Barrier           European fields + barrier > 0,
                  barrier_type: DOWN_IN | DOWN_OUT | UP_IN | UP_OUT,
                  monitoring: daily (1/252 steps)
Autocallable      expiry > 0
                  observation_times: strictly increasing, > 0, last == expiry
                                     (within 1e-9, then set to expiry exactly)
                  autocall_level >= 0    (fraction of initial spot)
                  coupon >= 0            (per observation period, memory)
                  0 <= knock_in_level <= autocall_level
                  knock_in_monitoring: DAILY | AT_MATURITY

Core validation checks the payoff is well defined; stricter desk rules
(knock-in strictly below autocall, sensible tenors) live in RFQ validation
(P-18). All numbers are coerced to finite floats, -0.0 normalised to 0.0;
bool and non-numbers raise TypeError, bad values raise ValueError naming the field.

Greeks            delta, gamma, vega, theta, rho   (units in WHEN_IN_DOUBT)
MCConfig          n_paths > 0 (even if antithetic), seed, antithetic, control_variate
MCResult          price, std_error, n_paths, seed
```

**Autocallable payoff per unit notional** (Agreed, D-10). With observation
times t₁ < … < tₙ = T, spot ratio Xᵢ = S(tᵢ)/S₀ and coupon c per period:

- first i with Xᵢ ≥ autocall level: pay (1 + i·c) at tᵢ, then stop;
- never called: pay 1 at T if not knocked in; pay X(T) at T if knocked in, where
  knocked in means min over daily path < KI (daily) or X(T) < KI (at maturity).

### Later layers

```text
MarketSnapshot (M3)  ticker, as_of, spot, rate curve, div_yield, cleaned chain, surface
                     snapshot_id = content hash; replaces MarketInputs as the pricing input
Position (M3)        instrument, quantity, position_id
Book (M3)            positions
Lineage (M2)         snapshot_id, engine_version, git_sha, method, seed, n_paths,
                     created_at, request_id
RFQ (M2)             underlying, product, notional, tenor, strike/barrier/autocall levels,
                     coupon, observation_frequency, knock_in_monitoring (all optional
                     while parsing; required set depends on product)
Quote (M2)           id, rfq, instrument, price, lineage, status, created_by, idempotency_key
Approval (M2)        quote_id, approver, role, decision, comment, decided_at
AuditEvent (M2)      at, actor, action, entity, payload, request_id, trace_id
```

---

## 5. Key flows

### 5.1 Sales: RFQ to approved quote (M2)

```
Sales ──"Price me a 1y autocallable on SPY, 60% KI, quarterly"──► CLI chat / MCP client
  1. guardrails.mask_pii(text)
  2. small model: parse_rfq(text) → RFQ (typed, structured output)
  3. business rules: KI < autocall level, tenor > 0, required fields present
       └─ missing field? → FollowUp question to Sales, stop here
  4. harness → MCP tool price_instrument(rfq→instrument, snapshot_id)
       └─ API POST /price → core.monte_carlo.price → result + lineage
  5. harness → MCP tool create_quote(instrument, price result)
       └─ API POST /quotes (idempotency key) → Quote{status=PENDING_APPROVAL} + audit event
  6. final answer: guardrails.check_numbers(answer, tool outputs, user text) → pass or block
Trader / desk head ── POST /quotes/{id}/approve (human, not the AI) ──► APPROVED + audit event
```

### 5.2 Trader: "vega by expiry if vol drops 5 points" (M3)

```
1. harness (large model) plans: run_scenario(vol_shift = -0.05), then ladder by expiry
2. MCP run_scenario → API POST /scenario
     └─ risk.scenarios: with graph.scenario(vol=-0.05): ladder(book, by="expiry", greek="vega")
3. tool output: table of vega by expiry + lineage
4. answer quotes the table; number check passes only if every number is in the tool output
```

### 5.3 Desk head: "why did P&L move yesterday?" (M3)

```
1. MCP pnl_explain(date) → API GET /pnl-explain?t0=…&t1=…
2. risk.pnl_explain: value book on snapshot t0 and t1; greeks at t0;
   explained = δΔS + ½ΓΔS² + νΔσ + ΘΔt (+ vanna ΔSΔσ); unexplained = actual − explained
3. answer: per-term contributions, unexplained amount, large-move flag if applicable
```

### 5.4 Daily capture (running now)

```
Windows Task Scheduler, weekdays 20:45 UK (15:45 New York)
  → uv run scripts/capture_chain.py
  → data/raw/chains/SPY/<trade_date>/chain.parquet + meta.json   (skip if already there)
M3: data/chains.py cleans these → MarketSnapshot → SnapshotStore
```

### 5.5 Quote lifecycle (Proposed, P-10)

```
             create_quote (AI or human)
                     │
                     ▼
            PENDING_APPROVAL ──── end of day ────► EXPIRED
              │            │
   approve    │            │  reject
 (human only) ▼            ▼
          APPROVED      REJECTED
```

Every transition writes an audit event. A repeated `POST /quotes` with the same
idempotency key returns the original quote, never a second one.

---

## 6. Storage

**Files** (raw data, git-ignored, backed up separately):

```
data/raw/chains/<TICKER>/<YYYY-MM-DD>/chain.parquet   raw Yahoo chain, all expiries
data/raw/chains/<TICKER>/<YYYY-MM-DD>/meta.json       spot, timestamps, counts, sha256
data/raw/chains/capture.log                           capture history
data/snapshots/<snapshot_id>/chain_clean.parquet      cleaned chain + implied vols (M3)
tests/fixtures/                                       small committed copies for tests
```

**Postgres with pgvector** (M2 onwards):

| Table | Key columns | Notes |
|---|---|---|
| `snapshots` | `snapshot_id` PK, ticker, as_of, spot, rate_curve jsonb, div_yield, chain_path, chain_sha256 | Insert-only |
| `quotes` | id PK, rfq jsonb, instrument jsonb, price, lineage jsonb, status, created_by, created_at, `idempotency_key` unique | |
| `approvals` | id PK, `quote_id` unique FK, approver, role, decision, comment, decided_at | One decision per quote |
| `audit_log` | id PK, at, actor, action, entity, payload jsonb, request_id, trace_id | Append-only |
| `termsheet_chunks` | id PK, doc_id, section, text, embedding vector, metadata jsonb | RAG (M3) |

Migrations: Proposed to use plain SQL files applied in order, to avoid another
dependency until it's clearly needed.

---

## 7. Lineage and reproducibility

Every API response includes:

```json
"lineage": {
  "snapshot_id": "sha256:…",
  "engine_version": "0.1.0",
  "git_sha": "…",
  "method": "black_scholes | monte_carlo",
  "seed": 42,
  "n_paths": 200000,
  "created_at": "2026-12-10T15:45:00Z",
  "request_id": "…"
}
```

**Reproduction rule**: checking out `git_sha`, loading `snapshot_id` and re-running
`method` with `seed` and `n_paths` gives the identical number. This is tested
(SPEC component 6). It works because core is pure, randomness is seeded and passed
in, and snapshots are immutable.

`engine_version` comes from package metadata; `git_sha` is set at build time
(`GIT_SHA` in the container) or read from git in development. Core never reads
either.

---

## 8. AI layer

### 8.1 MCP tools and role allow-list (Proposed, P-08)

| Tool | API call | sales | structurer | trader | desk_head |
|---|---|:-:|:-:|:-:|:-:|
| `price_instrument` | `POST /price` | ✓ | ✓ | ✓ | ✓ |
| `compute_greeks` | `POST /risk` | | ✓ | ✓ | ✓ |
| `run_scenario` | `POST /scenario` | | | ✓ | ✓ |
| `pnl_explain` | `GET /pnl-explain` | | | ✓ | ✓ |
| `get_snapshot` | `GET /snapshots/{id}` | ✓ | ✓ | ✓ | ✓ |
| `search_termsheets` | RAG search | ✓ | ✓ | ✓ | ✓ |
| `create_quote` | `POST /quotes` | ✓ | ✓ | | |
| *(approve)* | `POST /quotes/{id}/approve` | **not a tool: humans only, through the API** | | | |

Every tool validates its input with Pydantic and checks the caller's role before
calling the API.

### 8.2 Harness loop

```
plan ─► call tool ─► observe ─► verify ─► answer
  ▲                                │
  └──────── next step ◄────────────┘
```

| Limit | Value | Status |
|---|---|---|
| Max steps | 8 | Agreed (SPEC) |
| Per-tool timeout | 30 s | Proposed |
| Whole-task timeout | 120 s | Proposed |
| Retries | 3, exponential backoff, transient errors only | Proposed |
| Final answer | Typed schema; free text alone isn't accepted | Agreed (SPEC) |
| Tracing | Every model and tool call in Langfuse, with cost and latency | Agreed (SPEC) |

Validation errors from tools go back to the model as observations, so it can
correct the call; they count towards the step cap.

### 8.3 Guardrail pipeline

```
user text ──► mask PII ──► model
retrieved chunk ──► injection screen ──► model context (flagged chunks dropped and logged)
tool call ──► role allow-list ──► Pydantic validation ──► API
final answer ──► number check ──► typed schema ──► user
                     │
                     └── fails ► answer blocked, user told the answer couldn't be verified
```

**Number check sources** (Proposed, P-04): tool outputs, retrieved chunks, and the
user's own request (so "60% knock-in" isn't blocked), with matching that tolerates
formatting and stated rounding.

### 8.4 Model routing (Agreed in principle, SPEC; model IDs Open)

| Job | Model | Why |
|---|---|---|
| RFQ parsing, term-sheet extraction | Small Bedrock model | Structured output, high volume, low cost |
| Agent reasoning and answers | Larger Bedrock model | Multi-step planning over tools |
| Comparison and offline runs | Local model via Ollama | Shows the trade-off when data can't leave the machine |

Model IDs are configuration (`DESKGRAPH_MODEL_SMALL`, `DESKGRAPH_MODEL_LARGE`,
`DESKGRAPH_MODEL_LOCAL`), chosen at the start of M2.

---

## 9. Trust boundaries and security

- **Identity is demo-level.** The API reads the role from an `X-Desk-Role` header;
  the CLI and MCP server take it from `--role` or `DESKGRAPH_ROLE`. That's enough
  to show role-based tool lists, not real authentication (out of scope).
- **The AI can't approve.** No approve tool exists, and `/quotes/{id}/approve`
  rejects any approver id reserved for the agent.
- **Untrusted text**: user messages, retrieved term-sheet text and PDF content are
  data, never instructions. Retrieved chunks are screened before entering context;
  the injection eval must show zero successful injections.
- **PII** is masked before any model call, even though only public data is used,
  so the path is proven.
- **Secrets** come from environment variables locally (`.env`, git-ignored) and
  AWS Secrets Manager in the cloud. Never in code or the repo.

---

## 10. Testing and CI

| Layer | Tests | Where |
|---|---|---|
| core | Closed forms (Hull), parity identities, greeks vs finite differences, MC within 3 SE, barrier in-out parity, autocallable edge cases, Hypothesis properties, purity check | `tests/core/`, `tests/test_core_purity.py` |
| data | Loaders and cleaning on saved fixtures, no network; snapshot hash stability | `tests/data/` |
| graph | Recompute counts, scenario isolation | `tests/graph/` |
| risk | Ladder sums, P&L explain on known moves, unexplained bounds | `tests/risk/` |
| api | Lineage present and reproducible, idempotency, role checks, approval rules | `tests/api/` |
| ai | Schema validation, number check blocks made-up numbers, allow-list, harness limits (with a fake model) | `tests/ai/` |
| AI quality | RFQ accuracy, grounding, task success, injection rate, RAG recall | `evals/` |
| speed | MC throughput, loop vs vectorised, incremental vs full revaluation | `benchmarks/` |

**CI pipeline** (`.github/workflows/ci.yml`):

| Stage | Blocks merge | Status |
|---|---|---|
| `uv sync --locked` | Yes | Built |
| ruff lint + format check | Yes | Built |
| mypy strict | Yes | Built |
| pytest | Yes | Built |
| benchmarks | No at first (noisy runners), threshold later | M1 step 6 |
| fast eval suite | Yes, with the threshold rule in P-02 | M2 |
| full eval suite | Nightly, reports only | M2 |

---

## 11. Deployment

**Local** (M2): `docker compose up` starts the API and Postgres with pgvector.
Langfuse runs as its cloud free tier or a self-hosted profile (Proposed, P-06).

**AWS** (M4): Terraform in `infra/terraform/`:

```
GitHub Actions ─► ECR image ─► ECS Fargate (API) ─► RDS Postgres (+pgvector)
                                     │
                                     └─► Secrets Manager (keys)
```

Proposed (P-11): deploy for the demo and record it, then `terraform destroy`, since
Fargate, RDS and NAT cost money every day they run.

---

## 12. Build plan

Eight weeks from 2026-11-26. Each milestone ends in a demo and measured results.

| Milestone | Dates | Components | Demo at the end |
|---|---|---|---|
| **M1 Pricing core** | 2026-11-26 → 2026-12-09 | 2, CI, benchmarks | Price all three products; tests and benchmarks in CI |
| **M2 AI copilot** | 2026-12-10 → 2026-12-30 | 6 (price, quotes, approve), 7 (MCP, harness, RFQ, guardrails), 8 (RFQ and agent evals), Langfuse | RFQ in chat → pending quote → human approval; eval table |
| **M3 Data, vol and risk** | 2026-12-31 → 2027-01-13 | 1, 3, 4, 5, term-sheet extraction, RAG | SVI fits plotted; scenario answers; P&L explain on captured days |
| **M4 Polish** | 2027-01-14 → 2027-01-20 | 9: Terraform, ADRs, README results, demo GIF | Clone and run from the README |

**Minimum end-to-end demo if time runs short**: M1 plus the RFQ copilot from M2.

### M1 steps

| # | Step | Branch | Status |
|---|---|---|---|
| 1 | Project setup, purity check, CI skeleton | `m1/project-setup` | **Done** 2026-11-27 |
| 2 | `market.py`, `instruments.py` (tests first) | `m1/market-instruments` | **Done** 2026-11-28 |
| 3 | `black_scholes.py`: Hull, parity, analytic vs FD greeks | `m1/black-scholes` | **Done** 2026-11-28 |
| 4 | `monte_carlo.py`: antithetic, control variate, barrier, autocallable | `m1/monte-carlo` | Next |
| 5 | `greeks.py`: bumps with common random numbers | `m1/bump-greeks` | |
| 6 | Hypothesis properties, benchmarks to `benchmarks/results.json`, CI benchmark job | `m1/properties-benchmarks` | |

### M2 steps (planned)

1. RFQ Pydantic model, 20 labelled RFQs, field-accuracy eval (before the parser).
2. RFQ parser with structured output on Bedrock, business rules, follow-ups.
3. FastAPI `POST /price`, `POST /quotes`, `POST /quotes/{id}/approve` with lineage, idempotency, audit log; Postgres via Compose.
4. MCP server with `price_instrument`, `compute_greeks`, `create_quote`.
5. Harness with limits, typed final answer, Langfuse; number-check guardrail with a test that blocks a made-up number.
6. CLI chat; agent-task and injection evals; grow RFQ set towards 200; eval table in README.

### M3 steps (planned)

1. Snapshot loader on a captured SPY day as a fixture; cleaning and implied vols.
2. SVI fit per expiry with butterfly and calendar checks; fit plots.
3. Dependency graph with dirty flags and scenario contexts; recompute-count tests; benchmark.
4. Synthetic book (~1,000 positions); ladders and spot × vol grids.
5. P&L explain across captured days; vanna; unexplained report.
6. Term-sheet extraction and hybrid RAG with citations; remaining MCP tools and endpoints.

### M4 steps (planned)

1. Docker Compose polish, Terraform for ECS + RDS, deploy, record demo, destroy.
2. ADRs finished (written by the project owner from drafts).
3. README results table filled from scripts; demo GIF.

### Dependencies to watch

- **The daily capture must run every trading day.** M3's P&L explain uses days
  that can only be captured on the day.
- M2 lineage needs `MarketInputs.snapshot_id` from M1 step 2.
- RAG and extraction need public KIDs collected and labelled; start collecting
  during M2 so labelling isn't squeezed into M3.

---

## Decision log

Newest decisions at the bottom. Each big one also gets an ADR in `docs/adr/`.

| ID | Decision | Why | Rejected alternatives | Status, date |
|---|---|---|---|---|
| D-01 | Six layers, imports only downwards | Maths testable alone; AI can't bypass checks | Flat package; AI calling core directly | Agreed, 2026-11-26 (SPEC) |
| D-02 | `core/` is pure, enforced by an AST test | Purity is easy to break by accident; a test catches it in CI | Code review only; import-linter (extra dependency) | Agreed, built 2026-11-27 |
| D-03 | LLM never produces numbers; number-check guardrail | LLMs are unreliable with numbers; every figure must trace to the pricer | Trusting the model; prompt instructions only | Agreed, 2026-11-26 (SPEC) |
| D-04 | Human approval for every quote; no approve tool for the AI | Nothing leaves the desk without a person signing off | Auto-approve under a size limit | Agreed, 2026-11-26 (SPEC) |
| D-05 | MCP tools call the HTTP API, not lower layers | One path for validation, roles, lineage and audit | MCP tools importing core directly | Agreed, 2026-11-26 (SPEC) |
| D-06 | Own agent harness plus the official MCP SDK | Control flow, limits and tracing stay explicit and testable | LangChain, LlamaIndex, other agent frameworks | Agreed, 2026-11-26 (SPEC) |
| D-07 | Build order: pricing → copilot → data/vol/risk → polish | Pricing is the base; copilot gives the earliest end-to-end demo | Bottom-up data-first order | Agreed, 2026-11-26 (SPEC) |
| D-08 | Daily raw chain capture from 2026-11-25, standalone script, `data/raw/` git-ignored | Yahoo serves no past chains; history must be recorded daily | Waiting until M3; paid historical data | Agreed, built 2026-11-26 |
| D-09 | Capture at 15:45 New York (20:45 UK), spot read with the chain | Live quotes before the close; spot and chain consistent in time | After-close capture (stale or zero quotes) | Agreed, 2026-11-26 |
| D-10 | Autocallable terms: memory coupons paid on call only; last observation is maturity; KI monitoring daily or at maturity; levels as fractions of initial spot | Removes ambiguity in the SPEC payoff; both KI styles exist in the market | Phoenix coupons; single KI style | Agreed, 2026-11-27 |
| D-11 | M1 uses a flat-vol `MarketInputs` with a content-hash `snapshot_id`; full `MarketSnapshot` in M3 | M2 lineage needs a `snapshot_id` before snapshots exist | Waiting for M3 snapshots; no id in M2 | Agreed, 2026-11-27 |
| D-12 | Year fractions in M1; calendars and day counts later | Keeps M1 on the maths | Business-day calendars from day one | Agreed, 2026-11-27 |
| D-13 | Python 3.12 pinned (`.python-version`); uv with `uv_build`; src layout | SPEC target; CI matches local | Python 3.13 (installed locally); setuptools | Agreed, 2026-11-27 |
| D-14 | Runtime deps numpy, scipy; dev deps pytest, hypothesis, pytest-benchmark, mypy, ruff | Minimum needed for M1 | Adding pydantic and polars now | Agreed, 2026-11-27 |
| D-15 | Ruff at 100 columns; pytest turns warnings into errors | Readable formulas; numerical warnings in pricing are bugs | 88 columns; ignoring warnings | Agreed, 2026-11-27 |
| D-16 | Benchmarks in CI don't block merges at first | Shared runners give noisy timings | Hard threshold from day one | Agreed, 2026-11-27 |
| D-17 | README written as the finished system, with a status table and results that stay empty until measured | One stable front page; no unmeasured numbers | Growing the README piece by piece | Agreed, 2026-11-27 |
| D-18 | `scipy-stubs` as a dev dependency for typed scipy calls (was P-01) | Keeps `mypy --strict` meaningful in the maths code; dev-only, version matches scipy | Telling mypy to ignore scipy (leaves every scipy result untyped) | Agreed, 2026-11-28 |

---

## Open questions and proposals

| ID | Question | Recommendation | Decide by |
|---|---|---|---|
| P-02 | Eval gate: "block on a drop above 2%" is finer than a small fast suite can measure, and LLM output varies run to run | Threshold in number of cases (e.g. block on 2+ newly failing cases); replay recorded model responses on PRs; live run nightly | M2 step 1 |
| P-03 | "Unexplained under 5%" is unstable when total P&L is near zero | Measure unexplained against gross P&L (sum of absolute explained terms), and report the absolute amount too | M3 step 5 |
| P-04 | Number check would block numbers the user typed | Allow tool outputs, retrieved chunks and the user's own request; tolerate formatting and stated rounding | M2 step 5 |
| P-05 | Bump gamma is noisy near barriers for barrier and autocallable, even with common random numbers | Report delta and vega for those products; gamma reported with its standard error and flagged | M1 step 5 |
| P-06 | Self-hosted Langfuse v3 needs ClickHouse, Redis and blob storage | Use Langfuse Cloud free tier; keep a self-host profile optional | M2 step 5 |
| P-07 | Copilot entry point | CLI chat (`python -m deskgraph.ai.chat --role …`) plus MCP for desktop clients; no web UI | M2 step 2 |
| P-08 | Roles and tool allow-list | As in §1 and §8.1 | M2 step 4 |
| P-09 | Enforce the layer import rule | AST test like the purity check, listing allowed imports per package | M2 (before `api/`) |
| P-10 | Quote lifecycle | States as in §5.5; expiry at end of day | M2 step 3 |
| P-11 | AWS cost | Deploy for the demo, record it, `terraform destroy` | M4 step 1 |
| P-12 | Labelling load (50–100 KIDs, 50 answers, 200 RFQs, 100 RAG questions) | Start with 30 KIDs and grow if time allows; build RFQ set in steps of 20 | M2 step 1 |
| P-13 | Targets for metrics without one in the SPEC (RFQ accuracy, task success, recall@10, extraction accuracy, p95 latency, cost per query) | Set each target before its first measurement and record it here | Before each first measurement |
| P-14 | Bedrock model IDs for small and large models | Choose at the start of M2 from what's available in the account's region | M2 step 2 |
| P-15 | License | Choose before making the repo public (MIT is a common default for open-source code) | Before going public |
| P-16 | GitHub remote | Create the repo and push so CI runs | Now |
| P-17 | Database migrations | Plain ordered SQL files, no migration library until needed | M2 step 3 |
| P-18 | Where the rule "knock-in below autocall level" lives | Core allows `knock_in_level <= autocall_level` and `autocall_level >= 0`, because the SPEC's edge-case tests (autocall level 0, knock-in 0) must be priceable; RFQ validation enforces the strict desk rule. Built this way in M1 step 2; confirm | Now |
| P-19 | Seasoned autocallables: notes priced after inception need a fixed initial spot and observation times that move as time passes | Add `initial_spot` (optional, defaults to spot at pricing) and an aging method when theta and P&L explain need it | M1 step 5 (theta) |

When a proposal is confirmed, move it to the decision log with the date.

---

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Yahoo changes or blocks the unofficial endpoint | Daily capture stops; M3 data gaps | Capture log checked weekly; failures exit non-zero; fixtures already saved; FRED for rates |
| Capture misses days (PC off) | Fewer P&L explain days | Task runs when available; any captured pair of consecutive days is enough for the demo |
| Scope too big for eight weeks | Late or shallow milestones | Build order puts the end-to-end demo first; stretch items stay parked; P-12 trims labelling |
| LLM output varies run to run | Flaky evals, noisy CI | Recorded responses on PRs; thresholds in cases; live nightly run |
| MC noise in greeks for exotics | Unstable risk numbers | Common random numbers; standard errors reported; P-05 |
| AWS bills | Unexpected cost | Smallest instance sizes; destroy after demo; budget alarm |
| Docs drift from the code | Confusion, wrong direction | Done checklist requires fixing docs in the same change; decision log is the tie-breaker |
