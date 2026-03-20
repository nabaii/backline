# Backline LLM Architecture

## System Overview

Backline is a football betting analytics platform. The LLM acts as a **quantitative analyst** that combines structured historical data from the kitchen with live web intelligence to produce Bayesian-informed market evaluations.

Two independent pipelines use LLMs:

| Pipeline | Endpoint | Purpose |
|---|---|---|
| **Chat Orchestrator** | `POST /api/chat/stream` | Tool-augmented Bayesian betting analysis |
| **Bet Slip Analyzer** | `POST /api/betslip/analyze` | OCR a bet slip image, then analyze each selection |

A legacy **RAG Pipeline** (`/api/rag/stream`, `/api/rag/query`) also exists, using a numpy-backed vector store with OpenAI embeddings.

---

## Models

| Model | Provider | Role | Used In |
|---|---|---|---|
| `gemini-1.5-pro` | Google (via OpenAI-compat endpoint) | Primary reasoning — tool calling, Bayesian analysis | `ChatOrchestrator.stream` |
| `gemini-1.5-flash` | Google (via OpenAI-compat endpoint) | Lightweight classification | `classify_query` |
| `gpt-4o` | OpenAI | Bet slip OCR (vision), bet slip explanation | `BetSlipAnalyzer.extract_bets`, `BetSlipAnalyzer.analyze_bet` |
| `text-embedding-3-small` | OpenAI | Document embeddings for the vector store | `MatchVectorStore._embed` |

Pipeline 1 (Chat Orchestrator) uses Gemini via Google's OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`), keeping the OpenAI SDK for consistent tool-calling and streaming. Pipeline 2 (Bet Slip Analyzer) and the legacy RAG pipeline continue to use OpenAI directly.

**Environment variables:** `GEMINI_API_KEY` (Pipeline 1), `OPENAI_API_KEY` (Pipeline 2, RAG).

---

## Pipeline 1: Chat Orchestrator

**File:** `backend/rag/chat_orchestrator.py`

The orchestrator is a **tool-augmented agent**. Instead of a fixed pipeline where stages run sequentially, the LLM decides what data it needs and calls tools to get it. The LLM then synthesizes everything into a Bayesian analysis.

### System Prompt

The LLM operates as a professional Sports Quantitative Analyst:

```
You are a professional Sports (Football/Soccer) Quantitative Analyst.
Your goal is to evaluate betting markets. When the user asks about a match:

- Use the kitchen tools to find current team form and historical hit rates for the market.
- Use the search tool to find injuries, lineup news, and other relevant information.
- Define a 'Prior' based on season averages, home/away splits, and averages
  against teams with similar opponent xGD (use the kitchen tools for this).
- Identify 'New Evidence' (weather, lineup changes, motivation, quotes from
  coach press conferences).
- Perform a Bayesian update to calculate a 'Posterior Probability'.
- Compare your result to implied market odds.

Always clarify that these are mathematical models, not guaranteed financial outcomes.
```

### Query Classification (Unchanged)

**File:** `backend/rag/query_classifier.py`

A `gpt-4o-mini` call classifies the user message before entering the tool loop:

| Type | Response Path |
|---|---|
| `greeting` / `vague` | Simple LLM response with system prompt, no tools |
| `betting_intent` | Full tool-calling loop |
| `conversational` | Resolved to standalone query, then tool-calling loop |

### Tools

The LLM has access to three tools via OpenAI function calling:

#### `get_team_profile`

Returns a team's xGD strength profile from the kitchen database.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `team_id` | `int` | yes | Team ID |
| `team_name` | `str` | yes | Team name |
| `league_id` | `str` | yes | League identifier |

**Returns:** `{ team_name, team_id, xgd_season_rank, xgd_last_5, league_size }`

The LLM should call this first for both teams to obtain `opponent_xgd_rank` before requesting hit rates.

#### `get_hit_rates`

Returns layered historical hit rates for a specific betting market from the kitchen.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `team_name` | `str` | yes | Team to analyze |
| `team_id` | `int` | yes | Team ID |
| `opponent_name` | `str` | yes | Opponent name |
| `opponent_xgd_rank` | `int` | no | Opponent's xGD rank (from `get_team_profile`) |
| `perspective` | `"home" \| "away"` | yes | Team's venue |
| `league_id` | `str` | yes | League identifier |
| `bet_type` | `str` | yes | Market: `over_under`, `one_x_two`, `double_chance`, `corners`, `btts`, `first_half_ou`, `first_half_1x2`, `win_both_halves`, `win_either_half` |
| `line` | `float` | no | Goal/corner threshold (e.g. 2.5) |
| `outcome_type` | `str` | no | `1`/`X`/`2` for 1X2, `yes`/`no` for BTTS |

**Returns:** `HitRateProfile` with four layers:

| Layer | Description |
|---|---|
| `season_rate` | Hit rate across all matches this season |
| `perspective_rate` | Hit rate for home-only or away-only matches |
| `rank_filtered_rate` | Hit rate vs opponents within ±3 xGD positions |
| `combined_rate` | Perspective + rank-filtered (tightest contextual view) |

Each layer includes `hits`, `misses`, and `rate`.

#### `search_web`

Searches the web for live match intelligence using DuckDuckGo.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | `str` | yes | Search query |

**Returns:** Up to 5 results with `title`, `body`, `href`.

**Dependency:** `pip install duckduckgo-search`

### Tool-Calling Loop

The orchestrator runs a multi-turn loop (max 10 rounds):

```
User question + fixture context
         │
         ▼
┌─────────────────────┐
│  LLM decides what   │◄──────────────────────┐
│  data it needs      │                        │
└────────┬────────────┘                        │
         │                                     │
    tool_calls?                                │
     ╱        ╲                                │
   yes         no                              │
    │           │                               │
    ▼           ▼                               │
Execute      Yield final                       │
tools        answer to                         │
    │        frontend                          │
    │                                          │
    └──── tool results ────────────────────────┘
```

A typical flow for "Is over 2.5 good for Chelsea vs City?":

1. LLM calls `get_team_profile` for Chelsea and City (parallel)
2. LLM calls `get_hit_rates` for Chelsea (home, over_under 2.5) and City (away, over_under 2.5) using the xGD ranks from step 1
3. LLM calls `search_web` for "Chelsea vs Manchester City injuries lineup news"
4. LLM synthesizes: Prior (from hit rates) + New Evidence (from search) → Posterior → comparison to market odds

### Stream Format

Same delimiter-based format as before:

```
{}\n---CHART_DATA---\n<final analysis text>
```

The chart JSON is currently `{}` (empty). The analysis text is the LLM's complete Bayesian evaluation including Prior, New Evidence, Posterior, and market comparison.

For `greeting` and `vague` queries, the response is a simple streamed reply with no tools.

---

## Pipeline 2: Bet Slip Analyzer

**File:** `backend/rag/betslip_analyzer.py`

Processes uploaded bet slip images through OCR → structured extraction → per-bet workspace analysis.

### Step 1 — Vision OCR

A `gpt-4o` vision call receives the base64-encoded image and extracts structured bets:

```json
[
  {
    "home_team": "Manchester City",
    "away_team": "Arsenal",
    "bet_type": "over_under",
    "line": 2.5,
    "outcome_type": null,
    "selection_label": "Over 2.5 Goals"
  }
]
```

### Step 2 — Per-Bet Analysis

Each extracted bet is:
1. Matched to a fixture via `_resolve_league_team_id()` (fuzzy team name → ID)
2. Analyzed using the same `ChatOrchestrator` infrastructure (intent → profile → hit rates)
3. Explained by a `gpt-4o` call (max 200 tokens) with a concise system prompt (2-3 sentences, no follow-up questions since this is batch analysis)

### Stream Format

Results stream as newline-delimited JSON (NDJSON):

```
{"type": "bets_extracted", "count": 3, "bets": [...]}
{"type": "analysis", "index": 0, "data": {"home_team": "...", "explanation": "...", "chart_data": {...}, ...}}
{"type": "analysis", "index": 1, "data": {...}}
{"type": "analysis", "index": 2, "data": {...}}
{"type": "done"}
```

Each `analysis` event contains: `home_team`, `away_team`, `selection_label`, `bet_type`, `line`, `explanation`, `chart_data`, `home_metrics`, `away_metrics`, `home_sample_size`, `away_sample_size`, `matched`.

---

## Legacy RAG Pipeline

**Files:** `backend/rag/rag_pipeline.py`, `backend/rag/vector_store.py`, `backend/rag/document_builder.py`

A traditional retrieve-then-generate pipeline that still serves `/api/rag/query` and `/api/rag/stream`.

### Vector Store

- **Backend:** Numpy `.npy` file + JSON sidecar (replaced ChromaDB for Python 3.14 compat)
- **Embeddings:** `text-embedding-3-small` via OpenAI API
- **Storage:** `data/vector_store/embeddings.npy` + `data/vector_store/meta.json`
- **Retrieval:** Cosine similarity computed in-memory, supports Chroma-style metadata filters (`$or`, `$and`, `$eq`, `$ne`, `$in`, `$nin`)

### Document Builder

Converts CSV match rows into human-readable summaries with metadata:

```
Match: Chelsea vs Arsenal
Date: 2024-11-23
League: Premier League
Result: Chelsea 1 - 2 Arsenal
Winner: Arsenal win
Possession: Chelsea 55.3% | Arsenal 44.7%
Expected Goals (xG): Chelsea 1.2 | Arsenal 1.8
...
```

Metadata includes: `game_id`, `home_team`, `away_team`, `home_team_id`, `away_team_id`, `match_date`, `league_name`, `league_id`, `home_goals`, `away_goals`, `total_goals`.

### Pipeline Flow

1. User query → embed with `text-embedding-3-small`
2. Retrieve top-N matches (default 12) by cosine similarity, optionally filtered by team IDs
3. Concatenate match summaries as context (capped at 20,000 chars)
4. Send to `gpt-4o` with system prompt enforcing concise, data-grounded responses
5. Stream or return the full response

---

## Frontend Integration

**Files:** `frontend/src/components/ChatWindow.jsx`, `frontend/src/api/backendApi.js`

### Chat Window

- Sends `{ query, home_team_id, away_team_id, home_team_name, away_team_name, league_id }` to `/api/chat/stream`
- Buffers the stream until `\n---CHART_DATA---\n` delimiter is found
- Splits into chart JSON (rendered as `ChatMiniChart`) and explanation text
- Maintains conversation history in component state
- Supports abort via ref

### Bet Slip Upload

- Sends image as `multipart/form-data` to `/api/betslip/analyze`
- Parses NDJSON stream, updating UI progressively as each bet is analyzed
- Renders results via `BetSlipThread` component

### API Client Functions

| Function | Endpoint | Method |
|---|---|---|
| `chatStream()` | `/api/chat/stream` | POST, streaming |
| `analyzeBetSlip()` | `/api/betslip/analyze` | POST (multipart), NDJSON streaming |
| `ragStream()` | `/api/rag/stream` | POST, streaming (legacy) |

---

## API Endpoints

### `POST /api/chat/stream`

**Request body:**
```json
{
  "query": "Is over 2.5 good for Chelsea vs City?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "home_team_id": 38,
  "away_team_id": 17,
  "home_team_name": "Chelsea",
  "away_team_name": "Manchester City",
  "league_id": "EPL"
}
```

**Response:** Streaming text — chart JSON + delimiter + explanation.

### `POST /api/betslip/analyze`

**Request:** `multipart/form-data` with `image` (file) and optional `league_id` (string).

**Response:** Streaming NDJSON events.

### `POST /api/rag/ingest`

Ingests all CSV match data into the vector store. Idempotent (upsert). No request body required.

### `POST /api/rag/query`

**Request body:** `{ query, home_team_id, away_team_id, n_results, extra_context }`

**Response:** `{ answer, retrieved_matches, match_count }`

### `POST /api/rag/stream`

Same as `/api/rag/query` but streams the LLM response as plain text chunks.

---

## File Map

```
backend/rag/
├── chat_orchestrator.py    # Tool-augmented Bayesian analysis agent
├── query_classifier.py     # Query type classification (gpt-4o-mini)
├── intent_extractor.py     # ParsedIntent dataclass (used by tools)
├── team_profiler.py        # get_team_profile tool backend
├── hit_rate_builder.py     # get_hit_rates tool backend (4-layer analysis)
├── betslip_analyzer.py     # Bet slip OCR + per-bet analysis (Pipeline 2)
├── rag_pipeline.py         # Legacy RAG retrieve-then-generate
├── vector_store.py         # Numpy-backed vector store
├── document_builder.py     # CSV rows → match summaries for embedding
└── __init__.py

frontend/src/
├── api/backendApi.js       # chatStream(), analyzeBetSlip(), ragStream()
└── components/
    ├── ChatWindow.jsx      # Chat UI with stream parsing
    ├── ChatMiniChart.jsx   # Inline chart rendering
    └── BetSlipThread.jsx   # Bet slip analysis display
```
