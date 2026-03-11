# Backline LLM Architecture: RAG & Analytic Modules

This document details the complete flow of the Chat LLM orchestration.
It explains how natural-language queries are translated into structured analytic evidence, enriched with team profiles, layers of hit rates, and returned to the user.

---

## The Philosophy

The LLM is **not** an analyst; it is an *interpreter* and an *explainer*.

**Core Principles:**
1. **Never guess the data.** If the LLM computes numbers itself, it will hallucinate.
2. **Deterministic Retrieval.** All real analysis happens in strongly-typed Python modules querying PostgreSQL. The LLM only receives the final sliced data.
3. **Conciseness over Completeness.** The LLM should extract the 2–3 most critical insights rather than simply echoing the whole JSON payload back to the user.

---

## Pipeline Overview

The conversational architectue uses a single endpoint (`/api/chat/stream`) which triggers the `ChatOrchestrator`.

The orchestration happens in 5 distinct steps:

### Step 1: Query Classification

A fast, lightweight call (GPT-4o-mini) classifies the user's intent using their `history`.

**Types:**
- **`greeting`**: Friendly, non-betting banter ("Hi", "Thanks"). Gets a static friendly response.
- **`vague`**: Broad query ("What should I bet on?"). Gets a prompt suggesting specific markets to explore.
- **`conversational`**: A direct answer to a previous follow-up ("Yes", "Check over 2.5 instead"). The classifier generates a *resolved query* combining context with the answer (e.g. "Check Arsenal over 2.5").
- **`betting_intent`**: A specific request ("Can Chelsea win and is BTTS good?").

### Step 2: Multi-Intent Extraction

For betting queries, another lightweight extraction isolates **all** requested markets into structured intents (`ParsedIntent`).

An intent contains:
- `bet_type`: one of `over_under`, `one_x_two`, `double_chance`, `corners`, `btts`, `first_half_ou`, etc.
- `line`: The numerical threshold (e.g., `2.5`, `8.5`)
- `outcome_type`: For 3-way markets (`1`, `X`, `2`)
- `primary_team_hint`: The team the user focused on.

Example: *"Who wins Arsenal vs City and what about over 2.5?"*
Extracts: `[{bet_type: "one_x_two"}, {bet_type: "over_under", line: 2.5}]`

### Step 3: Team Profiling (xGD)

Before fetching hit rates, the `TeamProfiler` accesses the DB to calculate the inherent strength and form of both the home and away teams.

**Profile includes:**
- **`xgd_season_rank`**: The team's expected goal difference position across the full league table.
- **`xgd_last_5`**: Rolling form calculated as the average xGD in their last five matches.

This provides the context necessary to evaluate if a high hit rate is legitimate or inflated by an easy schedule.

### Step 4: Layered Hit Rates

The `HitRateBuilder` takes each extracted intent and routes it through the internal Workspaces to generate actual statistical evidence.

For the active team, hit rates are constructed across four increasingly specific layers:

| Layer | Description |
|---|---|
| 1. **Season** | Performance across all matches this season. |
| 2. **Perspective** | Performance narrowed down to home or away matches only. |
| 3. **Rank-filtered** | Performance against opponents mathematically similar to the upcoming opponent (±3 positions in xRank). |
| 4. **Combined** | Perspective + Rank-Filtered. The ultimate contextual metric. |

### Step 5: LLM Explanation

The original query, team profiles, and layered hit rates across all extracted intents are combined into a dense JSON payload and passed to the primary explaining model (`gpt-4o`).

The LLM uses strict instructions:
- Keep the response to 3-5 sentences.
- Lead with the strongest finding. Synthesize multiple intents gracefully.
- Include a specific, actionable follow-up question ("Want to see how this looks just for home matches?").

---

## Data Structures

### `ChatRequest`

Passed by the frontend. Contains the active fixture context and the conversation history.

```python
@dataclass
class ChatRequest:
    query: str
    history: list[dict[str, str]]
    home_team_id: int | None
    away_team_id: int | None
    home_team_name: str
    away_team_name: str
    league_id: str
    model: str
```

### `QueryClassification`
```python
@dataclass
class QueryClassification:
    query_type: Literal["greeting", "vague", "betting_intent", "conversational"]
    resolved_query: str | None
```

### `HitRateProfile`
```python
@dataclass
class HitRateProfile:
    team_name: str
    intent: ParsedIntent
    season_hits: int
    season_misses: int
    season_rate: float
    perspective: str
    perspective_hits: int
    perspective_misses: int
    perspective_rate: float
    rank_filtered_hits: int
    rank_filtered_misses: int
    rank_filtered_rate: float
    combined_hits: int
    combined_misses: int
    combined_rate: float
```