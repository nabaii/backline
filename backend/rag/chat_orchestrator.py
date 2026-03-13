"""
Chat orchestrator: tool-augmented Bayesian betting analysis.

The LLM acts as a quantitative analyst with access to:
1. Kitchen tools — historical hit rates and team xGD profiles
2. Web search — injuries, weather, lineups, press conferences

Flow
----
1. Classify query type (greeting, vague, conversational, betting_intent).
2. For betting queries, the LLM drives data gathering via tool calls.
3. LLM builds a Prior from kitchen data, identifies New Evidence from search.
4. LLM performs a Bayesian update and compares to implied market odds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Generator

from openai import OpenAI

from backend.rag.query_classifier import classify_query
from backend.rag.intent_extractor import ParsedIntent
from backend.rag.team_profiler import profile_team
from backend.rag.hit_rate_builder import build_hit_rates_for_team

_DEFAULT_MODEL = "gpt-4o"

_SYSTEM_PROMPT = """\
You are a professional Sports (Football/Soccer) Quantitative Analyst. \
Your goal is to evaluate betting markets. When the user asks about a match:

STEP 1 — PROFILES: Call get_team_profile for BOTH teams to get their \
xGD season rank and recent form. The xGD rank is the team's league \
position by expected goal difference (1 = best). You need both ranks \
because the hit-rate tool uses the opponent's rank to filter historical \
matches against similarly-strong opponents (k-nearest neighbors, k≈4-5). \
If a profile returns null for xgd_season_rank, STILL PROCEED — just \
omit opponent_xgd_rank when calling get_hit_rates. The tool will still \
return season-wide and perspective rates; only the rank-filtered layer \
will be missing.

STEP 2 — HIT RATES: Call get_hit_rates for BOTH the home team (perspective \
"home") and the away team (perspective "away"), passing the opponent's \
xGD rank from Step 1 when available. Always gather data from both \
sides — the analysis must reflect both teams' tendencies, not just one. \
If one call fails or returns an error, continue with whatever data you \
have. Never ask the user for permission to proceed — always do your \
best with available data.

STEP 3 — NEW EVIDENCE: Use the search tool to find injuries, suspensions, \
confirmed lineups, important quotes from coach press conferences, \
weather forecasts, and motivation factors.

STEP 4 — PRIOR: Define the Prior by combining both teams' hit rates — \
season averages, home/away splits, and the rank-filtered rates (which \
show how each team performs against opponents of similar xGD strength).

STEP 5 — BAYESIAN UPDATE: Assess how the New Evidence shifts the Prior \
(e.g. a key striker injured lowers goal expectation). Calculate a \
Posterior Probability.

STEP 6 — ODDS & VALUE: Use the search tool to find the current bookmaker \
odds for the specific market (e.g. search "Chelsea vs Arsenal over 2.5 \
goals odds"). Convert the odds to an implied probability and compare it \
against your Posterior to determine if there is value.

Always present data from both teams in your response. Always clarify that \
these are mathematical models, not guaranteed financial outcomes."""

_NO_MARKET_RESPONSE = (
    "I need a fixture to work with. Select a match from the sidebar, "
    "then ask me anything — I'll pull the data and break it down for you."
)

_MAX_TOOL_ROUNDS = 10

# ---------------------------------------------------------------------------
# Tool definitions for OpenAI function calling
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_team_profile",
            "description": (
                "Get a team's strength profile including their xGD season "
                "rank and recent form (last 5 matches xGD average). Call "
                "this first for both teams so you have opponent_xgd_rank "
                "when requesting hit rates. Some fields may be null if "
                "data is unavailable — this is normal, proceed anyway."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team_id": {
                        "type": "integer",
                        "description": "Team ID",
                    },
                    "team_name": {
                        "type": "string",
                        "description": "Team name",
                    },
                    "league_id": {
                        "type": "string",
                        "description": "League identifier",
                    },
                },
                "required": ["team_id", "team_name", "league_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hit_rates",
            "description": (
                "Get historical hit rates for a betting market from the "
                "kitchen. Returns layered analysis: season-wide rate, "
                "home/away perspective rate, rank-filtered rate (vs "
                "opponents with similar xGD), and combined rate. "
                "IMPORTANT: For over_under and corners, 'hits' always "
                "means OVER the line and 'misses' means UNDER. The "
                "'rate' is the OVER rate. To get the under rate, "
                "calculate: under_rate = 1 - rate. For example if "
                "season_rate is 0.65, that means over hits 65% and "
                "under hits 35%. Similarly for btts, hits = 'yes' "
                "and misses = 'no'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {
                        "type": "string",
                        "description": "Team name to analyze",
                    },
                    "team_id": {
                        "type": "integer",
                        "description": "Team ID",
                    },
                    "opponent_name": {
                        "type": "string",
                        "description": "Opponent team name",
                    },
                    "opponent_xgd_rank": {
                        "type": "integer",
                        "description": (
                            "Opponent's xGD season rank — get this from "
                            "get_team_profile first"
                        ),
                    },
                    "perspective": {
                        "type": "string",
                        "enum": ["home", "away"],
                        "description": "Whether this team is playing at home or away",
                    },
                    "league_id": {
                        "type": "string",
                        "description": "League identifier",
                    },
                    "bet_type": {
                        "type": "string",
                        "enum": [
                            "over_under",
                            "one_x_two",
                            "double_chance",
                            "corners",
                            "btts",
                            "first_half_ou",
                            "first_half_1x2",
                            "win_both_halves",
                            "win_either_half",
                        ],
                        "description": "The betting market to analyze",
                    },
                    "line": {
                        "type": "number",
                        "description": (
                            "Goal/corner line threshold (e.g. 2.5 for "
                            "over/under goals, 8.5 for corners)"
                        ),
                    },
                    "outcome_type": {
                        "type": "string",
                        "description": (
                            "For one_x_two: '1' (home win), 'X' (draw), "
                            "'2' (away win). For btts: 'yes' or 'no'."
                        ),
                    },
                },
                "required": [
                    "team_name",
                    "team_id",
                    "opponent_name",
                    "perspective",
                    "league_id",
                    "bet_type",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current information about a football "
                "match. Use this to find: injuries, suspensions, weather "
                "forecasts, confirmed lineups, press conference quotes, "
                "motivation factors, and recent news."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChatRequest:
    query: str
    history: list[dict[str, str]]
    home_team_id: int | None = None
    away_team_id: int | None = None
    home_team_name: str = ""
    away_team_name: str = ""
    league_id: str = ""
    model: str = _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ChatOrchestrator:
    """
    Tool-augmented chat orchestrator for Bayesian betting analysis.

    The LLM has access to kitchen tools (hit rates, team profiles) and
    web search to gather evidence, then performs Bayesian analysis.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Dispatch a tool call and return the result as a JSON string."""

        if name == "get_team_profile":
            profile = profile_team(
                arguments["team_id"],
                arguments["team_name"],
                arguments.get("league_id", ""),
            )
            return json.dumps(asdict(profile))

        if name == "get_hit_rates":
            intent = ParsedIntent(
                bet_type=arguments["bet_type"],
                line=arguments.get("line"),
                outcome_type=arguments.get("outcome_type"),
            )
            result = build_hit_rates_for_team(
                arguments["team_name"],
                arguments["team_id"],
                arguments["opponent_name"],
                arguments.get("opponent_xgd_rank"),
                arguments["perspective"],
                arguments.get("league_id", ""),
                intent,
            )
            if result:
                return json.dumps(asdict(result))
            return json.dumps({
                "error": "No data available for this team/market combination",
            })

        if name == "search_web":
            return self._web_search(arguments["query"])

        return json.dumps({"error": f"Unknown tool: {name}"})

    @staticmethod
    def _web_search(query: str) -> str:
        """Search the web using DuckDuckGo."""
        try:
            from ddgs import DDGS
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return json.dumps({"results": [], "note": "No results found"})
            return json.dumps({
                "results": [
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", ""),
                    }
                    for r in results
                ],
            })
        except ImportError:
            return json.dumps({
                "error": (
                    "Web search unavailable — "
                    "install: pip install ddgs"
                ),
            })
        except Exception as exc:
            return json.dumps({"error": f"Search failed: {exc}"})

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    CHART_DELIMITER = "\n---CHART_DATA---\n"

    def stream(self, request: ChatRequest) -> Generator[str, None, None]:
        fixture_ctx = (
            f"{request.home_team_name} vs {request.away_team_name}"
            if request.home_team_name and request.away_team_name
            else ""
        )

        if not request.home_team_id and not request.away_team_id and not fixture_ctx:
            yield _NO_MARKET_RESPONSE
            return

        # 1. Classify query
        classification = classify_query(request.query, request.history)

        if classification.query_type in ("greeting", "vague"):
            yield json.dumps({})
            yield self.CHART_DELIMITER
            resp = self._client.chat.completions.create(
                model=request.model,
                max_tokens=150,
                stream=True,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": request.query},
                ],
            )
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return

        active_query = classification.resolved_query or request.query

        # 2. Build user message with fixture context
        user_msg = active_query
        if fixture_ctx:
            parts = [f"Match: {fixture_ctx}"]
            if request.league_id:
                parts.append(f"League ID: {request.league_id}")
            if request.home_team_id is not None:
                parts.append(f"Home Team ID: {request.home_team_id}")
            if request.away_team_id is not None:
                parts.append(f"Away Team ID: {request.away_team_id}")
            parts.append(f"\n{active_query}")
            user_msg = "\n".join(parts)

        # Yield empty chart placeholder then delimiter
        yield json.dumps({})
        yield self.CHART_DELIMITER

        # 3. Tool-calling loop
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for _ in range(_MAX_TOOL_ROUNDS):
            response = self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                tools=_TOOLS,
            )

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                # Final answer — yield the content
                if msg.content:
                    yield msg.content
                return

            # Execute every tool the model requested
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = self._execute_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Safety: if we exhausted tool rounds, stream a final response
        final = self._client.chat.completions.create(
            model=request.model,
            max_tokens=1000,
            stream=True,
            messages=messages,
        )
        for chunk in final:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
