"""
Chat orchestrator: natural language → query classification → intent extraction → team profiling → layered hit-rate analysis → LLM explanation.

Replaces pure RAG with intent-driven workspace calls.

Flow
----
1. Classifier determines query type (greeting, vague, conversational, betting_intent).
2. IntentExtractor structures >= 1 betting intents from the query.
3. TeamProfiler gets xGD rank and recent form.
4. HitRateBuilder layered analysis for season, perspective, rank, and combined.
5. Explanation LLM summarizes the structured results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Generator

from openai import OpenAI

from backend.rag.query_classifier import classify_query, QueryClassification
from backend.rag.intent_extractor import extract_intents, ParsedIntent
from backend.rag.team_profiler import profile_team
from backend.rag.hit_rate_builder import build_hit_rates_for_team

_DEFAULT_MODEL = "gpt-4o"

_EXPLAIN_PROMPT = """\
You are Backline, a football betting research assistant.

You have structured analytics results from real match data, including multiple \
intents if the user asked a complex question, layered hit rates, and team xGD profiles.

RESPONSE RULES — follow these strictly:
- Keep responses to 3-5 sentences, plus a follow-up question. Never write \
more than a short paragraph.
- Lead with the key insight. No greetings, no filler, no "Great question!".
- Synthesize the findings if there are multiple intents. Pick the 2-3 most \
relevant stats across all data and weave them into clear sentences. Do not \
list every metric.
- Use the team xGD profiles (rank and recent form) to contextualize the hit \
rates. Point out when a team is over/under-performing their underlying numbers.
- Explain *why* the numbers matter, not just what they are.
- Flag standout numbers: notably high or low hit rates, stark contrast \
between season rate and rank-filtered rate, or streaks in recent form.
- Use plain language. Write like you're talking to a smart friend.
- Always tie claims to specific stats from the results. Never speculate.
- Do NOT make predictions or say "best bet". You can analyse and present \
data for any market.
- If the data is insufficient, say so briefly.
- ALWAYS end your response with a short follow-up question that guides the \
user toward a natural next step. Make it specific and actionable.
- Keep the conversation flowing. Your goal is to help the user explore the \
data, not to block them."""

_VAGUE_PROMPT = """\
You are Backline, a football betting research assistant.

The user asked a vague or broad question without a specific betting intent.
Respond briefly (1-2 sentences) and suggest 2-3 specific bet types they could \
ask you to analyze for the currently selected fixture (e.g., Over 2.5 goals, \
Match Winner, or Both Teams to Score). Ask them what they'd like to look at."""

_GREETING_PROMPT = """\
You are Backline, a football betting research assistant.

The user sent a greeting or off-topic message. Respond politely but briefly \
(1 sentence) and ask them what football match or betting market they'd like \
to analyze."""

_NO_MARKET_RESPONSE = (
    "I need a fixture to work with. Select a match from the sidebar, "
    "then ask me anything — I'll pull the data and break it down for you."
)


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


class ChatOrchestrator:
    """
    Turns natural-language questions into workspace queries,
    then explains the results with layered xGD analysis.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )

    def _generate_static_response(self, prompt: str, query: str, model: str) -> Generator[str, None, None]:
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=150,
            stream=True,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _query_and_profile(
        self,
        intents: list[ParsedIntent],
        request: ChatRequest,
    ) -> dict[str, Any]:
        """Profile teams and build layered hit rates for all intents."""
        home_team_id = request.home_team_id or -1
        away_team_id = request.away_team_id or -1

        # Profile teams
        home_profile = profile_team(home_team_id, request.home_team_name, request.league_id)
        away_profile = profile_team(away_team_id, request.away_team_name, request.league_id)

        all_results = []
        chart_intent = intents[0] if intents else None
        recent_matches_for_chart = []

        for intent in intents:
            # Build hit rates
            home_hits = build_hit_rates_for_team(
                request.home_team_name, home_team_id, request.away_team_name,
                away_profile.xgd_season_rank, "home", request.league_id, intent
            )
            away_hits = build_hit_rates_for_team(
                request.away_team_name, away_team_id, request.home_team_name,
                home_profile.xgd_season_rank, "away", request.league_id, intent
            )

            intent_result = {
                "intent": asdict(intent),
                "home_analysis": asdict(home_hits) if home_hits else None,
                "away_analysis": asdict(away_hits) if away_hits else None,
            }
            all_results.append(intent_result)

        # Legacy structured dictionary for the chart
        # (We use the first intent's data for the UI chart)
        first_intent_dict = {}
        if chart_intent:
            first_intent_dict = {
                "bet_type": chart_intent.bet_type,
                "line": chart_intent.line,
                "home_team": request.home_team_name,
                "recent_matches": [],  # We don't populate recent matches in the new intent flow
                                       # yet since the UI chart needs to be updated. Just passing empty list for now.
            }

        return {
            "teams": {
                "home": asdict(home_profile),
                "away": asdict(away_profile),
            },
            "intents_analysis": all_results,
            "_chart_data": first_intent_dict,
        }

    def explain_stream(
        self,
        query: str,
        metrics: dict[str, Any],
        model: str = _DEFAULT_MODEL,
    ) -> Generator[str, None, None]:
        user_msg = f"DATA:\n{json.dumps(metrics, indent=2)}\n\nUSER QUESTION:\n{query}"

        response = self._client.chat.completions.create(
            model=model,
            max_tokens=400,
            stream=True,
            messages=[
                {"role": "system", "content": _EXPLAIN_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

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

        # 1. Classify Query
        classification = classify_query(request.query, request.history)

        if classification.query_type == "greeting":
            yield json.dumps({})  # empty chart
            yield self.CHART_DELIMITER
            yield from self._generate_static_response(_GREETING_PROMPT, request.query, request.model)
            return

        if classification.query_type == "vague":
            yield json.dumps({})
            yield self.CHART_DELIMITER
            yield from self._generate_static_response(_VAGUE_PROMPT, request.query, request.model)
            return

        # Handle betting intent or resolved conversational intent
        active_query = classification.resolved_query or request.query

        # 2. Extract Intents
        intents = extract_intents(active_query, fixture_ctx)

        # 3. Profile & Layered Hit Rates
        metrics = self._query_and_profile(intents, request)
        chart_payload = metrics.pop("_chart_data", {})

        # Yield chart format
        yield json.dumps(chart_payload)
        yield self.CHART_DELIMITER

        # Yield explanation
        yield from self.explain_stream(active_query, metrics, model=request.model)
