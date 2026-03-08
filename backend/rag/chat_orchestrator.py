"""
Chat orchestrator: natural language → workspace query → LLM explanation.

Replaces pure RAG with intent-driven workspace calls.

Flow
----
1. LLM parses user question into structured intent (bet type, line, etc.)
2. Python calls the appropriate workspace internally to get metrics.
3. LLM explains the structured results concisely.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Generator, Literal

import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gpt-4o"
_INTENT_MODEL = "gpt-4o-mini"  # Cheaper/faster model for structured intent parsing

_INTENT_PROMPT = """\
You are an intent parser for a football betting analytics platform.

Given the user's question and fixture context, extract:
1. bet_type: one of "over_under", "one_x_two", "double_chance", "corners"
2. line: a float goal/corner line (default 2.5 for over_under, 8.5 for corners, null for others)
3. outcome_type: for one_x_two only — "1" (win), "X" (draw), "2" (loss). null otherwise.

Return ONLY valid JSON, no markdown:
{"bet_type": "...", "line": ..., "outcome_type": ...}

IMPORTANT — be intuitive. If the user asks about a team without naming a \
specific market, infer the most relevant one:
- "how does X play" / "how do X perform" / general team form → "over_under" (2.5)
- "can X win" / "X vs Y" / "who wins" → "one_x_two"
- "how do X play against similar teams" → "over_under" (2.5)
- "corners" / "set pieces" → "corners"
- "both teams score" / "BTTS" → "over_under" (0.5) with outcome_type null

ONLY return null bet_type for truly non-football queries: greetings, \
off-topic chat, or questions with zero football context."""

_EXPLAIN_PROMPT = """\
You are Backline, a football betting research assistant.

You have structured analytics results from real match data.

RESPONSE RULES — follow these strictly:
- Keep responses to 3-4 sentences, plus a follow-up question. Never write \
more than a short paragraph.
- Lead with the key insight. No greetings, no filler, no "Great question!".
- Pick the 2-3 most relevant stats and weave them into clear sentences. \
Do not list every metric.
- Explain *why* the numbers matter, not just what they are. Teach the user \
something about the data.
- Flag standout numbers: notably high or low hit rates, clear home/away \
splits, or streaks in recent form. If something jumps out, say so.
- Use plain language. Write like you're talking to a smart friend.
- Always tie claims to specific stats from the results. Never speculate.
- Do NOT make predictions or say "best bet". You can analyse and present \
data for any market — even if the user didn't name one explicitly. Be \
intuitive: if the user asks a general question, present the data you have \
and guide them deeper.
- If the data is insufficient, say so briefly.
- ALWAYS end your response with a short follow-up question that guides the \
user toward a natural next step. Make it specific and actionable, e.g.: \
"Want me to check how Villarreal's over 2.5 rate looks against opponents \
ranked similarly to Elche?" or "Should I look at this from the away team's \
perspective?" or "Want to see how this changes with a different line?"
- Keep the conversation flowing. Your goal is to help the user explore the \
data, not to block them."""

_NO_MARKET_RESPONSE = (
    "I need a fixture to work with. Select a match from the sidebar, "
    "then ask me anything — I'll pull the data and break it down for you."
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ChatRequest:
    query: str
    home_team_id: int | None = None
    away_team_id: int | None = None
    home_team_name: str = ""
    away_team_name: str = ""
    league_id: str = ""
    model: str = _DEFAULT_MODEL


@dataclass
class ParsedIntent:
    bet_type: str | None
    line: float | None
    outcome_type: str | None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ChatOrchestrator:
    """
    Turns natural-language questions into workspace queries,
    then explains the results.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )

    # ------------------------------------------------------------------
    # Step 1: Parse intent
    # ------------------------------------------------------------------

    def parse_intent(self, query: str, fixture_context: str, model: str = _DEFAULT_MODEL) -> ParsedIntent:
        user_msg = f"Fixture: {fixture_context}\nQuestion: {query}" if fixture_context else query

        response = self._client.chat.completions.create(
            model=_INTENT_MODEL,
            max_tokens=100,
            temperature=0,
            messages=[
                {"role": "system", "content": _INTENT_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ParsedIntent(bet_type=None, line=None, outcome_type=None)

        return ParsedIntent(
            bet_type=parsed.get("bet_type"),
            line=parsed.get("line"),
            outcome_type=parsed.get("outcome_type"),
        )

    # ------------------------------------------------------------------
    # Step 2: Query workspace
    # ------------------------------------------------------------------

    def query_workspace(self, intent: ParsedIntent, request: ChatRequest) -> dict[str, Any]:
        """Call the appropriate workspace and return structured metrics."""
        from backend.backend_api import (
            _build_store,
            _load_raw_df,
            _resolve_team_anchor_match,
            _split_over_under_counts,
            _split_result_counts,
            _split_double_chance_counts,
            _split_corner_metrics,
        )
        from backend.bet_type.over_under.over_under import OverUnderWorkspace
        from backend.bet_type.one_x_two.one_x_two import OneXTwoWorkspace
        from backend.bet_type.double_chance.double_chance import DoubleChanceWorkspace
        from backend.bet_type.corners.corners import CornerWorkspace

        store = _build_store()
        raw_df = _load_raw_df()

        home_team_id = request.home_team_id or -1
        away_team_id = request.away_team_id or -1

        home_anchor = _resolve_team_anchor_match(
            raw_df, home_team_id, request.home_team_name, "home",
            league_id=request.league_id,
        )
        away_anchor = _resolve_team_anchor_match(
            raw_df, away_team_id, request.away_team_name, "away",
            league_id=request.league_id,
        )

        bet_type = intent.bet_type or "over_under"
        line = intent.line
        filters: list = []

        # Build workspace
        if bet_type == "over_under":
            workspace = OverUnderWorkspace(store)
            line = line if line is not None else 2.5
        elif bet_type == "one_x_two":
            outcome = intent.outcome_type or "1"
            workspace = OneXTwoWorkspace(store, outcome_type=outcome)
        elif bet_type == "double_chance":
            workspace = DoubleChanceWorkspace(store)
        elif bet_type == "corners":
            workspace = CornerWorkspace(store)
            line = line if line is not None else 8.5
        else:
            workspace = OverUnderWorkspace(store)
            line = line if line is not None else 2.5

        # Query for each team
        home_df = pd.DataFrame()
        away_df = pd.DataFrame()

        evidence_kwargs: dict[str, Any] = {
            "bet_type": bet_type,
            "filters": filters,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
        }
        if bet_type in ("over_under", "corners"):
            evidence_kwargs["line"] = line

        if home_anchor is not None:
            anchor_id, perspective = home_anchor
            evidence = workspace.get_evidence(
                match_id=anchor_id, perspective=perspective, **evidence_kwargs,
            )
            home_df = evidence.df

        if away_anchor is not None:
            anchor_id, perspective = away_anchor
            evidence = workspace.get_evidence(
                match_id=anchor_id, perspective=perspective, **evidence_kwargs,
            )
            away_df = evidence.df

        # Compute metrics
        if bet_type == "over_under":
            home_metrics = _split_over_under_counts(home_df, line)
            away_metrics = _split_over_under_counts(away_df, line)
        elif bet_type == "one_x_two":
            home_metrics = _split_result_counts(home_df)
            away_metrics = _split_result_counts(away_df)
        elif bet_type == "double_chance":
            home_metrics = _split_double_chance_counts(home_df)
            away_metrics = _split_double_chance_counts(away_df)
        elif bet_type == "corners":
            home_metrics = _split_corner_metrics(home_df, line)
            away_metrics = _split_corner_metrics(away_df, line)
        else:
            home_metrics = {}
            away_metrics = {}

        home_sample = len(home_df)
        away_sample = len(away_df)

        # Build recent_matches for chart rendering (home team / primary)
        recent_matches = self._build_recent_matches(
            home_df, bet_type, line,
        )

        return {
            "bet_type": bet_type,
            "line": line,
            "outcome_type": intent.outcome_type,
            "home_team": request.home_team_name or str(home_team_id),
            "away_team": request.away_team_name or str(away_team_id),
            "home_sample_size": home_sample,
            "away_sample_size": away_sample,
            "home_metrics": home_metrics,
            "away_metrics": away_metrics,
            "recent_matches": recent_matches,
        }

    @staticmethod
    def _build_recent_matches(
        df: pd.DataFrame,
        bet_type: str,
        line: float | None,
    ) -> list[dict[str, Any]]:
        """Build lightweight match records for the chat mini-chart."""
        if df.empty:
            return []

        from backend.backend_api import _match_name_index

        match_lookup = _match_name_index()
        rows: list[dict[str, Any]] = []

        for row in df.to_dict("records"):
            match_id = int(row.get("match_id", 0))
            venue = str(row.get("venue", ""))
            home_name, away_name = match_lookup.get(match_id, ("Home", "Away"))
            opponent_name = away_name if venue == "home" else home_name

            entry: dict[str, Any] = {
                "match_id": match_id,
                "venue": venue,
                "opponent_name": opponent_name,
            }

            if bet_type == "over_under":
                total = float(row.get("total_goals", 0))
                entry["value"] = total
                entry["result"] = "O" if total > (line or 2.5) else "U"
            elif bet_type == "one_x_two":
                gs = float(row.get("goals_scored", 0))
                og = float(row.get("opponent_goals", 0))
                if gs > og:
                    entry["value"], entry["result"] = 1.0, "W"
                elif gs == og:
                    entry["value"], entry["result"] = 0.5, "D"
                else:
                    entry["value"], entry["result"] = 0.1, "L"
            elif bet_type == "double_chance":
                gs = float(row.get("goals_scored", 0))
                og = float(row.get("opponent_goals", 0))
                hit = gs >= og
                entry["value"] = 1.0 if hit else 0.1
                entry["result"] = "H" if hit else "M"
            elif bet_type == "corners":
                tc = float(row.get("total_corners", 0))
                entry["value"] = tc
                entry["result"] = "O" if tc > (line or 8.5) else "U"

            rows.append(entry)

        return rows

    # ------------------------------------------------------------------
    # Step 3: Explain results
    # ------------------------------------------------------------------

    def explain(
        self,
        query: str,
        metrics: dict[str, Any],
        model: str = _DEFAULT_MODEL,
    ) -> str:
        user_msg = f"DATA:\n{json.dumps(metrics, indent=2)}\n\nUSER QUESTION:\n{query}"

        response = self._client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=[
                {"role": "system", "content": _EXPLAIN_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content or ""

    def explain_stream(
        self,
        query: str,
        metrics: dict[str, Any],
        model: str = _DEFAULT_MODEL,
    ) -> Generator[str, None, None]:
        user_msg = f"DATA:\n{json.dumps(metrics, indent=2)}\n\nUSER QUESTION:\n{query}"

        response = self._client.chat.completions.create(
            model=model,
            max_tokens=300,
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

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, request: ChatRequest) -> str:
        fixture_ctx = (
            f"{request.home_team_name} vs {request.away_team_name}"
            if request.home_team_name and request.away_team_name
            else ""
        )

        # No fixture selected and no teams mentioned — can't do anything
        if not request.home_team_id and not request.away_team_id and not fixture_ctx:
            return _NO_MARKET_RESPONSE

        intent = self.parse_intent(request.query, fixture_ctx, model=request.model)

        # If intent parser couldn't pick a market, default to over_under
        if intent.bet_type is None:
            intent = ParsedIntent(bet_type="over_under", line=2.5, outcome_type=None)

        metrics = self.query_workspace(intent, request)
        return self.explain(request.query, metrics, model=request.model)

    # Delimiter separating chart JSON from streamed text
    CHART_DELIMITER = "\n---CHART_DATA---\n"

    def stream(self, request: ChatRequest) -> Generator[str, None, None]:
        fixture_ctx = (
            f"{request.home_team_name} vs {request.away_team_name}"
            if request.home_team_name and request.away_team_name
            else ""
        )

        # No fixture selected and no teams mentioned — can't do anything
        if not request.home_team_id and not request.away_team_id and not fixture_ctx:
            yield _NO_MARKET_RESPONSE
            return

        intent = self.parse_intent(request.query, fixture_ctx, model=request.model)

        # If intent parser couldn't pick a market, default to over_under
        if intent.bet_type is None:
            intent = ParsedIntent(bet_type="over_under", line=2.5, outcome_type=None)

        metrics = self.query_workspace(intent, request)

        # Yield chart data as JSON prefix, then delimiter, then text stream
        chart_payload = {
            "bet_type": metrics.get("bet_type"),
            "line": metrics.get("line"),
            "home_team": metrics.get("home_team"),
            "recent_matches": metrics.get("recent_matches", []),
        }
        yield json.dumps(chart_payload)
        yield self.CHART_DELIMITER

        yield from self.explain_stream(request.query, metrics, model=request.model)
