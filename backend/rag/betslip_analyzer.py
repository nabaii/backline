"""
Bet slip image analyzer: OCR via GPT-4o vision → structured bets → workspace analysis.

Flow
----
1. User uploads a bet slip image.
2. GPT-4o vision extracts structured bets (teams, bet type, line).
3. Each bet is matched to fixtures in the system.
4. Workspace analysis is run for each bet.
5. Results are returned as a list of analysis objects.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Generator

from openai import OpenAI

_DEFAULT_MODEL = "gpt-4o"

_OCR_PROMPT = """\
You are a bet slip OCR parser. Analyze this bet slip image and extract every \
individual bet/selection.

For each bet, extract:
- home_team: the home team name (full name, e.g. "Manchester City")
- away_team: the away team name (full name, e.g. "Arsenal")
- bet_type: one of "over_under", "one_x_two", "corners", "btts", "double_chance"
- line: the goal/corner line as a float (e.g. 2.5) — null if not applicable
- outcome_type: for one_x_two → "1" (home win), "X" (draw), "2" (away win). \
For btts → "yes" or "no". null otherwise.
- selection_label: the original selection text from the slip (e.g. "Over 2.5 Goals")

Return ONLY valid JSON array, no markdown:
[{"home_team": "...", "away_team": "...", "bet_type": "...", "line": ..., \
"outcome_type": ..., "selection_label": "..."}]

If you cannot read the image or find no bets, return: []"""

_SLIP_EXPLAIN_PROMPT = """\
You are Backline, a football betting research assistant analyzing a bet from \
a user's bet slip.

You have structured analytics results from real match data for this specific bet.

RESPONSE RULES — follow these strictly:
- Keep responses to 2-3 sentences. Be concise and direct.
- Lead with the key insight. No greetings, no filler.
- Pick the 2-3 most relevant stats and weave them into clear sentences.
- Explain *why* the numbers matter.
- Flag standout numbers: notably high or low hit rates, clear home/away \
splits, or streaks in recent form.
- Use plain language.
- Always tie claims to specific stats from the results. Never speculate.
- Do NOT make predictions, recommend bets, or say "best bet".
- Do NOT end with a follow-up question (this is a batch analysis)."""


@dataclass
class ExtractedBet:
    home_team: str
    away_team: str
    bet_type: str
    line: float | None
    outcome_type: str | None
    selection_label: str


class BetSlipAnalyzer:
    """Parses bet slip images and runs workspace analysis for each bet."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )

    def extract_bets(self, image_b64: str, mime_type: str = "image/png") -> list[ExtractedBet]:
        """Use GPT-4o vision to OCR the bet slip and extract structured bets."""
        response = self._client.chat.completions.create(
            model=_DEFAULT_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=[
                {"role": "system", "content": _OCR_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            return []

        bets: list[ExtractedBet] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            bets.append(ExtractedBet(
                home_team=item.get("home_team", ""),
                away_team=item.get("away_team", ""),
                bet_type=item.get("bet_type", "over_under"),
                line=item.get("line"),
                outcome_type=item.get("outcome_type"),
                selection_label=item.get("selection_label", ""),
            ))

        return bets

    def analyze_bet(self, bet: ExtractedBet, league_id: str = "") -> dict[str, Any]:
        """Run workspace analysis for a single extracted bet."""
        from backend.rag.chat_orchestrator import ChatOrchestrator, ChatRequest, ParsedIntent

        orchestrator = ChatOrchestrator()
        intent = ParsedIntent(
            bet_type=bet.bet_type,
            line=bet.line,
            outcome_type=bet.outcome_type,
        )

        # Resolve team IDs using backend team resolution
        from backend.backend_api import _resolve_league_team_id

        home_team_id = _resolve_league_team_id(league_id, bet.home_team)
        away_team_id = _resolve_league_team_id(league_id, bet.away_team)

        request = ChatRequest(
            query=f"Analyze {bet.selection_label} for {bet.home_team} vs {bet.away_team}",
            home_team_id=home_team_id or None,
            away_team_id=away_team_id or None,
            home_team_name=bet.home_team,
            away_team_name=bet.away_team,
            league_id=league_id,
        )

        try:
            metrics = orchestrator.query_workspace(intent, request)
        except Exception:
            metrics = {}

        # Generate explanation
        explanation = ""
        if metrics:
            try:
                user_msg = f"DATA:\n{json.dumps(metrics, indent=2)}\n\nBET: {bet.selection_label} — {bet.home_team} vs {bet.away_team}"
                resp = self._client.chat.completions.create(
                    model=_DEFAULT_MODEL,
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": _SLIP_EXPLAIN_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
                explanation = resp.choices[0].message.content or ""
            except Exception:
                explanation = "Could not generate analysis for this bet."

        chart_data = {
            "bet_type": metrics.get("bet_type", bet.bet_type),
            "line": metrics.get("line", bet.line),
            "home_team": bet.home_team,
            "recent_matches": metrics.get("recent_matches", []),
        }

        return {
            "home_team": bet.home_team,
            "away_team": bet.away_team,
            "selection_label": bet.selection_label,
            "bet_type": bet.bet_type,
            "line": bet.line,
            "explanation": explanation,
            "chart_data": chart_data,
            "home_metrics": metrics.get("home_metrics", {}),
            "away_metrics": metrics.get("away_metrics", {}),
            "home_sample_size": metrics.get("home_sample_size", 0),
            "away_sample_size": metrics.get("away_sample_size", 0),
            "matched": bool(home_team_id or away_team_id),
        }

    def analyze_slip(
        self,
        image_b64: str,
        mime_type: str = "image/png",
        league_id: str = "",
    ) -> Generator[str, None, None]:
        """
        Full pipeline: OCR → match → analyze each bet.
        Yields newline-delimited JSON events for streaming.
        """
        # Step 1: Extract bets
        bets = self.extract_bets(image_b64, mime_type)

        yield json.dumps({
            "type": "bets_extracted",
            "count": len(bets),
            "bets": [
                {
                    "home_team": b.home_team,
                    "away_team": b.away_team,
                    "selection_label": b.selection_label,
                    "bet_type": b.bet_type,
                    "line": b.line,
                }
                for b in bets
            ],
        }) + "\n"

        if not bets:
            yield json.dumps({"type": "done"}) + "\n"
            return

        # Step 2: Analyze each bet
        for i, bet in enumerate(bets):
            try:
                analysis = self.analyze_bet(bet, league_id=league_id)
                yield json.dumps({
                    "type": "analysis",
                    "index": i,
                    "data": analysis,
                }) + "\n"
            except Exception as exc:
                yield json.dumps({
                    "type": "analysis",
                    "index": i,
                    "data": {
                        "home_team": bet.home_team,
                        "away_team": bet.away_team,
                        "selection_label": bet.selection_label,
                        "bet_type": bet.bet_type,
                        "line": bet.line,
                        "explanation": f"Error analyzing this bet: {exc}",
                        "chart_data": {"recent_matches": []},
                        "matched": False,
                    },
                }) + "\n"

        yield json.dumps({"type": "done"}) + "\n"
