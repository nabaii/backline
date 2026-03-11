"""
Intent extractor: parse a user query into one or more structured betting intents.

Called only when the query classifier returns ``betting_intent``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

_MODEL = "gpt-4o-mini"

_EXTRACT_PROMPT = """\
You are an intent parser for a football betting analytics platform.

Given the user's question and fixture context, extract ALL distinct betting \
intents present in the query. There may be one or several.

For each intent, extract:
- bet_type: one of "over_under", "one_x_two", "double_chance", "corners", "btts"
- line: a float goal/corner line (default 2.5 for over_under, 8.5 for corners, \
null for others)
- outcome_type: for one_x_two — "1" (home win), "X" (draw), "2" (away win). \
For btts — "yes" or "no". null otherwise.
- primary_team_hint: which team the user seems focused on (string, or null if \
both/neither)

Be intuitive with implicit intents:
- "how does X play" / general form → over_under (2.5)
- "can X win" / "X vs Y" / "who wins" → one_x_two
- "both teams score" / "BTTS" → btts
- "corners" / "set pieces" → corners

Return ONLY valid JSON array, no markdown:
[{"bet_type": "...", "line": ..., "outcome_type": ..., "primary_team_hint": ...}]

If no football intent can be extracted, return: []"""


@dataclass
class ParsedIntent:
    """A single structured betting intent."""
    bet_type: str
    line: float | None
    outcome_type: str | None
    primary_team_hint: str | None = None


def extract_intents(
    query: str,
    fixture_context: str,
    *,
    api_key: str | None = None,
) -> list[ParsedIntent]:
    """Extract one or more betting intents from a user query."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))

    user_msg = (
        f"Fixture: {fixture_context}\nQuestion: {query}"
        if fixture_context
        else f"Question: {query}"
    )

    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=300,
        temperature=0,
        messages=[
            {"role": "system", "content": _EXTRACT_PROMPT},
            {"role": "user", "content": user_msg},
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
        parsed = [parsed] if isinstance(parsed, dict) else []

    intents: list[ParsedIntent] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        bet_type = item.get("bet_type")
        if not bet_type:
            continue
        intents.append(ParsedIntent(
            bet_type=bet_type,
            line=item.get("line"),
            outcome_type=item.get("outcome_type"),
            primary_team_hint=item.get("primary_team_hint"),
        ))

    # Fallback: if nothing extracted, default to over_under
    if not intents:
        intents = [ParsedIntent(bet_type="over_under", line=2.5, outcome_type=None)]

    return intents
