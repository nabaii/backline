"""
Query classifier: determines which type of input the user sent.

Types
-----
- greeting      : social/off-topic (e.g. "Hey", "Thanks")
- vague         : no specific bet intent (e.g. "What should I bet on?")
- betting_intent: contains ≥1 betting intent (e.g. "Is over 2.5 good?")
- conversational: response to a prior LLM question (e.g. "Yes", "Sure")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

QueryType = Literal["greeting", "vague", "betting_intent", "conversational"]

_MODEL = "gpt-4o-mini"

_CLASSIFY_PROMPT = """\
You are a query classifier for a football betting analytics assistant called Backline.

Given the user's message and recent conversation history, classify it into ONE of:

1. "greeting" — social pleasantries, thank-yous, or off-topic chat with zero \
football/betting context. Examples: "Hey", "Thanks!", "How are you?"
2. "vague" — football-related but no specific bet/market/team mentioned. \
Examples: "What should I bet on?", "Any good bets today?", "Help me find a bet"
3. "betting_intent" — contains one or more specific betting intents \
(team names, bet types, markets, match references). \
Examples: "Is over 2.5 good for Chelsea vs Villa?", "Can Arsenal win and BTTS?"
4. "conversational" — a short reply to the assistant's previous question \
(e.g. "Yes", "Sure", "Check that", "No, the other one"). \
Only classify as conversational when there IS prior conversation context \
and the message clearly responds to it.

When the type is "conversational", also produce a "resolved_query" that \
inlines the context so it reads as a standalone betting question. \
For example, if the assistant asked "Want me to check Villarreal's over 2.5 \
against similar opponents?" and the user says "Yes", the resolved_query \
should be "Check Villarreal's over 2.5 against similar opponents".

Return ONLY valid JSON, no markdown:
{"query_type": "...", "resolved_query": "...or null if not conversational"}"""


@dataclass
class QueryClassification:
    query_type: QueryType
    resolved_query: str | None  # Only set for "conversational" type


def classify_query(
    query: str,
    history: list[dict[str, str]],
    *,
    api_key: str | None = None,
) -> QueryClassification:
    """Classify the user query using conversation history for context."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))

    # Build a concise history summary (last 4 messages max)
    recent = history[-4:] if history else []
    history_block = ""
    if recent:
        lines = []
        for msg in recent:
            role = msg.get("role", "user")
            content = (msg.get("content") or "")[:300]
            lines.append(f"{role}: {content}")
        history_block = "\n".join(lines)

    user_content = f"CONVERSATION HISTORY:\n{history_block}\n\nUSER MESSAGE:\n{query}" if history_block else f"USER MESSAGE:\n{query}"

    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=150,
        temperature=0,
        messages=[
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Default to betting_intent if we can't parse
        return QueryClassification(query_type="betting_intent", resolved_query=None)

    query_type = parsed.get("query_type", "betting_intent")
    if query_type not in ("greeting", "vague", "betting_intent", "conversational"):
        query_type = "betting_intent"

    resolved = parsed.get("resolved_query")
    if resolved and isinstance(resolved, str) and resolved.lower() == "null":
        resolved = None

    return QueryClassification(
        query_type=query_type,
        resolved_query=resolved if query_type == "conversational" else None,
    )
