"""
Converts season_df.csv rows into text documents for RAG ingestion.

Each document is a human-readable match summary with rich metadata
so Chroma can filter by team, date, league, etc. before semantic search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatchDocument:
    """A single match encoded as text plus filterable metadata."""

    doc_id: str                      # unique id — game_id as string
    text: str                        # the passage that gets embedded
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe(value: Any, decimals: int = 2) -> str:
    """Return a tidy string for a value that may be NaN/None."""
    if value is None:
        return "N/A"
    try:
        f = float(value)
        if math.isnan(f):
            return "N/A"
        return str(round(f, decimals)) if decimals else str(int(f))
    except (TypeError, ValueError):
        return str(value)


def _winner_label(row: dict) -> str:
    home = row.get("home_team", "Home")
    away = row.get("away_team", "Away")
    hw = row.get("home_normaltime")
    aw = row.get("away_normaltime")
    try:
        if float(hw) > float(aw):
            return f"{home} win"
        if float(aw) > float(hw):
            return f"{away} win"
        return "Draw"
    except (TypeError, ValueError):
        return "N/A"


def _build_text(row: dict) -> str:
    home = str(row.get("home_team", "")).replace("_", " ").title()
    away = str(row.get("away_team", "")).replace("_", " ").title()
    date = str(row.get("match_datetime", ""))[:10]
    league = str(row.get("league_name", ""))

    home_goals = _safe(row.get("home_normaltime"), 0)
    away_goals = _safe(row.get("away_normaltime"), 0)
    result = f"{home} {home_goals} - {away_goals} {away}"

    return f"""Match: {home} vs {away}
            Date: {date}
            League: {league}
            Result: {result}
            Winner: {_winner_label(row)}

            Possession: {home} {_safe(row.get('ball_possession_home'))}% | {away} {_safe(row.get('ball_possession_away'))}%
            Expected Goals (xG): {home} {_safe(row.get('expected_goals_home'))} | {away} {_safe(row.get('expected_goals_away'))}
            Total Goals: {_safe(row.get('total_goals'), 0)}
            Total Shots: {home} {_safe(row.get('total_shots_home'), 0)} | {away} {_safe(row.get('total_shots_away'), 0)}
            Shots on Target: {home} {_safe(row.get('shots_on_target_home'), 0)} | {away} {_safe(row.get('shots_on_target_away'), 0)}
            Big Chances: {home} {_safe(row.get('big_chances_home'), 0)} | {away} {_safe(row.get('big_chances_away'), 0)}
            Corners: {home} {_safe(row.get('corner_kicks_home'), 0)} | {away} {_safe(row.get('corner_kicks_away'), 0)}
            Fouls: {home} {_safe(row.get('fouls_home'), 0)} | {away} {_safe(row.get('fouls_away'), 0)}
            Yellow Cards: {home} {_safe(row.get('yellow_cards_home'), 0)} | {away} {_safe(row.get('yellow_cards_away'), 0)}
            Field Tilt: {home} {_safe(row.get('field_tilt_home'))} | {away} {_safe(row.get('field_tilt_away'))}
            Momentum: {home} {_safe(row.get('home_momentum'))} | {away} {_safe(row.get('away_momentum'))}"""


def build_documents(rows: list[dict]) -> list[MatchDocument]:
    """
    Convert a list of season_df row dicts into MatchDocument objects.

    Parameters
    ----------
    rows:
        Each dict is one row from season_df.csv (e.g. from pandas to_dict('records')).

    Returns
    -------
    List of MatchDocument ready for ingestion into the vector store.
    """
    docs: list[MatchDocument] = []

    for row in rows:
        game_id = str(row.get("game_id", ""))
        if not game_id:
            continue

        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        date_str = str(row.get("match_datetime", ""))[:10]
        league_name = str(row.get("league_name", ""))

        metadata: dict[str, Any] = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_id": str(row.get("home_team_id", "")),
            "away_team_id": str(row.get("away_team_id", "")),
            "match_date": date_str,
            "league_name": league_name,
            "league_id": str(row.get("league_id", "")),
            "home_goals": _safe(row.get("home_normaltime"), 0),
            "away_goals": _safe(row.get("away_normaltime"), 0),
            "total_goals": _safe(row.get("total_goals"), 0),
        }

        docs.append(MatchDocument(
            doc_id=game_id,
            text=_build_text(row),
            metadata=metadata,
        ))

    return docs
