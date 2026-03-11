"""
Hit Rate Builder: evaluates a team's betting intent across four layers:
1. Season (all matches)
2. Perspective (home/away only)
3. Rank-filtered (against opponents with similar xGD rank, ±3)
4. Combined (perspective + rank-filtered)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.backend_api import (
    _apply_opponent_rank_filters,
    _load_raw_df,
    _resolve_team_anchor_match,
    _build_store,
    _split_btts_counts,
    _split_corner_metrics,
    _split_double_chance_counts,
    _split_first_half_over_under_counts,
    _split_first_half_result_counts,
    _split_over_under_counts,
    _split_result_counts_from_goals,
    _split_win_both_halves_counts,
    _split_win_either_half_counts,
)
from backend.rag.intent_extractor import ParsedIntent
from backend.store.analytics_store import EvidenceRequest


@dataclass
class HitRateProfile:
    team_name: str
    intent: ParsedIntent

    # Layer 1: Season
    season_hits: int
    season_misses: int
    season_rate: float

    # Layer 2: Perspective (Home/Away)
    perspective: str  # "home" or "away"
    perspective_hits: int
    perspective_misses: int
    perspective_rate: float

    # Layer 3: xGD Rank-Filtered
    rank_filtered_hits: int
    rank_filtered_misses: int
    rank_filtered_rate: float

    # Layer 4: Combined (Perspective + Rank-Filtered)
    combined_hits: int
    combined_misses: int
    combined_rate: float


def _calculate_hits_misses(df: pd.DataFrame, intent: ParsedIntent) -> tuple[int, int]:
    """Helper to route an intent to the correct metric splitter."""
    if df.empty:
        return 0, 0

    bet_type = intent.bet_type
    line = intent.line or 2.5

    if bet_type == "over_under":
        metrics = _split_over_under_counts(df, line)
        return metrics["over"], metrics["under"]

    elif bet_type == "one_x_two":
        metrics = _split_result_counts_from_goals(df)
        if intent.outcome_type == "1":
            return metrics["wins"], metrics["draws"] + metrics["losses"]
        elif intent.outcome_type == "X":
            return metrics["draws"], metrics["wins"] + metrics["losses"]
        elif intent.outcome_type == "2":
            return metrics["losses"], metrics["wins"] + metrics["draws"]
        # Default to win if no outcome specified
        return metrics["wins"], metrics["draws"] + metrics["losses"]

    elif bet_type == "double_chance":
        metrics = _split_double_chance_counts(df)
        return metrics["hits"], metrics["misses"]

    elif bet_type == "corners":
        metrics = _split_corner_metrics(df, intent.line or 8.5)
        return metrics.get("over", 0), metrics.get("under", 0)

    elif bet_type == "btts":
        metrics = _split_btts_counts(df)
        if intent.outcome_type == "no":
            return metrics["misses"], metrics["hits"]
        return metrics["hits"], metrics["misses"]

    elif bet_type == "first_half_ou":
        metrics = _split_first_half_over_under_counts(df, line)
        return metrics["over"], metrics["under"]

    elif bet_type == "first_half_1x2":
        metrics = _split_first_half_result_counts(df)
        if intent.outcome_type == "1":
            return metrics["wins"], metrics["draws"] + metrics["losses"]
        elif intent.outcome_type == "X":
            return metrics["draws"], metrics["wins"] + metrics["losses"]
        elif intent.outcome_type == "2":
            return metrics["losses"], metrics["wins"] + metrics["draws"]
        return metrics["wins"], metrics["draws"] + metrics["losses"]

    elif bet_type == "win_both_halves":
        metrics = _split_win_both_halves_counts(df)
        return metrics["hits"], metrics["misses"]

    elif bet_type == "win_either_half":
        metrics = _split_win_either_half_counts(df)
        return metrics["hits"], metrics["misses"]

    return 0, 0


def build_hit_rates_for_team(
    team_name: str,
    team_id: int,
    opponent_name: str,
    opponent_xgd_rank: int | None,
    perspective: str,  # "home" or "away"
    league_id: str | None,
    intent: ParsedIntent,
) -> HitRateProfile | None:
    """
    Build hit rates across four analytical layers for the given semantic intent.
    """
    if team_id < 0:
        return None

    raw_df = _load_raw_df()
    anchor = _resolve_team_anchor_match(
        raw_df, team_id, team_name, perspective, league_id=league_id
    )
    if not anchor:
        return None

    anchor_match_id, anchor_perspective = anchor

    # 1. Fetch entire season history via workspace
    store = _build_store()
    request = EvidenceRequest(
        match_id=anchor_match_id,
        bet_type=intent.bet_type,
        perspective="all",  # Start with all to get full season
        filters={},
        required_features=[
            "total_goals",
            "goals_scored",
            "opponent_goals",
            "team_h1_goals",
            "opponent_h1_goals",
            "team_h2_goals",
            "opponent_h2_goals",
            "total_corners",
            "home_corners",
            "away_corners",
            "one_x_two_result",
            "double_chance_outcome",
        ],
        home_team_id=team_id if perspective == "home" else -1,
        away_team_id=team_id if perspective == "away" else -1,
    )
    
    try:
        full_season_df = store.query(request).df
    except Exception:
        full_season_df = pd.DataFrame()

    if full_season_df.empty:
        return None

    # Helper to calculate rate + handle div by zero
    def calc_rate(h: int, m: int) -> float:
        total = h + m
        return float(h / total) if total > 0 else 0.0

    # Layer 1: Season
    s_hits, s_misses = _calculate_hits_misses(full_season_df, intent)

    # Layer 2: Perspective
    perspective_df = full_season_df[full_season_df["venue"] == perspective]
    p_hits, p_misses = _calculate_hits_misses(perspective_df, intent)

    # Rank filtering logic
    rank_df = full_season_df
    combined_df = perspective_df

    if opponent_xgd_rank is not None:
        # Construct a synthetic filter payload mimicking the frontend selection
        filters_payload = {
            "opponent_rank_xgd_range": [
                max(1, opponent_xgd_rank - 3),
                opponent_xgd_rank + 3,
            ]
        }
        rank_df = _apply_opponent_rank_filters(
            full_season_df, filters_payload, league_id or "", side_label=perspective
        )
        combined_df = _apply_opponent_rank_filters(
            perspective_df, filters_payload, league_id or "", side_label=perspective
        )

    # Layer 3: Rank Filtered
    r_hits, r_misses = _calculate_hits_misses(rank_df, intent)

    # Layer 4: Combined
    c_hits, c_misses = _calculate_hits_misses(combined_df, intent)

    return HitRateProfile(
        team_name=team_name,
        intent=intent,
        season_hits=s_hits,
        season_misses=s_misses,
        season_rate=calc_rate(s_hits, s_misses),
        perspective=perspective,
        perspective_hits=p_hits,
        perspective_misses=p_misses,
        perspective_rate=calc_rate(p_hits, p_misses),
        rank_filtered_hits=r_hits,
        rank_filtered_misses=r_misses,
        rank_filtered_rate=calc_rate(r_hits, r_misses),
        combined_hits=c_hits,
        combined_misses=c_misses,
        combined_rate=calc_rate(c_hits, c_misses),
    )
