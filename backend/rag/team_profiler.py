"""
Team Profiler: fetches team strength and form using xGD.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.backend_api import (
    _load_raw_df,
    _normalize_team_name,
    _opponent_rank_maps_for_league,
    _resolve_team_anchor_match,
    _build_store,
)
from backend.store.analytics_store import EvidenceRequest


@dataclass
class TeamProfile:
    team_name: str
    team_id: int
    xgd_season_rank: int | None
    xgd_last_5: float | None
    league_size: int | None


def profile_team(
    team_id: int,
    team_name: str,
    league_id: str | None,
) -> TeamProfile:
    """
    Profile a team by fetching its current xGD rank and recent xGD form.
    """
    if team_id < 0 and not team_name:
        return TeamProfile(team_name, team_id, None, None, None)

    # 1. Get xGD rank using the existing league table/rank map builder
    xgd_season_rank = None
    league_size = None
    if league_id:
        rank_maps = _opponent_rank_maps_for_league(league_id)
        if "opponent_rank_xgd_range" in rank_maps:
            xgd_map = rank_maps["opponent_rank_xgd_range"]
            league_size = len(xgd_map)
            # Use the same normalization as the rank map builder
            norm_name = _normalize_team_name(team_name)
            if norm_name in xgd_map:
                xgd_season_rank = xgd_map[norm_name]
            else:
                # Fallback: substring match
                for rank_name, rank_val in xgd_map.items():
                    if rank_name in norm_name or norm_name in rank_name:
                        xgd_season_rank = rank_val
                        break

    # 2. Get recent xGD form (last 5 matches)
    xgd_last_5 = None
    try:
        raw_df = _load_raw_df()
        anchor = _resolve_team_anchor_match(
            raw_df,
            team_id,
            team_name,
            "home",  # perspective doesn't matter for getting the full history
            league_id=league_id,
        )

        if anchor:
            anchor_match_id, anchor_perspective = anchor
            store = _build_store()
            request = EvidenceRequest(
                match_id=anchor_match_id,
                bet_type="one_x_two",  # basic bet type to get general history
                perspective=anchor_perspective,
                filters=[],
                required_features=["expected_goals_home", "expected_goals_away"],
                home_team_id=team_id if anchor_perspective == "home" else -1,
                away_team_id=team_id if anchor_perspective == "away" else -1,
            )
            df = store.query(request).df
            
            # Sort chronological and take last 5
            if not df.empty and "match_datetime" in df.columns:
                df = df.sort_values("match_datetime", ascending=False).head(5)
                
                # Calculate xGD for these matches
                # If team is home, xGD = expected_goals_home - expected_goals_away
                # If team is away, xGD = expected_goals_away - expected_goals_home
                # The generic store maps team stats correctly
                expected_goals = df["expected_goals_home"] if anchor_perspective == "home" else df["expected_goals_away"]
                opp_expected_goals = df["expected_goals_away"] if anchor_perspective == "home" else df["expected_goals_home"]
                
                xgd_series = expected_goals - opp_expected_goals
                xgd_last_5 = float(xgd_series.mean())
    except Exception:
        pass

    return TeamProfile(
        team_name=team_name,
        team_id=team_id,
        xgd_season_rank=xgd_season_rank,
        xgd_last_5=round(xgd_last_5, 2) if xgd_last_5 is not None else None,
        league_size=league_size,
    )
