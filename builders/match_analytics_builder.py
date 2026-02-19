from typing import Dict, Any, Set, Literal
import json
import pandas as pd

from contracts.match_analytics import MatchAnalyticsContract

class MatchAnalytics(MatchAnalyticsContract):
    def __init__(
        self,
        match_id: str,
        league_id: str,
        home_team_id: str,
        away_team_id: str,
        available_features: Dict[str, Any],
    ):
        self._match_id = match_id
        self._league_id = league_id
        self._home_team_id = home_team_id
        self._away_team_id = away_team_id
        self._features = available_features.copy()          # shallow copy

    @property
    def match_id(self) -> str:
        return self._match_id

    @property
    def league_id(self) -> str:
        return self._league_id

    @property
    def home_team_id(self) -> str:
        return self._home_team_id

    @property
    def away_team_id(self) -> str:
        return self._away_team_id

    @property
    def available_features(self) -> Set[str]:
        return set(self._features.keys())           # or frozenset

    def get_feature(self, key: str) -> Any:
        return self._features[key]

    def for_perspective(self, side: Literal['home', 'away']) -> "MatchAnalytics":
        if side not in {"home", "away"}:
            raise ValueError("side must be 'home' or 'away'")

        if side == "home":
            return self

        # Better swapping logic
        swapped = {}
        for k, v in self._features.items():
            if k.startswith("home_"):
                swapped["away_" + k[5:]] = v
            elif k.startswith("away_"):
                swapped["home_" + k[5:]] = v
            else:
                swapped[k] = v

        return MatchAnalytics(
            match_id=self._match_id,
            league_id=self._league_id,
            home_team_id=self._away_team_id,
            away_team_id=self._home_team_id,
            available_features=swapped,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self._match_id,
            "league_id": self._league_id,
            "home_team_id": self._home_team_id,     # fixed
            "away_team_id": self._away_team_id,     # fixed
            "features": self._features.copy(),      # or list(self._features.items())
        }
            

class MatchAnalyticsBuilder:
    """
    Builds MatchAnalytics objects from raw match data.
    Feature engineering lives here.
    """
    def __init__(self):
        self.feature_keys = set()

    @staticmethod
    def _json_shot_payload(value: Any) -> str:
        """
        Persist nested shot payload as JSON string in season_df-derived features.
        Expected shape: {"xg": [..], "count": N}
        """
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return json.dumps({"xg": [], "count": 0}, ensure_ascii=False)

    def build(self, match_df: pd.DataFrame) -> MatchAnalytics:
        """
        Match_df: raw match data for a single match.
        Must contain at least:
        -'match_id', 'league_id', 'home_team_id', 'away_team_id'
        """

        match_id = match_df.attrs.get("match_id") or match_df.iloc[0]["match_id"]
        league_id = match_df.attrs.get("league_id") or match_df.iloc[0]["league_id"]
        home_team_id = match_df.attrs.get("home_team_id") or match_df.iloc[0]["home_team_id"]
        away_team_id = match_df.attrs.get("away_team_id") or match_df.iloc[0]["away_team_id"]

        # Feature names aligned with filter field expectations
        # Use .iloc[0] to extract scalar values (each match_df has one row per match)
        home_goals = match_df["home_normaltime"].iloc[0]
        away_goals = match_df["away_normaltime"].iloc[0]
        home_corners = pd.to_numeric(
            match_df["corner_kicks_home"].iloc[0] if "corner_kicks_home" in match_df.columns else 0,
            errors="coerce",
        )
        away_corners = pd.to_numeric(
            match_df["corner_kicks_away"].iloc[0] if "corner_kicks_away" in match_df.columns else 0,
            errors="coerce",
        )
        if pd.isna(home_corners):
            home_corners = 0.0
        if pd.isna(away_corners):
            away_corners = 0.0
        
        available_features = {
            "home_momentum": match_df["home_momentum"].iloc[0],
            "away_momentum": match_df["away_momentum"].iloc[0],
            "xg_diff": match_df["xg_diff"].iloc[0],  # For XGDifferenceFilter
            "xg_diff_home": match_df["xg_diff_home"].iloc[0],
            "xg_diff_away": match_df["xg_diff_away"].iloc[0],
            "total_goals": match_df["total_goals"].iloc[0],
            "home_corners": float(home_corners),
            "away_corners": float(away_corners),
            "total_corners": float(home_corners) + float(away_corners),
            "home_normaltime": home_goals,  # For HomeTotalGoals
            "away_normaltime": away_goals,  # For AwayTotalGoals
            "expected_goals_home": match_df["expected_goals_home"].iloc[0],
            "expected_goals_away": match_df["expected_goals_away"].iloc[0],
            "ball_possession_home": match_df["ball_possession_home"].iloc[0],
            "ball_possession_away": match_df["ball_possession_away"].iloc[0],
            "field_tilt_home": match_df["field_tilt_home"].iloc[0],
            "field_tilt_away": match_df["field_tilt_away"].iloc[0],
            "home_shots": self._json_shot_payload(
                match_df["home_shots"].iloc[0] if "home_shots" in match_df.columns else None
            ),
            "away_shots": self._json_shot_payload(
                match_df["away_shots"].iloc[0] if "away_shots" in match_df.columns else None
            ),
            # Perspective-aware goals_scored: swapped when for_perspective("away") is called
            "home_goals_scored": home_goals,  # For GoalsScored filter (home perspective)
            "away_goals_scored": away_goals,  # For GoalsScored filter (away perspective)
        }

        self.feature_keys.update(available_features.keys())

        return MatchAnalytics(
            match_id=match_id,
            league_id=league_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            available_features=available_features
        )
    
    def build_many(self, matches: list[pd.DataFrame]) -> list[MatchAnalytics]:
        return [self.build(m) for m in matches]
    
