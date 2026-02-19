from typing import Iterable, List, Literal, Any
import ast
import json
import pandas as pd

from contracts.analytics_store_contract import AnalyticsStoreContract
from builders.match_analytics_builder import MatchAnalytics
from contracts.evidence import EvidenceRequest, EvidenceSubsetImpl
from contracts.filter_spec import FilterSpec


class AnalyticsStore(AnalyticsStoreContract):
    def __init__(self):
        self._matches: dict[str, MatchAnalytics] = {}

    def ingest(self, analytics: Iterable[MatchAnalytics]) -> None:
        for ma in analytics:
            self._matches[ma.match_id] = ma

    def query(self, request: EvidenceRequest) -> "EvidenceSubsetImpl":
        """Query team games based on match_id, perspective, and filters."""
        if request.match_id not in self._matches:
            raise KeyError(f"Match {request.match_id} not found")

        team_id = self._get_team_id(request.match_id, request.perspective)
        required_cols = self._get_required_columns(request.filters)
        df = self._materialize_team_games(team_id, required_cols, request.required_features)
        filtered_df = self._filter_dataframe(df, request.filters, request)

        return EvidenceSubsetImpl(
            dataframe=filtered_df,
            perspective=request.perspective,
            bet_type=request.bet_type,
            outcome_feature=request.required_features[0] if request.required_features else None
        )

    def _get_team_id(self, match_id: int, perspective: str) -> str:
        """Get the team ID based on match_id and perspective."""
        match = self._matches[match_id]
        return match.home_team_id if perspective == "home" else match.away_team_id

    def _get_required_columns(self, filters: List[FilterSpec]) -> set:
        """Collect all required columns from filters."""
        columns = {"match_id", "team_id", "venue", "home_team_id", "away_team_id"}  # Base columns always included
        for f in filters:
            if f.required_columns:
                columns.update(f.required_columns)
            if f.field:
                columns.add(f.field)
        return columns

    def _materialize_team_games(
        self,
        team_id: str,
        required_columns: set,
        extra_features: List[str] | None = None
    ) -> pd.DataFrame:
        """Materialize all games where the team played (home or away)."""
        rows = []
        for match_id, ma in self._matches.items():
            row = None
            
            if ma.home_team_id == team_id:
                row = self._build_row(match_id, team_id, "home", ma, required_columns, extra_features)
            elif ma.away_team_id == team_id:
                row = self._build_row(match_id, team_id, "away", ma, required_columns, extra_features)
            
            if row:
                rows.append(row)

        return pd.DataFrame(rows)

    def _build_row(
        self,
        match_id: int,
        team_id: str,
        venue: str,
        ma: MatchAnalytics,
        required_columns: set,
        extra_features: List[str] | None
    ) -> dict:
        """Build a row dict for a single match."""
        goals_col = "home_normaltime" if venue == "home" else "away_normaltime"
        opponent_goals_col = "away_normaltime" if venue == "home" else "home_normaltime"
        
        row = {
            "match_id": match_id,
            "team_id": team_id,
            "venue": venue,
            "home_team_id": ma.home_team_id,  # Always preserve actual home team
            "away_team_id": ma.away_team_id,  # Always preserve actual away team
        }
        
        # Add goals_scored if needed
        # Check both required_columns and extra_features
        if "goals_scored" in required_columns or (extra_features and "goals_scored" in extra_features):
            row["goals_scored"] = ma.get_feature(goals_col)
        
        # Add opponent_goals if needed (opposite team's goals)
        # Check both required_columns and extra_features
        if "opponent_goals" in required_columns or (extra_features and "opponent_goals" in extra_features):
            row["opponent_goals"] = ma.get_feature(opponent_goals_col)

        # Add required feature columns used by filters.
        for feat in required_columns:
            if feat in row or feat in ("goals_scored", "opponent_goals"):
                continue
            if feat in ma.available_features:
                row[feat] = ma.get_feature(feat)
        
        # Add extra features if specified
        if extra_features:
            for feat in extra_features:
                # Skip opponent_goals and goals_scored as they are handled above
                if feat in ("opponent_goals", "goals_scored"):
                    continue
                if feat in ma.available_features:
                    row[feat] = ma.get_feature(feat)
        
        return row
    
    def _filter_dataframe(
        self, 
        df: pd.DataFrame, 
        filters: List[FilterSpec], 
        request: EvidenceRequest
    ) -> pd.DataFrame:
        """Apply all filters to the dataframe, delegating by filter kind."""
        if not filters:
            return df.copy()

        filtered_df = df.copy()
        
        # Group filters by kind and apply
        for f in filters:
            if f.kind == "column":
                filtered_df = self._apply_column_filter(filtered_df, f)
            elif f.kind == "context":
                filtered_df = self._apply_context_filter(filtered_df, f, request)
        
        return filtered_df
    
    def _apply_column_filter(self, df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
        """Apply a single column filter to the dataframe."""
        try:
            spec.validate()
        except ValueError as e:
            raise ValueError(f"Invalid filter spec: {spec} -> {e}")

        if spec.key == "team_shot_xg":
            return self._apply_team_shot_xg_filter(df, spec)

        col = self._get_filter_series(df, spec)
        if col is None:
            return df.copy()
        return self._apply_operator_filter(df, col, spec)

    def _parse_shot_payload(self, payload: Any) -> list[float]:
        if payload is None:
            return []

        parsed = payload
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    return []

        if isinstance(parsed, dict):
            xg_values = parsed.get("xg", [])
            if not isinstance(xg_values, list):
                return []
            out: list[float] = []
            for val in xg_values:
                try:
                    out.append(float(val))
                except (TypeError, ValueError):
                    continue
            return out

        if isinstance(parsed, list):
            out: list[float] = []
            for val in parsed:
                if isinstance(val, dict) and "xg" in val:
                    try:
                        out.append(float(val["xg"]))
                    except (TypeError, ValueError):
                        continue
                else:
                    try:
                        out.append(float(val))
                    except (TypeError, ValueError):
                        continue
            return out

        return []

    def _apply_team_shot_xg_filter(self, df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
        required = {"venue", "home_shots", "away_shots"}
        if not required.issubset(df.columns):
            return df.copy()

        value = spec.value if isinstance(spec.value, dict) else {}
        try:
            min_xg = float(value.get("min_xg", 0.0))
            min_shots = int(value.get("min_shots", 0))
        except (TypeError, ValueError):
            return df.copy()

        if min_shots <= 0:
            return df.copy()

        rows_with_shot_payload = 0

        def count_qualifying_shots(row: pd.Series) -> int:
            nonlocal rows_with_shot_payload
            venue = str(row.get("venue", ""))
            payload = row.get("home_shots") if venue == "home" else row.get("away_shots")
            xg_values = self._parse_shot_payload(payload)
            if xg_values:
                rows_with_shot_payload += 1
            return int(sum(1 for xg in xg_values if xg >= min_xg))

        counts = df.apply(count_qualifying_shots, axis=1)
        # If no row has shot payload data, do not wipe the dataset.
        # This keeps behavior safe until shot payloads are backfilled.
        if rows_with_shot_payload == 0:
            return df.copy()
        return df[counts >= min_shots]

    def _get_filter_series(self, df: pd.DataFrame, spec: FilterSpec) -> pd.Series | None:
        """
        Resolve the effective series for a filter.
        Momentum/xG filters are venue-aware per row.
        """
        if spec.key in {"team_momentum", "opponent_momentum"}:
            required = {"venue", "home_momentum", "away_momentum"}
            if not required.issubset(df.columns):
                return None

            if spec.key == "team_momentum":
                return df["home_momentum"].where(df["venue"] == "home", df["away_momentum"])
            return df["away_momentum"].where(df["venue"] == "home", df["home_momentum"])

        if spec.key in {"team_xg", "opponent_xg"}:
            required = {"venue", "expected_goals_home", "expected_goals_away"}
            if not required.issubset(df.columns):
                return None

            if spec.key == "team_xg":
                return df["expected_goals_home"].where(df["venue"] == "home", df["expected_goals_away"])
            return df["expected_goals_away"].where(df["venue"] == "home", df["expected_goals_home"])

        if spec.key in {"team_possession", "opponent_possession"}:
            required = {"venue", "ball_possession_home", "ball_possession_away"}
            if not required.issubset(df.columns):
                return None

            if spec.key == "team_possession":
                return df["ball_possession_home"].where(df["venue"] == "home", df["ball_possession_away"])
            return df["ball_possession_away"].where(df["venue"] == "home", df["ball_possession_home"])

        if spec.key == "field_tilt":
            required = {"venue", "field_tilt_home", "field_tilt_away"}
            if not required.issubset(df.columns):
                return None
            return df["field_tilt_home"].where(df["venue"] == "home", df["field_tilt_away"])

        field = spec.field
        if not field or field not in df.columns:
            return None
        return df[field]

    def _apply_operator_filter(self, df: pd.DataFrame, col: pd.Series, spec: FilterSpec) -> pd.DataFrame:
        """Apply filter operator logic to a resolved series."""
        match spec.operator:
            case '>':
                return df[col > spec.value]
            case '>=':
                return df[col >= spec.value]
            case '<':
                return df[col < spec.value]
            case '<=':
                return df[col <= spec.value]
            case '==':
                return df[col == spec.value]
            case '!=':
                return df[col != spec.value]
            case 'between':
                min_val, max_val = spec.value
                return df[(col >= min_val) & (col <= max_val)]
            case 'in':
                return df[col.isin(spec.value)]
            case _:
                raise ValueError(f"Unsupported operator: {spec.operator}")

    def _apply_context_filter(self, df: pd.DataFrame, spec: FilterSpec, request: EvidenceRequest) -> pd.DataFrame:
        """Apply a single context filter to the dataframe."""
        try:
            spec.validate()
        except ValueError as e:
            raise ValueError(f"Invalid filter spec: {spec} -> {e}")

        if spec.key == 'head_to_head':
            return self._apply_head_to_head_filter(df, request)
        elif spec.key == 'last_n_games':
            return df.tail(spec.value)

        return df.copy()

    def _apply_head_to_head_filter(self, df: pd.DataFrame, request: EvidenceRequest) -> pd.DataFrame:
        """Filter to head-to-head matches between two teams."""
        match_row = df[df['match_id'] == request.match_id]
        if match_row.empty:
            return df.copy()
        
        home_team = match_row['home_team_id'].iloc[0]
        away_team = match_row['away_team_id'].iloc[0]
        
        return df[
            ((df['home_team_id'] == home_team) & (df['away_team_id'] == away_team))
            | 
            ((df['home_team_id'] == away_team) & (df['away_team_id'] == home_team))
        ]
    
    def apply_filters(
            self, 
            df: pd.DataFrame, 
            filters: List[FilterSpec], 
            request: EvidenceRequest
        ) -> pd.DataFrame:

        if not filters:
            return df.copy()

        # Separate filters by kind
        column_filters = [f for f in filters if f.kind == "column"]
        context_filters = [f for f in filters if f.kind == "context"]

        filtered_df = df.copy()

        # Apply column filters first
        if column_filters:
            filtered_df = self._apply_column_filters(filtered_df, column_filters)

        # Apply context filters 
        if context_filters:
            filtered_df = self._apply_context_filters(filtered_df, context_filters, request)

        return filtered_df
            

    def _apply_column_filters(
        self,
        df: pd.DataFrame,
        filters: List[FilterSpec]
    ) -> pd.DataFrame:
        """
        Applies a list of FilterSpec rules to a pandas DataFrame.
        Returns the filtered DataFrame.
        """
        if not filters:
            return df.copy()

        filtered_df = df.copy()  # avoid mutating original

        for spec in filters:
            try:
                spec.validate()  # enforce structure
            except ValueError as e:
                raise ValueError(f"Invalid filter spec: {spec} -> {e}")

            col = self._get_filter_series(filtered_df, spec)
            if col is None:
                continue

            filtered_df = self._apply_operator_filter(filtered_df, col, spec)

        return filtered_df

    def _apply_context_filters(
        self,
        df: pd.DataFrame,
        filters: List[FilterSpec],
        request: EvidenceRequest
    ) -> pd.DataFrame:
        if not filters:
            return df.copy()

        filtered_df = df.copy()  # avoid mutating original

        for spec in filters:
            try:
                spec.validate()  # enforce structure
            except ValueError as e:
                raise ValueError(f"Invalid filter spec: {spec} -> {e}")

            if spec.key == 'head_to_head':
                match_row = filtered_df[filtered_df['match_id'] == request.match_id]
                if not match_row.empty:
                    home_team = match_row['home_team_id'].iloc[0]
                    away_team = match_row['away_team_id'].iloc[0]

                    filtered_df = filtered_df[
                        ((filtered_df['home_team_id'] == home_team) & (filtered_df['away_team_id'] == away_team))
                        |
                        ((filtered_df['home_team_id'] == away_team) & (filtered_df['away_team_id'] == home_team))
                    ]

            if spec.key == "last_n_games":
                length = spec.value
                filtered_df = filtered_df.tail(length)

        return filtered_df

    def available_features(self) -> List[str]:
        feats = set()
        for ma in self._matches.values():
            feats.update(ma.available_features)
        return sorted(feats)

    def materialize(
        self,
        perspective: Literal["home", "away"],
        required_features: List[str] | None = None
    ) -> pd.DataFrame:
        rows = []
        for ma in self._matches.values():
            view = ma.for_perspective(perspective)
            row = {"match_id": view.match_id, "league_id": view.league_id}
            feats = required_features or list(view.available_features)
            for f in feats:
                row[f] = view.get_feature(f) if f in view.available_features else None
            rows.append(row)
        return pd.DataFrame(rows)




