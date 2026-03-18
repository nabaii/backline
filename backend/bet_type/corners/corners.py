from typing import Literal, Type

from backend.chart.chart_spec import AxisSpec, ChartSpec
from backend.contracts.analytics_store_contract import AnalyticsStoreContract
from backend.contracts.corners_workspace_contract import CornersWorkspaceContract
from backend.contracts.evidence import EvidenceRequest, EvidenceSubsetImpl
from backend.contracts.filter_spec import FilterSpec
from backend.filters.filters import (
    BaseFilter,
    FieldTiltFilter,
    GoalsConceded,
    GoalsScored,
    HeadToHead,
    LastNGames,
    OpponentMomentumFilter,
    OpponentPossessionFilter,
    OpponentXG,
    TeamMomentumFilter,
    TeamPossessionFilter,
    TeamShotXGFilter,
    TeamXG,
    TotalMatchGoals,
    TotalXG,
    VenueFilter,
    XGDifferenceFilter,
)
from backend.metrics.metric_spec import HitRateMetric, MetricSpec, SampleSizeMetric


_CORNERS_FILTERS: list[Type[BaseFilter]] = [
    XGDifferenceFilter,
    VenueFilter,
    HeadToHead,
    TeamMomentumFilter,
    OpponentMomentumFilter,
    TeamXG,
    OpponentXG,
    TeamPossessionFilter,
    OpponentPossessionFilter,
    FieldTiltFilter,
    TeamShotXGFilter,
    LastNGames,
    GoalsScored,
    GoalsConceded,
    TotalMatchGoals,
    TotalXG,
]

_CORNERS_REQUIRED_FEATURES = [
    "total_corners", "home_corners", "away_corners", "goals_scored", "opponent_goals",
]


class CornerWorkspace(CornersWorkspaceContract):
    """
    Workspace for total-corners analysis.
    Analyzes whether matches go over or under a specified corners line.
    """

    DEFAULT_CORNERS_LINE = 8.5

    def __init__(self, store: AnalyticsStoreContract):
        super().__init__(store)

    @property
    def name(self) -> str:
        return "corners"

    @property
    def allowed_filters(self) -> list[Type[BaseFilter]]:
        return _CORNERS_FILTERS

    @property
    def available_metrics(self) -> list[MetricSpec]:
        return [
            HitRateMetric(
                key="corners_hit_rate",
                name="Corners Over Hit Rate",
                description="Proportion of games where total corners exceeded the line",
                outcome_column="corners_outcome",
            ),
            SampleSizeMetric(
                key="sample_size",
                name="Sample Size",
                description="Number of matches in the corners evidence set",
            ),
        ]

    @property
    def chart_spec(self) -> ChartSpec:
        return ChartSpec(
            chart_type="bar",
            x_axis=AxisSpec(
                name="opponent_id",
                label="Opponent",
                data_column="opponent_id",
            ),
            y_axis=AxisSpec(
                name="total_corners",
                label="Total Corners",
                data_column="total_corners",
            ),
            title="Total Corners by Match",
            description="Bar chart of total corners (home + away) per match",
        )

    def validate_filters(self, filters: list[FilterSpec]) -> None:
        allowed_keys = {cls.key for cls in self.allowed_filters}
        for f in filters:
            if f.key not in allowed_keys:
                raise ValueError(f"Filter '{f.key}' is not allowed for {self.name}")

    def get_evidence(
        self,
        match_id: str,
        bet_type: str,
        filters: list[FilterSpec],
        perspective: Literal["home", "away"],
        line: float = None,
        home_team_id: int = None,
        away_team_id: int = None,
    ) -> EvidenceSubsetImpl:
        self.validate_filters(filters)

        if line is None:
            line = self.DEFAULT_CORNERS_LINE

        request = EvidenceRequest(
            match_id=match_id,
            bet_type=self.name,
            perspective=perspective,
            filters=filters,
            required_features=_CORNERS_REQUIRED_FEATURES,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        evidence = self.store.query(request)
        return self._enrich_with_corners_outcome(evidence, line)

    def _enrich_with_corners_outcome(
        self, evidence: EvidenceSubsetImpl, line: float
    ) -> EvidenceSubsetImpl:
        """
        Add corners_outcome and ensure total_corners is present.
        Outcome = 1 when total_corners > line (over)
        Outcome = 0 when total_corners <= line (under)
        """
        df = evidence.df.copy()

        if "total_corners" in df.columns:
            df["total_corners"] = df["total_corners"].astype(float)
        else:
            home_corners = df["home_corners"].astype(float) if "home_corners" in df.columns else 0.0
            away_corners = df["away_corners"].astype(float) if "away_corners" in df.columns else 0.0
            df["total_corners"] = home_corners + away_corners

        df["corners_outcome"] = (df["total_corners"] > line).astype(int)

        if "opponent_id" not in df.columns:
            df["opponent_id"] = df.index

        return EvidenceSubsetImpl(
            dataframe=df,
            perspective=evidence.perspective,
            bet_type=evidence.bet_type,
            outcome_feature="corners_outcome",
        )


class HomeCornerWorkspace(CornersWorkspaceContract):
    """
    Workspace for home-team corners analysis.
    Analyzes whether the home team's corners go over or under a specified line.
    """

    DEFAULT_CORNERS_LINE = 4.5

    def __init__(self, store: AnalyticsStoreContract):
        super().__init__(store)

    @property
    def name(self) -> str:
        return "home_corners"

    @property
    def allowed_filters(self) -> list[Type[BaseFilter]]:
        return _CORNERS_FILTERS

    @property
    def available_metrics(self) -> list[MetricSpec]:
        return [
            HitRateMetric(
                key="corners_hit_rate",
                name="Home Corners Over Hit Rate",
                description="Proportion of games where home corners exceeded the line",
                outcome_column="corners_outcome",
            ),
            SampleSizeMetric(
                key="sample_size",
                name="Sample Size",
                description="Number of matches in the home corners evidence set",
            ),
        ]

    @property
    def chart_spec(self) -> ChartSpec:
        return ChartSpec(
            chart_type="bar",
            x_axis=AxisSpec(name="opponent_id", label="Opponent", data_column="opponent_id"),
            y_axis=AxisSpec(name="home_corners", label="Home Corners", data_column="home_corners"),
            title="Home Corners by Match",
            description="Bar chart of home-team corners per match",
        )

    def validate_filters(self, filters: list[FilterSpec]) -> None:
        allowed_keys = {cls.key for cls in self.allowed_filters}
        for f in filters:
            if f.key not in allowed_keys:
                raise ValueError(f"Filter '{f.key}' is not allowed for {self.name}")

    def get_evidence(
        self,
        match_id: str,
        bet_type: str,
        filters: list[FilterSpec],
        perspective: Literal["home", "away"],
        line: float = None,
        home_team_id: int = None,
        away_team_id: int = None,
    ) -> EvidenceSubsetImpl:
        self.validate_filters(filters)
        if line is None:
            line = self.DEFAULT_CORNERS_LINE

        request = EvidenceRequest(
            match_id=match_id,
            bet_type=self.name,
            perspective=perspective,
            filters=filters,
            required_features=_CORNERS_REQUIRED_FEATURES,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        evidence = self.store.query(request)
        return self._enrich(evidence, line)

    def _enrich(self, evidence: EvidenceSubsetImpl, line: float) -> EvidenceSubsetImpl:
        df = evidence.df.copy()
        if "home_corners" in df.columns:
            df["home_corners"] = df["home_corners"].astype(float)
        else:
            df["home_corners"] = 0.0
        df["corners_outcome"] = (df["home_corners"] > line).astype(int)
        if "opponent_id" not in df.columns:
            df["opponent_id"] = df.index
        return EvidenceSubsetImpl(
            dataframe=df,
            perspective=evidence.perspective,
            bet_type=evidence.bet_type,
            outcome_feature="corners_outcome",
        )


class AwayCornerWorkspace(CornersWorkspaceContract):
    """
    Workspace for away-team corners analysis.
    Analyzes whether the away team's corners go over or under a specified line.
    """

    DEFAULT_CORNERS_LINE = 4.5

    def __init__(self, store: AnalyticsStoreContract):
        super().__init__(store)

    @property
    def name(self) -> str:
        return "away_corners"

    @property
    def allowed_filters(self) -> list[Type[BaseFilter]]:
        return _CORNERS_FILTERS

    @property
    def available_metrics(self) -> list[MetricSpec]:
        return [
            HitRateMetric(
                key="corners_hit_rate",
                name="Away Corners Over Hit Rate",
                description="Proportion of games where away corners exceeded the line",
                outcome_column="corners_outcome",
            ),
            SampleSizeMetric(
                key="sample_size",
                name="Sample Size",
                description="Number of matches in the away corners evidence set",
            ),
        ]

    @property
    def chart_spec(self) -> ChartSpec:
        return ChartSpec(
            chart_type="bar",
            x_axis=AxisSpec(name="opponent_id", label="Opponent", data_column="opponent_id"),
            y_axis=AxisSpec(name="away_corners", label="Away Corners", data_column="away_corners"),
            title="Away Corners by Match",
            description="Bar chart of away-team corners per match",
        )

    def validate_filters(self, filters: list[FilterSpec]) -> None:
        allowed_keys = {cls.key for cls in self.allowed_filters}
        for f in filters:
            if f.key not in allowed_keys:
                raise ValueError(f"Filter '{f.key}' is not allowed for {self.name}")

    def get_evidence(
        self,
        match_id: str,
        bet_type: str,
        filters: list[FilterSpec],
        perspective: Literal["home", "away"],
        line: float = None,
        home_team_id: int = None,
        away_team_id: int = None,
    ) -> EvidenceSubsetImpl:
        self.validate_filters(filters)
        if line is None:
            line = self.DEFAULT_CORNERS_LINE

        request = EvidenceRequest(
            match_id=match_id,
            bet_type=self.name,
            perspective=perspective,
            filters=filters,
            required_features=_CORNERS_REQUIRED_FEATURES,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        evidence = self.store.query(request)
        return self._enrich(evidence, line)

    def _enrich(self, evidence: EvidenceSubsetImpl, line: float) -> EvidenceSubsetImpl:
        df = evidence.df.copy()
        if "away_corners" in df.columns:
            df["away_corners"] = df["away_corners"].astype(float)
        else:
            df["away_corners"] = 0.0
        df["corners_outcome"] = (df["away_corners"] > line).astype(int)
        if "opponent_id" not in df.columns:
            df["opponent_id"] = df.index
        return EvidenceSubsetImpl(
            dataframe=df,
            perspective=evidence.perspective,
            bet_type=evidence.bet_type,
            outcome_feature="corners_outcome",
        )


