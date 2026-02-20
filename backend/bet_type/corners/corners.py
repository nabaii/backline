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
    VenueFilter,
    XGDifferenceFilter,
)
from backend.metrics.metric_spec import HitRateMetric, MetricSpec, SampleSizeMetric


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
        return [
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
        ]

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
    ) -> EvidenceSubsetImpl:
        self.validate_filters(filters)

        if line is None:
            line = self.DEFAULT_CORNERS_LINE

        request = EvidenceRequest(
            match_id=match_id,
            bet_type=self.name,
            perspective=perspective,
            filters=filters,
            required_features=["total_corners", "home_corners", "away_corners"],
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


