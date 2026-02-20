from typing import List, Type, Literal

from backend.contracts.bet_type_workspace import BetTypeWorkspace
from backend.contracts.analytics_store_contract import AnalyticsStoreContract
from backend.contracts.filter_spec import FilterSpec
from backend.contracts.evidence import EvidenceRequest, EvidenceSubsetImpl
from backend.chart.chart_spec import ChartSpec, AxisSpec
from backend.metrics.metric_spec import MetricSpec, HitRateMetric, SampleSizeMetric

from backend.filters.filters import (
    BaseFilter,
    XGDifferenceFilter,
    VenueFilter,
    HeadToHead,
    TeamMomentumFilter,
    OpponentMomentumFilter,
    TeamPossessionFilter,
    OpponentPossessionFilter,
    FieldTiltFilter,
    TeamShotXGFilter,
    HomeTotalGoals,
    AwayTotalGoals,
    LastNGames,
    GoalsScored,
    GoalsConceded,
    TeamXG,
    OpponentXG,
    TotalMatchGoals,
)

class OverUnderWorkspace(BetTypeWorkspace):
    """
    Workspace for Over/Under betting market.
    Analyzes whether matches go over or under a specified goal line.
    """
    
    DEFAULT_OVER_UNDER_LINE = 2.5  # Default line: Over/Under 2.5 goals
    
    def __init__(self, store: AnalyticsStoreContract):
        super().__init__(store)

    # Required contract properties
    @property
    def name(self) -> str:
        return "over_under"
    
    @property
    def allowed_filters(self) -> List[Type[BaseFilter]]:
        return [
            XGDifferenceFilter,
            VenueFilter,
            HeadToHead,
            TeamMomentumFilter,
            OpponentMomentumFilter,
            TeamPossessionFilter,
            OpponentPossessionFilter,
            FieldTiltFilter,
            TeamShotXGFilter,
            TotalMatchGoals,
            HomeTotalGoals,
            AwayTotalGoals,
            LastNGames,
            GoalsScored,
            GoalsConceded,
            TeamXG,
            OpponentXG,
        ]
    
    @property
    def available_metrics(self) -> List[MetricSpec]:
        """
        Metrics available for over/under betting:
        - Hit rate: proportion of games that went over the line
        - Sample size: number of games in the dataset
        """
        return [
            HitRateMetric(
                key="over_under_hit_rate",
                name="Over Hit Rate",
                description="Proportion of games where total goals exceeded the line",
                outcome_column="over_under_outcome"
            ),
            SampleSizeMetric(
                key="sample_size",
                name="Sample Size",
                description="Number of data points in the evidence set"
            )
        ]
    
    @property
    def chart_spec(self) -> ChartSpec:
        """
        Chart specification for over/under visualization.
        X-axis: Match opponent
        Y-axis: Total goals in the match
        """
        return ChartSpec(
            chart_type="scatter",
            x_axis=AxisSpec(
                name="opponent_id",
                label="Opponent",
                data_column="opponent_id"
            ),
            y_axis=AxisSpec(
                name="total_goals",
                label="Total Goals",
                data_column="total_goals"
            ),
            title="Over/Under: Total Goals vs Opponents",
            description="Scatter plot showing total goals scored in matches against each opponent"
        )

    def validate_filters(self, filters: List[FilterSpec]) -> None:
        """
        Validate if filters are valid for over/under bet type.
        """
        allowed_keys = {cls.key for cls in self.allowed_filters}
        
        for f in filters:
            if f.key not in allowed_keys:
                raise ValueError(
                    f"Filter '{f.key}' is not allowed for {self.name}"
                )       

    def get_evidence(
            self, 
            match_id: str, 
            bet_type: str, 
            filters: List[FilterSpec], 
            perspective: Literal['home', 'away'],
            line: float = None
        ) -> EvidenceSubsetImpl:
        """
        Builds an evidence request and delegates execution to analytics store.
        Enriches the resulting evidence with over_under outcome column.
        
        Args:
            match_id: The match to get evidence for
            bet_type: Type of bet (should be 'over_under')
            filters: Filters to apply to the evidence
            perspective: Either 'home' or 'away'
            line: The over/under line (default: 2.5 goals)
        """
        self.validate_filters(filters)
        
        if line is None:
            line = self.DEFAULT_OVER_UNDER_LINE

        request = EvidenceRequest(
            match_id=match_id,
            bet_type=self.name,
            perspective=perspective,
            filters=filters,
            required_features=["total_goals"]
        )

        evidence = self.store.query(request)
        
        # Enrich evidence with over_under outcome column
        return self._enrich_with_over_under_outcome(evidence, line)
    
    def _enrich_with_over_under_outcome(
            self, 
            evidence: EvidenceSubsetImpl, 
            line: float
        ) -> EvidenceSubsetImpl:
        """
        Add an over_under_outcome column to the evidence.
        Outcome = 1 when total goals > line (over)
        Outcome = 0 when total goals <= line (under)
        
        Args:
            evidence: The evidence subset to enrich
            line: The over/under line threshold
        """
        df = evidence.df.copy()
        
        # Determine if match went over the line using the total_goals column
        df["over_under_outcome"] = (df["total_goals"] > line).astype(int)
        
        # Add opponent_id if not present
        if "opponent_id" not in df.columns:
            df["opponent_id"] = df.index
        
        return EvidenceSubsetImpl(
            dataframe=df,
            perspective=evidence.perspective,
            bet_type=evidence.bet_type,
            outcome_feature="over_under_outcome"
        )

