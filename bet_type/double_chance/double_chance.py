from typing import List, Type, Literal

from contracts.bet_type_workspace import BetTypeWorkspace
from contracts.analytics_store_contract import AnalyticsStoreContract
from contracts.filter_spec import FilterSpec
from contracts.evidence import EvidenceRequest, EvidenceSubsetImpl
from chart.chart_spec import ChartSpec, AxisSpec
from metrics.metric_spec import MetricSpec, HitRateMetric, SampleSizeMetric

from filters.filters import (
    BaseFilter,
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
)

class DoubleChanceWorkspace(BetTypeWorkspace):
    def __init__(self, store: AnalyticsStoreContract):
        super().__init__(store)

    # Required contract properties
    @property
    def name(self) -> str:
        return "double_chance"
    
    @property
    def allowed_filters(self) -> List[Type[BaseFilter]]:
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
    def available_metrics(self) -> List[MetricSpec]:
        """
        Metrics available for double chance:
        - Hit rate: proportion of games where team wins or draws (goals >= opponent goals)
        - Sample size: number of games in the dataset
        """
        return [
            HitRateMetric(
                key="double_chance_hit_rate",
                name="Hit Rate (Win/Draw)",
                description="Proportion of games where team wins or draws",
                outcome_column="double_chance_outcome"
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
        Chart specification for double chance visualization.
        X-axis: Opponent identifier
        Y-axis: Double chance outcome (1 = win/draw, 0 = loss)
        """
        return ChartSpec(
            chart_type="scatter",
            x_axis=AxisSpec(
                name="opponent_id",
                label="Opponent",
                data_column="opponent_id"
            ),
            y_axis=AxisSpec(
                name="double_chance_outcome",
                label="Win/Draw (1) vs Loss (0)",
                data_column="double_chance_outcome"
            ),
            title="Double Chance: Performance vs Opponents",
            description="Scatter plot showing win/draw outcomes against each opponent"
        )

    def validate_filters(self, filters: List[FilterSpec]) -> None:
        """
        Validate if filter is a valid filter
        according to the contract FilterSpec
        """
        # Build set of allowed filter keys
        allowed_keys = {cls.key for cls in self.allowed_filters}
        
        for f in filters:
            if f.key not in allowed_keys:
                raise ValueError(
                    f"Filter '{f.key}' is not allowed for {self.name}"
                )       

    # Build EvidenceRequest and delegate to store
    def get_evidence(
        self,
        match_id: str,
        bet_type: str,
        filters: List[FilterSpec],
        perspective: Literal['home', 'away']
    ) -> EvidenceSubsetImpl:
        """
        Builds an evidence request and delegates execution to analytics store.
        Enriches the resulting evidence with double_chance outcome column.
        """
        self.validate_filters(filters)

        request = EvidenceRequest(
            match_id=match_id,
            bet_type=self.name,
            perspective=perspective,
            filters=filters,
            required_features=["goals_scored", "opponent_goals"]
        )

        evidence = self.store.query(request)

        # Enrich evidence with double_chance outcome column
        return self._enrich_with_double_chance_outcome(evidence)
    
    def _enrich_with_double_chance_outcome(self, evidence: EvidenceSubsetImpl) -> EvidenceSubsetImpl:
        """
        Add a double_chance_outcome column to the evidence.
        Outcome = 1 when team's goals >= opponent's goals (win or draw)
        Outcome = 0 when team's goals < opponent's goals (loss)
        """
        df = evidence.df.copy()

        if "goals_scored" not in df.columns:
            df["goals_scored"] = 0
        if "opponent_goals" not in df.columns:
            df["opponent_goals"] = 0

        df["double_chance_outcome"] = (df["goals_scored"] >= df["opponent_goals"]).astype(int)
        df["double_chance_result"] = df["double_chance_outcome"].apply(lambda value: 1.0 if value == 1 else 0.1)
        
        # Also add opponent_id (placeholder for now)
        if "opponent_id" not in df.columns:
            df["opponent_id"] = df.index
        
        return EvidenceSubsetImpl(
            dataframe=df,
            perspective=evidence.perspective,
            bet_type=evidence.bet_type,
            outcome_feature="double_chance_outcome"
        )
