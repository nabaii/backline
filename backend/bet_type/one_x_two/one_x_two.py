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


class OneXTwoWorkspace(BetTypeWorkspace):
    """
    Workspace for 1x2 betting market.
    Analyzes whether a team wins (1), draws (X), or loses (2).
    """
    
    def __init__(self, store: AnalyticsStoreContract, outcome_type: Literal["1", "X", "2"] = "1"):
        """
        Initialize OneXTwoWorkspace
        
        Args:
            store: Analytics store contract
            outcome_type: Which outcome to track - "1" (win), "X" (draw), or "2" (loss)
        """
        super().__init__(store)
        
        if outcome_type not in ("1", "X", "2"):
            raise ValueError(f"outcome_type must be '1', 'X', or '2', got {outcome_type}")
        
        self.outcome_type = outcome_type
    
    # Required contract properties
    @property
    def name(self) -> str:
        return "one_x_two"
    
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
        Metrics available for 1x2 betting:
        - Hit rate: proportion of matches with the specified outcome
        - Sample size: number of games in the dataset
        """
        outcome_names = {"1": "Win", "X": "Draw", "2": "Loss"}
        outcome_label = outcome_names[self.outcome_type]
        
        return [
            HitRateMetric(
                key="one_x_two_hit_rate",
                name=f"{outcome_label} Hit Rate",
                description=f"Proportion of games where the specified outcome ({self.outcome_type}) occurred",
                outcome_column="one_x_two_outcome"
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
        Chart specification for 1x2 visualization.
        X-axis: Opponent
        Y-axis: Match outcome (1=win, 0.5=draw, 0=loss)
        """
        outcome_names = {"1": "Win", "X": "Draw", "2": "Loss"}
        outcome_label = outcome_names[self.outcome_type]
        
        return ChartSpec(
            chart_type="scatter",
            x_axis=AxisSpec(
                name="opponent_id",
                label="Opponent",
                data_column="opponent_id"
            ),
            y_axis=AxisSpec(
                name="one_x_two_result",
                label=f"Match Result ({outcome_label})",
                data_column="one_x_two_result"
            ),
            title=f"1x2: {outcome_label} Results vs Opponents",
            description=f"Scatter plot showing match outcomes ({outcome_label}) against each opponent"
        )

    def validate_filters(self, filters: List[FilterSpec]) -> None:
        """
        Validate if filters are valid for 1x2 bet type.
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
            perspective: Literal['home', 'away']
        ) -> EvidenceSubsetImpl:
        """
        Builds an evidence request and delegates execution to analytics store.
        Enriches the resulting evidence with 1x2 outcome column.
        
        Args:
            match_id: The match to get evidence for
            bet_type: Type of bet (should be 'one_x_two')
            filters: Filters to apply to the evidence
            perspective: Either 'home' or 'away'
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
        
        # Enrich evidence with 1x2 outcome column
        return self._enrich_with_one_x_two_outcome(evidence)
    
    def _enrich_with_one_x_two_outcome(
            self, 
            evidence: EvidenceSubsetImpl
        ) -> EvidenceSubsetImpl:
        """
        Add a one_x_two_outcome column to the evidence.
        Compares team's goals vs opponent's goals to determine outcome.
        
        Outcome logic:
        - 1 (Win): team_goals > opponent_goals
        - X (Draw): team_goals == opponent_goals
        - 2 (Loss): team_goals < opponent_goals
        
        The outcome_column contains 1 if the match result matches the specified
        outcome_type, 0 otherwise.
        
        Args:
            evidence: The evidence subset to enrich
        """
        df = evidence.df.copy()
        
        # Ensure one_x_two_result column is created
        if "one_x_two_result" not in df.columns:
            # If opponent_goals is not available, we'll try to infer it
            if "opponent_goals" not in df.columns:
                df["opponent_goals"] = 0
            
            if "goals_scored" in df.columns:
                # Determine actual match result for each row
                # 1 = team win, 0.5 = draw, 0 = team loss
                df["one_x_two_result"] = df.apply(
                    lambda row: self._calculate_result(row["goals_scored"], row["opponent_goals"]),
                    axis=1
                )
            else:
                # Default: all draws if goals_scored not available
                df["one_x_two_result"] = 0.5
        
        # Create outcome column: 1 if matches specified outcome_type, 0 otherwise
        df["one_x_two_outcome"] = df["one_x_two_result"].apply(
            lambda result: self._matches_outcome_type(result)
        )
        
        # Add opponent_id if not present
        if "opponent_id" not in df.columns:
            df["opponent_id"] = df.index
        
        return EvidenceSubsetImpl(
            dataframe=df,
            perspective=evidence.perspective,
            bet_type=evidence.bet_type,
            outcome_feature="one_x_two_outcome"
        )
    
    def _calculate_result(self, team_goals: int, opponent_goals: int) -> float:
        """
        Calculate the match result based on goals.
        
        Returns:
            1.0 for win, 0.5 for draw, 0.1 for loss (0.1 instead of 0.0 so bar is visible)
        """
        if team_goals > opponent_goals:
            return 1.0
        elif team_goals == opponent_goals:
            return 0.5
        else:
            return 0.1
    
    def _matches_outcome_type(self, result: float) -> int:
        """
        Check if the result matches the specified outcome type.
        
        Args:
            result: The match result (1.0 = win, 0.5 = draw, 0.1 = loss)
        
        Returns:
            1 if matches outcome_type, 0 otherwise
        """
        if self.outcome_type == "1":  # Win
            return 1 if result == 1.0 else 0
        elif self.outcome_type == "X":  # Draw
            return 1 if result == 0.5 else 0
        else:  # "2" - Loss
            return 1 if result == 0.1 else 0

