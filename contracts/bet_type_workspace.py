from typing import List, Any, Literal, Dict, Type
from abc import ABC, abstractmethod

from contracts.filter_spec import FilterSpec
from contracts.evidence import EvidenceRequest
from contracts.analytics_store_contract import AnalyticsStoreContract
from chart.chart_spec import ChartSpec
from metrics.metric_spec import MetricSpec

class BetTypeWorkspace(ABC):

    """
    Contract for all betting market workspaces
    (Double Chance, BTTS, Over/Under, 1X2, etc.)
    """
    def __init__(self, store: AnalyticsStoreContract):
        self.store = store

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human readable bet type name
        e.g. 'Double Chance'
        """
        pass

    @property
    @abstractmethod
    def allowed_filters(self) -> List[Type]:
        """
        A list of all the allowed filter classes
        e.g. [VenueFilter, MomentumFilter]
        """
        pass

    @abstractmethod
    def validate_filters(self, filters: List[FilterSpec]) -> None:
        """
        Validate if filter is a valid filter
        according to the contract FilterSpec
        """
        pass

    @property
    @abstractmethod
    def available_metrics(self) -> List[MetricSpec]:
        """
        Metrics that can be computed from evidence
        e.g. hit rate, sample size
        """
        pass

    @property
    @abstractmethod
    def chart_spec(self) -> ChartSpec:
        """
        Declarative chart definition for this bet type.
        Specifies how to visualize outcomes.
        """
        pass

    @abstractmethod
    def get_evidence(
        self,
        match_id: str,
        bet_type: str,
        filters: List[FilterSpec],
        perspective: Literal['home', 'away']
    ) -> EvidenceRequest:
        """
        Builds an evidence request and delegates execution to analytics store
        """
        pass
