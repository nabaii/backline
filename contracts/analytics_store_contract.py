from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, List

from contracts.match_analytics import MatchAnalyticsContract
from contracts.evidence import EvidenceRequest, EvidenceSubset
from contracts.filter_spec import FilterSpec

class AnalyticsStoreContract(ABC):
    """
    central analytics query engine.
    Owns match analytics and executes evidence requests
    """

    @abstractmethod
    def ingest(self, analytics: Iterable[MatchAnalyticsContract]) -> None:
        """
        Adds MatchAnalytics objects to the store.
        Called by the analytics builder pipeline
        """
        pass

    @abstractmethod
    def query(self, request: EvidenceRequest) -> EvidenceSubset:
        """
        Executes an evidence request and returns a filtered evidence subset
        """
        pass

    @abstractmethod
    def available_features(self) -> List[str]:
        """
        All features keys available across stored matches
        """
        pass

    @abstractmethod
    def apply_filters(
        self, 
        df,
        filters: List[FilterSpec],
        request: EvidenceRequest
    ):
        """
        Applies FilterSpec rules to a dataframe.
        Returns a filtered dataframe.
        """
        pass

    @abstractmethod
    def _apply_column_filters(
        self, 
        df,
        filters: List[FilterSpec],
    ):
        """
        Applies FilterSpec rules to a dataframe.
        Returns a filtered dataframe.
        """
        pass

    @abstractmethod
    def _apply_context_filters(
        self, 
        df,
        filters: List[FilterSpec],
        request: EvidenceRequest
    ):
        """
        Applies FilterSpec rules to a dataframe.
        Returns a filtered dataframe.
        """
        pass

    @abstractmethod
    def materialize(
        self,
        perspective: str,
        required_features: List[str] | None = None
    ):
        """
        Builds a tabular dataset from MatchAnalytics for a given perspective
        """
        pass