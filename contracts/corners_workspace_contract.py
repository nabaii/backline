from abc import abstractmethod
from typing import Literal

from contracts.bet_type_workspace import BetTypeWorkspace
from contracts.evidence import EvidenceSubsetImpl
from contracts.filter_spec import FilterSpec


class CornersWorkspaceContract(BetTypeWorkspace):
    """
    Contract for corners workspace implementations.
    """

    @abstractmethod
    def get_evidence(
        self,
        match_id: str,
        bet_type: str,
        filters: list[FilterSpec],
        perspective: Literal["home", "away"],
        line: float = None,
    ) -> EvidenceSubsetImpl:
        """
        Return a perspective-specific evidence subset enriched for corners analysis.
        """
        pass
