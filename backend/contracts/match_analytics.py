from abc import ABC, abstractmethod
from typing import Any, Dict, Set

class MatchAnalyticsContract(ABC):
    """
    Contract for match analytics produced for a single match.
    Immutable analytical representation
    """

    @property
    @abstractmethod
    def match_id(self) -> str:
        """
        Unique identifier for each match
        """
        pass

    @property
    @abstractmethod
    def league_id(self) -> str:
        """
        Unique identifier for each league
        """
        pass

    @property
    @abstractmethod
    def home_team_id(self) -> str:
        """
        Unique identifier for home_team
        """
        pass

    @property
    @abstractmethod
    def away_team_id(self) -> str:
        """
        Unique identifier for away team
        """
        pass

    @property
    @abstractmethod
    def available_features(self) -> Set[str]:
        """
        All feature keys available for this match
        """
        pass

    @abstractmethod
    def get_feature(self, key: str) -> Any:
        pass

    @abstractmethod
    def for_perspective(self, side: str) -> "MatchAnalyticsContract":
        """
        A perspective adjested view of the match
        side belongs to {'home', 'away'}
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialiizable representation for transport or caching
        """
        pass