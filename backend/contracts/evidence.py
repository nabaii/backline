from typing import List, Literal, Optional, Any, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

from backend.contracts.filter_spec import FilterSpec

@dataclass(frozen=True)
class EvidenceRequest():
    match_id: int
    filters: List[FilterSpec]
    perspective: Literal['home', 'away']
    bet_type: str
    required_features: Optional[List[str]] = None
    time_scope: Optional[Any] = None

    def validate_self(self) -> None:
        """
        Validate if filter is a valid filter
        """
        if self.perspective not in {'home', 'away'}:
            raise ValueError('Perspective must be home or away')
        
        for f in self.filters:
            f.validate()

    def to_dict(self) -> dict:
        return{
            "match_id": self.match_id,
            "perspective": self.perspective,
            "filters": [f.to_dict() for f in self.filters],
            "required_features": self.required_features,
            "bet_type": self.bet_type,
            "time_scope": self.time_scope
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EvidenceRequest":
        return cls(
            match_id=payload["match_id"],
            perspective=payload["perspective"],
            bet_type=payload["bet_type"],
            filters=[
                FilterSpec.from_dict(f) for f in payload.get('filters', [])
            ],
            required_features=payload.get("required_features"),
            time_scope=payload.get("time_scope")
        )


class EvidenceSubset(ABC):
    """
    Read-only wrapper around filtered tabular dataset
    """

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """
        Internal filtered dataframe.
        Must be treated as read-only
        """
        pass

    @property
    @abstractmethod
    def sample_size(self) -> int:
        pass

    @property
    @abstractmethod
    def perspective(self) -> str:
        pass

    @property
    @abstractmethod
    def bet_type(self) -> str:
        pass

    @property
    @abstractmethod
    def outcome_feature(self) -> str:
        pass

    # Helpers
    def columns(self) -> List[str]:
        return list(self.df.columns)
    
    def rows(self) -> List[Dict[str, Any]]:
        return self.df.to_dict(orient="records")


class EvidenceSubsetImpl(EvidenceSubset):
    """
    Concrete implementation of EvidenceSubset
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        perspective: str,
        bet_type: str,
        outcome_feature: Optional[str] = None
    ):
        self._df = dataframe
        self._perspective = perspective
        self._bet_type = bet_type
        self._outcome_feature = outcome_feature

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def sample_size(self) -> int:
        return len(self._df)

    @property
    def perspective(self) -> str:
        return self._perspective

    @property
    def bet_type(self) -> str:
        return self._bet_type

    @property
    def outcome_feature(self) -> str:
        return self._outcome_feature or ""
