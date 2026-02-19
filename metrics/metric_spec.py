from typing import Any, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
from contracts.evidence import EvidenceSubset


@dataclass(frozen=True)
class MetricSpec:
    """
    Specification for a metric that can be computed from evidence.
    """
    key: str
    name: str
    description: Optional[str] = None
    
    def compute(self, evidence: "EvidenceSubset") -> Any:
        """
        Override in subclasses or use compute_fn
        """
        raise NotImplementedError


@dataclass(frozen=True)
class HitRateMetric(MetricSpec):
    """Computes the hit rate (success rate) from evidence outcomes."""
    outcome_column: str = "outcome"
    
    def compute(self, evidence: "EvidenceSubset") -> float:
        """Calculate hit rate: (count of 1s) / (total count)"""
        if evidence.sample_size == 0:
            return 0.0
        
        df = evidence.df
        if self.outcome_column not in df.columns:
            raise ValueError(f"Outcome column '{self.outcome_column}' not found in evidence")
        
        hits = (df[self.outcome_column] == 1).sum()
        return hits / len(df) if len(df) > 0 else 0.0


@dataclass(frozen=True)
class SampleSizeMetric(MetricSpec):
    """Returns the sample size of the evidence."""
    
    def compute(self, evidence: "EvidenceSubset") -> int:
        """Return the sample size"""
        return evidence.sample_size
