from typing import Literal, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisSpec:
    """Specification for a single axis in a chart"""
    name: str
    label: str
    data_column: Optional[str] = None  # Column from dataframe, if None use computed value


@dataclass(frozen=True)
class ChartSpec:
    """
    Declarative chart definition for a bet type.
    Defines how to visualize evidence outcomes.
    """
    chart_type: Literal['scatter', 'bar', 'line', 'histogram']
    x_axis: AxisSpec
    y_axis: AxisSpec
    title: str
    description: Optional[str] = None
