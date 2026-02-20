from typing import Any, Literal, Optional, List, Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class FilterSpec:
    """
    Declarative description of a single filter rule.
    Execution is handled in the Analytics Store
    """
    key: str
    kind: Literal['column', 'context']
    operator: Literal['>', '>=', '<=', '<', '==', '!=', 'between', 'in']
    value: Any
    field: Optional[str] = None
    perspective: Optional[Literal['home', 'away']] = None
    display_name: Optional[str] = None
    required_columns: Tuple[str, ...] = ()  # Columns needed for this filter

    def validate(self) -> None:
        """
        Validate filter structure (not data).
        Raises ValueError if invalid.
        """
        # For column filters, field is required
        if self.kind == 'column' and not isinstance(self.field, str):
            raise ValueError("Filter field must be a string for column filters")
        
        if self.operator == 'between':
            if not (
                isinstance(self.value, (tuple, list))
                and len(self.value) == 2
            ):
                raise ValueError(
                    "Between operator requires (min, max)"
                )
            
        if self.operator == 'in':
            if not isinstance(self.value, (list, set, tuple)):
                raise ValueError(
                    "In operator requires a collection"
                )
            
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "perspective": self.perspective,
            "display_name": self.display_name,
            "required_columns": list(self.required_columns),
        }
    
    @classmethod
    def from_dict(cls, payload: dict) -> "FilterSpec":
        return cls(
            key=payload['key'],
            kind=payload['kind'],
            field=payload.get('field'),
            operator=payload['operator'],
            value=payload['value'],
            perspective=payload.get("perspective"),
            display_name=payload.get("display_name"),
            required_columns=tuple(payload.get("required_columns", [])),
        )