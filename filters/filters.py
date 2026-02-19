from abc import ABC, abstractmethod
from typing import Any, Optional, Literal

from contracts.filter_spec import FilterSpec


class BaseFilter(ABC):
    """
    Base class for all filters.
    Each filter defines how to build a FilterSpec.
    """

    # unique identifier for validation
    key: str

    # human readable name (UI-friendly)
    display_name: str

    @classmethod
    @abstractmethod
    def build(
        cls,
        *,
        operator: Literal['>', '>=', '<=', '<', '==', '!=', 'between', 'in'],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        """
        Returns a FilterSpec for this filter
        """
        pass

class XGDifferenceFilter(BaseFilter):
    key = "xg_difference"
    display_name = "xG Difference"
    required_columns = ("xg_diff",)

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['>=', '<=', 'between'],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="xg_diff",  # Aligned with builder feature name
            operator=operator,
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec

class VenueFilter(BaseFilter):
    key = "venue"
    display_name = "Match Venue"
    required_columns = ("venue",)

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['=='],
        value: Literal['home', 'away'],
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="venue",
            operator=operator,
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec

class TeamMomentumFilter(BaseFilter):
    key = "team_momentum"
    display_name = "Team Momentum"
    required_columns = ("home_momentum", "away_momentum")

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['>=', '<=', 'between'],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        # Team momentum is evaluated per row using venue context in the store.
        # Keep field stable for schema validation and compatibility.
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("TeamMomentumFilter 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))
        
        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="home_momentum",
            operator=operator,
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec
    
class OpponentMomentumFilter(BaseFilter):
    key = "opponent_momentum"
    display_name = "Opponent Momentum"
    required_columns = ("home_momentum", "away_momentum")

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['>=', '<=', 'between'],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        # Opponent momentum is evaluated per row using venue context in the store.
        # Keep field stable for schema validation and compatibility.
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("OpponentMomentumFilter 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))
        
        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="away_momentum",
            operator=operator,
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec

class HomeTotalGoals(BaseFilter):
    key = 'home_goals'
    display_name = 'Home Total Goals'
    required_columns = ("home_normaltime",)

    @classmethod
    def build(
        cls, 
        *, 
        operator: Literal['>=', '<=', '>', '<', '=='], 
        value: float, 
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="home_normaltime",
            operator=operator, 
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec
    
class AwayTotalGoals(BaseFilter):
    key = 'away_goals'
    display_name = 'Away Total Goals'
    required_columns = ("away_normaltime",)

    @classmethod
    def build(
        cls, 
        *, 
        operator: Literal['>=', '<=', '>', '<', '=='], 
        value: float, 
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="away_normaltime",
            operator=operator, 
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec

    
class LastNGames(BaseFilter):
    key = 'last_n_games'
    display_name = 'Last N Games'
    required_columns = ()

    @classmethod
    def build(
        cls, 
        *, 
        operator: Literal['<=', '=='],
        value: int,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        spec = FilterSpec(
            key=cls.key,
            kind='context',
            field=None,  # Context filters don't need a field
            operator=operator, 
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        # Skip validation for context filters without field
        return spec
    
class HeadToHead(BaseFilter):
    key = 'head_to_head'
    display_name = 'Head-To-Head'
    required_columns = ()

    @classmethod
    def build(
        cls, 
        *, 
        operator: Literal['=='],
        value: bool,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        spec = FilterSpec(
            key=cls.key,
            kind='context',
            field=None,  # Context filters don't need a field
            operator=operator, 
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        # Skip validation for context filters without field
        return spec

class GoalsScored(BaseFilter):
    key = 'goals_scored'
    display_name = 'Team Goals'
    required_columns = ('goals_scored', 'venue')  # Columns this filter needs

    @classmethod
    def build(
        cls, 
        *, 
        operator: Literal['>=', '<=', '>', '<', '==', 'between'],
        value: Any, 
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("GoalsScored 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))

        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="goals_scored",
            operator=operator, 
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec

class GoalsConceded(BaseFilter):
    key = 'goals_conceded'
    display_name = 'Opposition Goals'
    required_columns = ('opponent_goals', 'venue')

    @classmethod
    def build(
        cls, 
        *, 
        operator: Literal['>=', '<=', '>', '<', '==', 'between'],
        value: Any, 
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("GoalsConceded 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))

        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="opponent_goals",
            operator=operator, 
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec

class TotalMatchGoals(BaseFilter):
    key = 'total_match_goals'
    display_name = 'Total Match Goals'
    required_columns = ('total_goals',)

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['>=', '<=', '>', '<', '==', 'between'],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("TotalMatchGoals 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))

        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="total_goals",
            operator=operator,
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec

class TeamXG(BaseFilter):
    key = 'team_xg'
    display_name = 'Team Expected Goals'
    required_columns = ('expected_goals_home', 'expected_goals_away', 'venue')

    @classmethod
    def build(
        cls, 
        *, 
        operator: Literal['>=', '<=', '>', '<', '==', 'between'],
        value: Any, 
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("TeamXG 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))

        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="team_xg",
            operator=operator, 
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec

class OpponentXG(BaseFilter):
    key = 'opponent_xg'
    display_name = 'Opponent Expected Goals'
    required_columns = ('expected_goals_home', 'expected_goals_away', 'venue')

    @classmethod
    def build(
        cls, 
        *, 
        operator: Literal['>=', '<=', '>', '<', '==', 'between'],
        value: Any, 
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("OpponentXG 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))

        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="opponent_xg",
            operator=operator, 
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec


class TeamPossessionFilter(BaseFilter):
    key = "team_possession"
    display_name = "Team Possession"
    required_columns = ("ball_possession_home", "ball_possession_away", "venue")

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['>=', '<=', '>', '<', '==', 'between'],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("TeamPossessionFilter 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))

        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="ball_possession_home",
            operator=operator,
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec


class OpponentPossessionFilter(BaseFilter):
    key = "opponent_possession"
    display_name = "Opponent Possession"
    required_columns = ("ball_possession_home", "ball_possession_away", "venue")

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['>=', '<=', '>', '<', '==', 'between'],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("OpponentPossessionFilter 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))

        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="ball_possession_away",
            operator=operator,
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec


class FieldTiltFilter(BaseFilter):
    key = "field_tilt"
    display_name = "Field Tilt"
    required_columns = ("field_tilt_home", "field_tilt_away", "venue")

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['>=', '<=', '>', '<', '==', 'between'],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("FieldTiltFilter 'between' requires (min, max)")
            low, high = value
            value = (min(low, high), max(low, high))

        spec = FilterSpec(
            key=cls.key,
            kind='column',
            field="field_tilt_home",
            operator=operator,
            value=value,
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec


class TeamShotXGFilter(BaseFilter):
    key = "team_shot_xg"
    display_name = "Team Shot xG Count"
    required_columns = ("home_shots", "away_shots", "venue")

    @classmethod
    def build(
        cls,
        *,
        operator: Literal['>='],
        value: Any,
        perspective: Optional[Literal['home', 'away']] = None,
    ) -> FilterSpec:
        if operator != ">=":
            raise ValueError("TeamShotXGFilter supports only '>=' operator")
        if not isinstance(value, dict):
            raise ValueError("TeamShotXGFilter value must be a dict with min_xg and min_shots")

        min_xg = value.get("min_xg")
        min_shots = value.get("min_shots")
        try:
            min_xg = float(min_xg)
            min_shots = int(min_shots)
        except (TypeError, ValueError):
            raise ValueError("TeamShotXGFilter requires numeric min_xg and integer min_shots")

        if min_xg < 0:
            raise ValueError("TeamShotXGFilter min_xg must be >= 0")
        if min_shots < 0:
            raise ValueError("TeamShotXGFilter min_shots must be >= 0")

        spec = FilterSpec(
            key=cls.key,
            kind="column",
            field="home_shots",
            operator=operator,
            value={"min_xg": min_xg, "min_shots": min_shots},
            perspective=perspective,
            display_name=cls.display_name,
            required_columns=cls.required_columns,
        )
        spec.validate()
        return spec
