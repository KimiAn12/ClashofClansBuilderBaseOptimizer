"""
Troop data module for BASELINE.

Defines troop types with their combat statistics for optimization.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TroopType:
    """
    Represents a troop type with combat statistics.
    
    Attributes:
        name: Troop type name (e.g., 'barbarian', 'archer', 'giant')
        dps: Damage per second
        health: Hit points
        housing_cost: Housing space cost (for camp capacity constraint)
        movement_type: Movement type - 'ground' or 'air'
    """
    name: str
    dps: float
    health: float
    housing_cost: int
    movement_type: str  # 'ground' or 'air'
    
    def __post_init__(self):
        """Validate troop data after initialization."""
        if self.dps < 0:
            raise ValueError(f"dps must be non-negative, got {self.dps}")
        if self.health < 0:
            raise ValueError(f"health must be non-negative, got {self.health}")
        if self.housing_cost <= 0:
            raise ValueError(f"housing_cost must be positive, got {self.housing_cost}")
        if self.movement_type not in ('ground', 'air'):
            raise ValueError(f"movement_type must be 'ground' or 'air', got {self.movement_type}")


# Troop type definitions (heuristic values for Builder Base)
# These represent typical troop statistics at mid-level Builder Base
TROOP_TYPES: Dict[str, TroopType] = {
    'barbarian': TroopType(
        name='barbarian',
        dps=136.0,
        health=1179.0,
        housing_cost=1,
        movement_type='ground'
    ),
    'archer': TroopType(
        name='archer',
        dps=125.0,
        health=462.0,
        housing_cost=1,
        movement_type='ground'
    ),
    'giant': TroopType(
        name='giant',
        dps=129.0,
        health=5423.0,
        housing_cost=1,
        movement_type='ground'
    ),
    'minion': TroopType(
        name='minion',
        dps=126.0,
        health=429.0,
        housing_cost=1,
        movement_type='air'
    ),
    'bomber': TroopType(
        name='bomber',
        dps=150.0,
        health=1179.0,
        housing_cost=1,
        movement_type='ground'
    ),
    'cannon_cart': TroopType(
        name='cannon_cart',
        dps=240.0,
        health=1179.0,
        housing_cost=1,
        movement_type='ground'
    ),
        'baby_dragon': TroopType(
        name='baby_dragon',
        dps=120.0,
        health=2358.0,
        housing_cost=1,
        movement_type='ground'
    ),
    'night_witch': TroopType(
        name='night_witch',
        dps=278.0,
        health=1220.0,
        housing_cost=1,
        movement_type='ground'
    ),
    'drop_ship': TroopType(
        name='drop_ship',
        dps=0.0,
        health=4978.0,
        housing_cost=1,
        movement_type='air'
    ),
    'super_pekka': TroopType(
        name='super_pekka',
        dps=560.0,
        health=5191.0,
        housing_cost=1,
        movement_type='ground'
    ),
    'hog_glider': TroopType(
        name='hog_glider',
        dps=139.0,
        health=1533.0,
        housing_cost=1,
        movement_type='ground'
    ),
    'electrofire_wizard': TroopType(
        name='electrofire_wizard',
        dps=252.0,
        health=1210.0,
        housing_cost=1,
        movement_type='ground'
    ),

}


def get_troop_type(name: str) -> TroopType:
    """
    Get a troop type by name.
    
    Args:
        name: Troop type name
    
    Returns:
        TroopType object
    
    Raises:
        KeyError: If troop type not found
    """
    return TROOP_TYPES[name.lower()]


def get_all_troop_types() -> List[TroopType]:
    """
    Get all available troop types.
    
    Returns:
        List of all TroopType objects
    """
    return list(TROOP_TYPES.values())


def get_troop_names() -> List[str]:
    """
    Get list of all troop type names.
    
    Returns:
        List of troop type names
    """
    return list(TROOP_TYPES.keys())


# Hero type definitions (heuristic values for Builder Base)
HERO_TYPES: Dict[str, TroopType] = {
    'battle_machine': TroopType(
        name='battle_machine',
        dps=290.0,
        health=5380.0,
        housing_cost=1,  # Heroes don't use housing, but kept for consistency
        movement_type='ground'
    ),
    'battle_copter': TroopType(
        name='battle_copter',
        dps=193.0,
        health=3456.0,
        housing_cost=1,
        movement_type='air'
    ),
}


def get_hero_type(name: str) -> TroopType:
    """
    Get a hero type by name.
    
    Args:
        name: Hero type name
    
    Returns:
        TroopType object
    
    Raises:
        KeyError: If hero type not found
    """
    return HERO_TYPES[name.lower()]


def get_all_hero_types() -> List[TroopType]:
    """
    Get all available hero types.
    
    Returns:
        List of all HeroType objects
    """
    return list(HERO_TYPES.values())


def get_hero_names() -> List[str]:
    """
    Get list of all hero type names.
    
    Returns:
        List of hero type names
    """
    return list(HERO_TYPES.keys())
