"""
Data schema module for BASELINE.

Defines data structures for defense objects and conversion utilities.
"""

from .schema import (
    Defense,
    detections_to_defenses,
    defense_to_dict,
    defenses_to_dicts,
    get_defense_stats
)
from .troops import (
    TroopType,
    get_troop_type,
    get_all_troop_types,
    get_troop_names,
    TROOP_TYPES,
    get_hero_type,
    get_all_hero_types,
    get_hero_names,
    HERO_TYPES
)

__all__ = [
    'Defense',
    'detections_to_defenses',
    'defense_to_dict',
    'defenses_to_dicts',
    'get_defense_stats',
    'TroopType',
    'get_troop_type',
    'get_all_troop_types',
    'get_troop_names',
    'TROOP_TYPES',
    'get_hero_type',
    'get_all_hero_types',
    'get_hero_names',
    'HERO_TYPES'
]

