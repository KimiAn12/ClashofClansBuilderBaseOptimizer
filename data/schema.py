"""
Data schema module for BASELINE.

Defines data structures for defense objects and conversion utilities.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Defense:
    """
    Represents a defense structure in the Builder Base.
    
    Attributes:
        id: Unique identifier for this defense instance
        type: Type of defense (e.g., 'cannon', 'archer_tower', 'crusher')
        x: Normalized x-coordinate [0, 1]
        y: Normalized y-coordinate [0, 1]
        base_weight: Base weight/priority for optimization (default placeholder: 1.0)
        health: Defense hit points (default: 1000.0)
        dps: Defense damage per second (default: 50.0)
    """
    id: int
    type: str
    x: float
    y: float
    base_weight: float = 1.0
    health: float = 1000.0
    dps: float = 50.0
    
    def __post_init__(self):
        """Validate defense data after initialization."""
        if not 0.0 <= self.x <= 1.0:
            raise ValueError(f"x must be in [0, 1], got {self.x}")
        if not 0.0 <= self.y <= 1.0:
            raise ValueError(f"y must be in [0, 1], got {self.y}")
        if self.base_weight < 0:
            raise ValueError(f"base_weight must be non-negative, got {self.base_weight}")
        if self.health < 0:
            raise ValueError(f"health must be non-negative, got {self.health}")
        if self.dps < 0:
            raise ValueError(f"dps must be non-negative, got {self.dps}")


def get_defense_attack_type(defense_type: str) -> str:
    """
    Get the attack type capability of a defense.
    
    Args:
        defense_type: Type of defense
    
    Returns:
        Attack type: 'ground', 'air', or 'both'
    """
    # Defense attack capabilities based on Clash of Clans mechanics
    attack_types = {
        # Ground-only defenses (cannot attack air troops)
        'crusher': 'ground',
        'giant_cannon': 'ground',
        'multi_mortar': 'ground',
        'double_cannon': 'ground',
        'cannon': 'ground',
        'guard_post': 'ground',
        
        # Air-only defenses (can only attack air troops)
        'air_bombs': 'air',
        
        # Both ground and air defenses
        'archer_tower': 'both',
        'x_bow': 'both',
        'mega_tesla': 'both',
        'lava_launcher': 'both',
        'roaster': 'both',
        'firecrackers': 'both',
        
        # Non-defensive buildings (don't attack)
        'gold_mine': 'both',  # Doesn't attack, but treat as both for consistency
        'elixir_collector': 'both',
        'gem_mine': 'both',
        'gold_storage': 'both',
        'elixir_storage': 'both',
        'barracks': 'both',
        'army_camp': 'both',
        'laboratory': 'both',
        'clock_tower': 'both',
        'healing_hut': 'both',
        'wall': 'both',  # Walls don't attack
    }
    return attack_types.get(defense_type.lower(), 'both')  # Default to 'both' if unknown


def get_defense_stats(defense_type: str) -> tuple[float, float]:
    """
    Get default health and DPS for a defense type.
    
    Args:
        defense_type: Type of defense
    
    Returns:
        Tuple of (health, dps)
    """
    # Heuristic default values based on defense type
    stats = {
        # High-tier defensive buildings
        'crusher': (2900.0, 296.0),
        'giant_cannon': (2400.0, 96.0),
        'multi_mortar': (1850.0, 58.0),
        'double_cannon': (2200.0, 202.0),
        'x_bow': (2600.0, 116.0),          # Long range, high damage
        'mega_tesla': (2400.0, 224.0),    # Hidden defense, high damage
        'lava_launcher': (1850.0, 118.0),  # Area damage, moderate-high threat
        'roaster': (2600.0, 136.0),         # Area damage, moderate-high threat
        
        # Medium-tier defensive buildings
        'archer_tower': (1850.0, 94.0),
        'cannon': (1850.0, 165.0),
        'firecrackers': (1500.0, 128.0),
        
        # Air defenses
        'air_bombs': (2900.0, 212.0),
        
        # Other structures
        'guard_post': (1150.0, 0.0),
        'wall': (3350.0, 0.0),
        
        # Resource and utility buildings (non-defensive)
        'gold_mine': (1150.0, 0.0),
        'elixir_collector': (1150.0, 0.0),
        'gem_mine': (1150.0, 0.0),
        'gold_storage': (2750.0, 0.0),
        'elixir_storage': (2750.0, 0.0),
        'barracks': (1450.0, 0.0),
        'army_camp': (300.0, 0.0),
        'laboratory': (2750.0, 0.0),
        'clock_tower': (2750.0, 0.0),
        'healing_hut': (1150.0, 0.0),
    }
    return stats.get(defense_type.lower(), (1000.0, 50.0))


def detections_to_defenses(
    detections: List[Dict[str, Any]],
    start_id: int = 0
) -> List[Defense]:
    """
    Convert vision detection dictionaries to Defense objects.
    
    Args:
        detections: List of detection dictionaries from vision module, each containing:
            - defense_type: Name of the defense (str)
            - x_center: Normalized x-coordinate [0, 1] (float)
            - y_center: Normalized y-coordinate [0, 1] (float)
            - match_score: Match confidence score [0, 1] (float, optional)
            - health: Defense health (float, optional)
            - dps: Defense DPS (float, optional)
        start_id: Starting ID for defense objects. Default: 0
    
    Returns:
        List of Defense objects with sequential IDs
    
    Example:
        >>> detections = [
        ...     {'defense_type': 'cannon', 'x_center': 0.5, 'y_center': 0.5, 'match_score': 0.9},
        ...     {'defense_type': 'archer', 'x_center': 0.2, 'y_center': 0.2, 'match_score': 0.8}
        ... ]
        >>> defenses = detections_to_defenses(detections)
        >>> # Returns [Defense(id=0, type='cannon', x=0.5, y=0.5, base_weight=1.0, health=1000.0, dps=50.0),
        >>> #          Defense(id=1, type='archer', x=0.2, y=0.2, base_weight=1.0, health=1200.0, dps=70.0)]
    """
    defenses = []
    
    for idx, detection in enumerate(detections):
        defense_type = detection['defense_type']
        
        # Get health and DPS (use provided or default)
        if 'health' in detection:
            health = detection['health']
        else:
            health, _ = get_defense_stats(defense_type)
        
        if 'dps' in detection:
            dps = detection['dps']
        else:
            _, dps = get_defense_stats(defense_type)
        
        defense = Defense(
            id=start_id + idx,
            type=defense_type,
            x=detection['x_center'],
            y=detection['y_center'],
            base_weight=1.0,  # Default placeholder
            health=health,
            dps=dps
        )
        defenses.append(defense)
    
    return defenses


def defense_to_dict(defense: Defense) -> Dict[str, Any]:
    """
    Convert a Defense object to a dictionary.
    
    Args:
        defense: Defense object to convert
    
    Returns:
        Dictionary representation of the defense
    """
    return {
        'id': defense.id,
        'type': defense.type,
        'x': defense.x,
        'y': defense.y,
        'base_weight': defense.base_weight,
        'health': defense.health,
        'dps': defense.dps
    }


def defenses_to_dicts(defenses: List[Defense]) -> List[Dict[str, Any]]:
    """
    Convert a list of Defense objects to a list of dictionaries.
    
    Args:
        defenses: List of Defense objects
    
    Returns:
        List of dictionary representations
    """
    return [defense_to_dict(d) for d in defenses]

