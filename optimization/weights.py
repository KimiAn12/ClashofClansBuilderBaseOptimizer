"""
Weight assignment module for BASELINE.

Assigns heuristic threat scores to defenses based on:
1. Defense type (base threat level)
2. Clustering factor (number of nearby neighbors)

Uses simple deterministic formulas - no machine learning.
"""

from typing import Dict, List
import networkx as nx

from data.schema import Defense


# Base threat weights for different defense types
# Higher values = higher priority/threat
# These are heuristic values based on typical Clash of Clans defense characteristics
DEFENSE_TYPE_WEIGHTS: Dict[str, float] = {
    # High-threat defenses (long range, high damage)
    'crusher': 10.0,           # Melee defense, very high damage
    'x_bow': 9.5,              # Long range, very high damage, dual mode
    'giant_cannon': 9.0,       # Long range, high damage
    'mega_tesla': 8.8,         # Hidden defense, very high damage
    'multi_mortar': 8.5,       # Area damage, long range
    'double_cannon': 8.0,      # High damage, fast attack
    'lava_launcher': 7.5,      # Area damage, high threat
    'roaster': 7.0,            # Area damage, moderate-high threat
    
    # Medium-high threat defenses
    'archer_tower': 6.0,       # Good range, moderate damage
    'cannon': 5.5,             # Standard defense, moderate damage
    'firecrackers': 5.0,       # Area damage, moderate threat
    'firecracker': 5.0,        # Alias for firecrackers
    'air_bombs': 4.5,          # Air-only, moderate threat
    
    # Medium threat defenses
    'guard_post': 4.0,         # Spawns troops, moderate threat
    'mega_mine': 3.5,          # High damage but single-use
    'push_trap': 3.0,          # Utility defense, lower threat
    
    # Lower threat defenses
    'spring_trap': 2.0,        # Utility, low damage
    'mine': 1.5,               # Single-use, low threat
    'wall': 0.5,               # Obstacle, minimal direct threat
    
    # Resource and utility buildings (non-defensive, low threat)
    'gold_mine': 1.0,          # Resource building, no attack
    'elixir_collector': 1.0,   # Resource building, no attack
    'gem_mine': 1.0,           # Resource building, no attack
    'gold_storage': 1.2,       # Storage, slightly higher priority
    'elixir_storage': 1.2,     # Storage, slightly higher priority
    'barracks': 1.5,           # Production building, moderate priority
    'army_camp': 1.0,          # Storage building, low priority
    'laboratory': 1.3,         # Upgrade building, moderate priority
    'clock_tower': 1.0,        # Utility building, low priority
    'healing_hut': 1.0,        # Utility building, low priority
}


def get_base_weight(defense_type: str, default_weight: float = 5.0) -> float:
    """
    Get the base threat weight for a defense type.
    
    Args:
        defense_type: Type of defense (e.g., 'cannon', 'archer_tower')
        default_weight: Weight to use if defense type is not in the mapping.
                       Default: 5.0 (medium threat)
    
    Returns:
        Base threat weight for the defense type
    """
    return DEFENSE_TYPE_WEIGHTS.get(defense_type.lower(), default_weight)


def calculate_clustering_bonus(
    neighbor_count: int,
    base_clustering_factor: float = 0.2,
    max_bonus: float = 2.0
) -> float:
    """
    Calculate weight bonus based on number of nearby neighbors.
    
    Formula: bonus = min(base_factor * neighbor_count, max_bonus)
    
    This rewards defenses that are part of dense clusters, as they:
    - Create overlapping fire zones
    - Are harder to attack individually
    - Represent strategic chokepoints
    
    Args:
        neighbor_count: Number of defenses within connection radius
        base_clustering_factor: Multiplier for neighbor count. Default: 0.2
        max_bonus: Maximum bonus multiplier. Default: 2.0
    
    Returns:
        Clustering bonus multiplier (1.0 + bonus)
    
    Example:
        - 0 neighbors: 1.0 (no bonus)
        - 2 neighbors: 1.4 (20% bonus per neighbor)
        - 10+ neighbors: 3.0 (capped at max_bonus)
    """
    if neighbor_count < 0:
        neighbor_count = 0
    
    # Calculate raw bonus
    raw_bonus = base_clustering_factor * neighbor_count
    
    # Cap at maximum bonus
    capped_bonus = min(raw_bonus, max_bonus)
    
    # Return as multiplier (1.0 = no change, 2.0 = double weight)
    return 1.0 + capped_bonus


def calculate_threat_score(
    base_weight: float,
    neighbor_count: int,
    base_clustering_factor: float = 0.2,
    max_bonus: float = 2.0
) -> float:
    """
    Calculate final threat score for a defense.
    
    Formula: threat_score = base_weight * clustering_bonus
    
    Where:
        base_weight = defense type weight
        clustering_bonus = 1.0 + min(0.2 * neighbor_count, 2.0)
    
    Args:
        base_weight: Base threat weight from defense type
        neighbor_count: Number of nearby defenses
        base_clustering_factor: Multiplier for neighbor count. Default: 0.2
        max_bonus: Maximum clustering bonus. Default: 2.0
    
    Returns:
        Final threat score (higher = more threatening)
    """
    clustering_bonus = calculate_clustering_bonus(
        neighbor_count,
        base_clustering_factor,
        max_bonus
    )
    
    threat_score = base_weight * clustering_bonus
    
    return threat_score


def assign_weights_to_defenses(
    defenses: List[Defense],
    graph: nx.Graph,
    base_clustering_factor: float = 0.2,
    max_bonus: float = 2.0
) -> Dict[int, float]:
    """
    Assign threat scores to all defenses based on type and clustering.
    
    Process:
    1. Get base weight from defense type
    2. Count neighbors in graph
    3. Calculate clustering bonus
    4. Compute final threat score = base_weight * clustering_bonus
    
    Args:
        defenses: List of Defense objects
        graph: NetworkX graph with defense connections
        base_clustering_factor: Multiplier for neighbor count. Default: 0.2
        max_bonus: Maximum clustering bonus. Default: 2.0
    
    Returns:
        Dictionary mapping defense ID to threat score
    """
    threat_scores = {}
    
    for defense in defenses:
        # Get base weight from defense type
        base_weight = get_base_weight(defense.type)
        
        # Count neighbors in graph
        if defense.id in graph:
            neighbor_count = len(list(graph.neighbors(defense.id)))
        else:
            neighbor_count = 0
        
        # Calculate final threat score
        threat_score = calculate_threat_score(
            base_weight,
            neighbor_count,
            base_clustering_factor,
            max_bonus
        )
        
        threat_scores[defense.id] = threat_score
    
    return threat_scores


def update_defense_weights(
    defenses: List[Defense],
    graph: nx.Graph,
    base_clustering_factor: float = 0.2,
    max_bonus: float = 2.0
) -> List[Defense]:
    """
    Update Defense objects with calculated threat scores.
    
    Creates new Defense objects with updated base_weight values
    set to the calculated threat scores.
    
    Args:
        defenses: List of Defense objects
        graph: NetworkX graph with defense connections
        base_clustering_factor: Multiplier for neighbor count. Default: 0.2
        max_bonus: Maximum clustering bonus. Default: 2.0
    
    Returns:
        List of Defense objects with updated base_weight values
    """
    threat_scores = assign_weights_to_defenses(
        defenses,
        graph,
        base_clustering_factor,
        max_bonus
    )
    
    # Create updated defense objects
    updated_defenses = []
    for defense in defenses:
        updated_defense = Defense(
            id=defense.id,
            type=defense.type,
            x=defense.x,
            y=defense.y,
            base_weight=threat_scores[defense.id]
        )
        updated_defenses.append(updated_defense)
    
    return updated_defenses


def get_weight_summary(defenses: List[Defense], graph: nx.Graph) -> Dict:
    """
    Get summary statistics about defense weights.
    
    Args:
        defenses: List of Defense objects
        graph: NetworkX graph with defense connections
    
    Returns:
        Dictionary with weight statistics:
            - total_threat: Sum of all threat scores
            - avg_threat: Average threat score
            - max_threat: Maximum threat score
            - min_threat: Minimum threat score
            - high_threat_count: Number of defenses with threat > avg_threat
    """
    threat_scores = assign_weights_to_defenses(defenses, graph)
    
    if not threat_scores:
        return {
            'total_threat': 0.0,
            'avg_threat': 0.0,
            'max_threat': 0.0,
            'min_threat': 0.0,
            'high_threat_count': 0
        }
    
    scores = list(threat_scores.values())
    total_threat = sum(scores)
    avg_threat = total_threat / len(scores) if scores else 0.0
    max_threat = max(scores)
    min_threat = min(scores)
    high_threat_count = sum(1 for s in scores if s > avg_threat)
    
    return {
        'total_threat': total_threat,
        'avg_threat': avg_threat,
        'max_threat': max_threat,
        'min_threat': min_threat,
        'high_threat_count': high_threat_count
    }

