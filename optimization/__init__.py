"""
Optimization logic module for BASELINE.

Implements linear integer programming and attack strategy optimization.
"""

from .weights import (
    assign_weights_to_defenses,
    update_defense_weights,
    calculate_threat_score,
    calculate_clustering_bonus,
    get_base_weight,
    get_weight_summary,
    DEFENSE_TYPE_WEIGHTS
)

from .solver import (
    solve_troop_composition,
    solve_troop_composition_with_weights,
    print_troop_composition,
    TroopComposition,
    # Legacy functions (deprecated)
    solve_attack_plan,
    solve_attack_plan_with_weights,
    get_attack_priority_list,
    print_attack_plan,
    AttackPlan
)

from .effectiveness import (
    get_effectiveness,
    get_effectiveness_matrix,
    EFFECTIVENESS_MATRIX
)

__all__ = [
    # Weight functions
    'assign_weights_to_defenses',
    'update_defense_weights',
    'calculate_threat_score',
    'calculate_clustering_bonus',
    'get_base_weight',
    'get_weight_summary',
    'DEFENSE_TYPE_WEIGHTS',
    # Solver functions (new)
    'solve_troop_composition',
    'solve_troop_composition_with_weights',
    'print_troop_composition',
    'TroopComposition',
    # Legacy solver functions (deprecated)
    'solve_attack_plan',
    'solve_attack_plan_with_weights',
    'get_attack_priority_list',
    'print_attack_plan',
    'AttackPlan',
    # Effectiveness
    'get_effectiveness',
    'get_effectiveness_matrix',
    'EFFECTIVENESS_MATRIX'
]

