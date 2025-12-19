"""
Mixed-integer linear programming solver for BASELINE.

Formulates troop composition selection as a MILP problem to maximize
weighted defense neutralization subject to housing and combat constraints.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pulp

from data.schema import Defense
from data.troops import TroopType, get_troop_type, get_all_troop_types, get_all_hero_types
from optimization.effectiveness import get_effectiveness


@dataclass
class TroopComposition:
    """
    Represents the optimal troop composition solution.
    
    Attributes:
        troop_counts: Dictionary mapping troop type name to count
        hero_type: Selected hero type name (None if no hero selected)
        neutralized_defenses: List of Defense objects that will be neutralized
        total_housing_used: Total housing space used (for display purposes)
        total_threat_neutralized: Sum of strategic weights of neutralized defenses
        objective_value: Optimal objective value from the solver
    """
    troop_counts: Dict[str, int]
    hero_type: Optional[str]
    neutralized_defenses: List[Defense]
    total_housing_used: int
    total_threat_neutralized: float
    objective_value: float


def solve_troop_composition(
    defenses: List[Defense],
    max_army_camps: int = 6,
    troop_types: Optional[List[TroopType]] = None,
    hero_types: Optional[List[TroopType]] = None,
    strategic_weights: Optional[Dict[int, float]] = None,
    use_survivability: bool = True,
    time_estimate: float = 30.0
) -> TroopComposition:
    """
    Solve the troop composition problem as a mixed-integer linear program.
    
    New Constraint Model:
    - 6 Army Camps: Each camp can hold 1 different troop type (exactly 6 different troop types must be selected)
    - 1 Hero Camp: Can hold 1 hero type (exactly 1 hero selected)
    
    Mathematical Formulation:
    
    Sets:
        T: Set of troop types
        H: Set of hero types
        D: Set of defenses
    
    Decision Variables:
        x[i] ∈ Z≥0  : Number of troops of type i
        z[i] ∈ {0,1}: Whether troop type i is selected (for army camp constraint)
        h[k] ∈ {0,1}: Whether hero type k is selected
        y[j] ∈ {0,1}: Whether defense j is neutralized
    
    Objective:
        maximize: Σ w[j] · y[j]
    
    Constraints:
        Army Camps: Σ z[i] = 6 (exactly 6 different troop types must be selected)
        Troop Selection: x[i] ≤ M · z[i]  ∀i (can only use troops if type is selected)
        Hero Camp: Σ h[k] = 1 (exactly 1 hero must be selected)
        Damage Sufficiency: (Σ d[i] · x[i] · e[i,j]) + (Σ d_h[k] · h[k] · e_h[k,j]) ≥ H[j] · y[j]  ∀j
        Survivability (optional): (Σ h_t[i] · x[i]) + (Σ h_h[k] · h[k]) ≥ Σ D[j] · y[j] · T
    
    Where:
        d[i]: DPS of troop type i
        d_h[k]: DPS of hero type k
        h_t[i]: Health of troop type i
        h_h[k]: Health of hero type k
        H[j]: Health of defense j
        D[j]: DPS of defense j
        w[j]: Strategic weight of defense j
        e[i,j]: Effectiveness of troop i against defense j
        e_h[k,j]: Effectiveness of hero k against defense j
        M: Large constant (big-M)
        T: Estimated combat time
    
    Args:
        defenses: List of Defense objects with health and DPS
        max_army_camps: Number of army camps (exactly this many different troop types must be selected). Default: 6
        troop_types: List of available troop types. If None, uses all default types.
        hero_types: List of available hero types. If None, uses all default heroes.
        strategic_weights: Dictionary mapping defense ID to strategic weight.
                          If None, uses defense.base_weight. Default: None
        use_survivability: Whether to include survivability constraint. Default: True
        time_estimate: Estimated combat duration for survivability calculation.
                      Default: 30.0 seconds
    
    Returns:
        TroopComposition object with optimal solution
    
    Raises:
        ValueError: If max_army_camps is not positive or if no defenses/troops provided
    """
    if max_army_camps <= 0:
        raise ValueError(f"max_army_camps must be positive, got {max_army_camps}")
    
    if not defenses:
        return TroopComposition(
            troop_counts={},
            hero_type=None,
            neutralized_defenses=[],
            total_housing_used=0,
            total_threat_neutralized=0.0,
            objective_value=0.0
        )
    
    # Get troop types
    if troop_types is None:
        troop_types = get_all_troop_types()
    
    if not troop_types:
        raise ValueError("No troop types available")
    
    # Validate that we have enough troop types to fill all army camps
    if len(troop_types) < max_army_camps:
        raise ValueError(
            f"Not enough troop types available: need {max_army_camps} camps, "
            f"but only {len(troop_types)} troop types available"
        )
    
    # Get hero types
    if hero_types is None:
        hero_types = get_all_hero_types()
    
    if not hero_types:
        raise ValueError("No hero types available")
    
    # Create mapping from troop/hero name to TroopType
    troop_map = {troop.name: troop for troop in troop_types}
    hero_map = {hero.name: hero for hero in hero_types}
    
    # Use provided strategic weights or fall back to base_weight
    if strategic_weights is None:
        strategic_weights = {defense.id: defense.base_weight for defense in defenses}
    
    # Create defense mapping
    defense_map = {defense.id: defense for defense in defenses}
    
    # Create the optimization problem
    problem = pulp.LpProblem("Troop_Composition", pulp.LpMaximize)
    
    # Decision variables
    # x[i]: Integer variable for number of troops of type i
    troop_vars = {}
    for troop in troop_types:
        var_name = f"troop_{troop.name}"
        troop_vars[troop.name] = pulp.LpVariable(var_name, lowBound=0, cat='Integer')
    
    # z[i]: Binary variable for whether troop type i is selected (for army camp)
    troop_selection_vars = {}
    for troop in troop_types:
        var_name = f"select_troop_{troop.name}"
        troop_selection_vars[troop.name] = pulp.LpVariable(var_name, cat='Binary')
    
    # h[k]: Binary variable for whether hero type k is selected
    hero_vars = {}
    for hero in hero_types:
        var_name = f"hero_{hero.name}"
        hero_vars[hero.name] = pulp.LpVariable(var_name, cat='Binary')
    
    # y[j]: Binary variable for whether defense j is neutralized
    defense_vars = {}
    for defense in defenses:
        var_name = f"neutralize_defense_{defense.id}"
        defense_vars[defense.id] = pulp.LpVariable(var_name, cat='Binary')
    
    # Objective: Maximize weighted defense neutralization
    # maximize: Σ w[j] · y[j]
    objective = pulp.lpSum([
        strategic_weights.get(defense.id, 0.0) * defense_vars[defense.id]
        for defense in defenses
    ])
    problem += objective, "Total_Neutralized_Threat"
    
    # Constraint 1: Army camps - exactly max_army_camps different troop types must be selected
    # Σ z[i] = max_army_camps
    army_camp_constraint = pulp.lpSum([
        troop_selection_vars[troop.name]
        for troop in troop_types
    ]) == max_army_camps
    problem += army_camp_constraint, "Army_Camps_Exact"
    
    # Constraint 2: Troop selection - can only use troops if type is selected
    # x[i] ≤ M · z[i]  ∀i (if not selected, cannot use)
    # x[i] ≥ z[i]  ∀i (if selected, must use at least 1)
    # Use big-M: estimate max troops needed (e.g., 1000 per type)
    big_M = 1000
    for troop in troop_types:
        # Upper bound: can only use troops if type is selected
        upper_constraint = troop_vars[troop.name] <= big_M * troop_selection_vars[troop.name]
        problem += upper_constraint, f"Troop_Selection_Upper_{troop.name}"
        
        # Lower bound: if selected, must use at least 1 troop
        lower_constraint = troop_vars[troop.name] >= troop_selection_vars[troop.name]
        problem += lower_constraint, f"Troop_Selection_Lower_{troop.name}"
    
    # Constraint 3: Hero camp - exactly 1 hero must be selected
    # Σ h[k] = 1
    hero_constraint = pulp.lpSum([
        hero_vars[hero.name]
        for hero in hero_types
    ]) == 1
    problem += hero_constraint, "Hero_Camp"
    
    # Constraint 4: Damage sufficiency for each defense
    # (Σ d[i] · x[i] · e[i,j]) + (Σ d_h[k] · h[k] · e_h[k,j]) ≥ H[j] · y[j]  ∀j
    for defense in defenses:
        # Total effective DPS from troops
        troop_dps = pulp.lpSum([
            troop.dps * troop_vars[troop.name] * get_effectiveness(troop.name, defense.type)
            for troop in troop_types
        ])
        
        # Total effective DPS from heroes
        hero_dps = pulp.lpSum([
            hero.dps * hero_vars[hero.name] * get_effectiveness(hero.name, defense.type)
            for hero in hero_types
        ])
        
        # Total effective DPS
        effective_dps = troop_dps + hero_dps
        
        # Required damage to neutralize = defense health if y[j] = 1, else 0
        damage_constraint = effective_dps >= defense.health * defense_vars[defense.id]
        problem += damage_constraint, f"Damage_Sufficiency_Defense_{defense.id}"
    
    # Constraint 5: Survivability (optional)
    # (Σ h_t[i] · x[i]) + (Σ h_h[k] · h[k]) ≥ Σ D[j] · y[j] · T
    if use_survivability:
        total_troop_health = pulp.lpSum([
            troop.health * troop_vars[troop.name]
            for troop in troop_types
        ])
        
        total_hero_health = pulp.lpSum([
            hero.health * hero_vars[hero.name]
            for hero in hero_types
        ])
        
        total_defense_damage = pulp.lpSum([
            defense.dps * defense_vars[defense.id] * time_estimate
            for defense in defenses
        ])
        
        survivability_constraint = (total_troop_health + total_hero_health) >= total_defense_damage
        problem += survivability_constraint, "Survivability"
    
    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    troop_counts = {}
    for troop in troop_types:
        count = int(troop_vars[troop.name].varValue or 0)
        if count > 0:
            troop_counts[troop.name] = count
    
    selected_hero = None
    for hero in hero_types:
        if hero_vars[hero.name].varValue == 1:
            selected_hero = hero.name
            break
    
    neutralized_defense_ids = []
    for defense in defenses:
        if defense_vars[defense.id].varValue == 1:
            neutralized_defense_ids.append(defense.id)
    
    neutralized_defenses = [
        defense_map[def_id] for def_id in neutralized_defense_ids
    ]
    
    # Calculate totals (for display purposes)
    total_housing_used = sum(
        troop.housing_cost * troop_counts.get(troop.name, 0)
        for troop in troop_types
    )
    
    total_threat_neutralized = sum(
        strategic_weights.get(def_id, 0.0)
        for def_id in neutralized_defense_ids
    )
    
    objective_value = pulp.value(problem.objective) if problem.status == pulp.LpStatusOptimal else 0.0
    
    return TroopComposition(
        troop_counts=troop_counts,
        hero_type=selected_hero,
        neutralized_defenses=neutralized_defenses,
        total_housing_used=total_housing_used,
        total_threat_neutralized=total_threat_neutralized,
        objective_value=objective_value
    )


def solve_troop_composition_with_weights(
    defenses: List[Defense],
    max_army_camps: int = 6,
    graph = None,
    troop_types: Optional[List[TroopType]] = None,
    hero_types: Optional[List[TroopType]] = None,
    base_clustering_factor: float = 0.2,
    max_bonus: float = 2.0,
    use_survivability: bool = True,
    time_estimate: float = 30.0
) -> TroopComposition:
    """
    Solve troop composition using strategic weights calculated from graph structure.
    
    This is a convenience function that:
    1. Calculates strategic weights using graph structure
    2. Solves the MILP problem
    3. Returns the troop composition
    
    Args:
        defenses: List of Defense objects
        max_army_camps: Number of army camps (exactly this many different troop types must be selected). Default: 6
        graph: NetworkX graph with defense connections
        troop_types: List of available troop types. If None, uses all default types.
        hero_types: List of available hero types. If None, uses all default heroes.
        base_clustering_factor: Multiplier for neighbor count in weight calculation.
                               Default: 0.2
        max_bonus: Maximum clustering bonus. Default: 2.0
        use_survivability: Whether to include survivability constraint. Default: True
        time_estimate: Estimated combat duration. Default: 30.0 seconds
    
    Returns:
        TroopComposition object with optimal solution
    """
    from optimization.weights import assign_weights_to_defenses
    
    # Calculate strategic weights from graph
    strategic_weights = assign_weights_to_defenses(
        defenses,
        graph,
        base_clustering_factor,
        max_bonus
    )
    
    # Solve using calculated weights
    return solve_troop_composition(
        defenses,
        max_army_camps,
        troop_types,
        hero_types,
        strategic_weights,
        use_survivability,
        time_estimate
    )


def print_troop_composition(composition: TroopComposition) -> None:
    """
    Print a human-readable troop composition.
    
    Args:
        composition: TroopComposition object to print
    """
    print("=" * 60)
    print("TROOP COMPOSITION")
    print("=" * 60)
    print(f"Troop Types Selected: {len(composition.troop_counts)}")
    print(f"Defenses Neutralized: {len(composition.neutralized_defenses)}")
    print(f"Total Threat Neutralized: {composition.total_threat_neutralized:.2f}")
    print(f"Objective Value: {composition.objective_value:.2f}")
    print()
    print("Army Camps (Troop Composition):")
    print("-" * 60)
    
    for idx, (troop_name, count) in enumerate(sorted(composition.troop_counts.items()), 1):
        troop = get_troop_type(troop_name)
        print(f"Camp {idx}: {troop_name.capitalize()} - {count} troops")
        print(f"  DPS: {troop.dps:.1f}, Health: {troop.health:.1f}")
        print(f"  Total DPS: {troop.dps * count:.1f}, Total Health: {troop.health * count:.1f}")
        print()
    
    print("Hero Camp:")
    print("-" * 60)
    if composition.hero_type:
        from data.troops import get_hero_type
        hero = get_hero_type(composition.hero_type)
        print(f"Hero: {composition.hero_type.replace('_', ' ').title()}")
        print(f"  DPS: {hero.dps:.1f}, Health: {hero.health:.1f}")
    else:
        print("No hero selected")
    print()
    
    print("Neutralized Defenses:")
    print("-" * 60)
    for defense in composition.neutralized_defenses:
        print(f"Defense ID {defense.id} ({defense.type})")
        print(f"  Health: {defense.health:.1f}, DPS: {defense.dps:.1f}")
        print(f"  Position: ({defense.x:.3f}, {defense.y:.3f})")
        print()
    
    print("=" * 60)


# Legacy functions for backward compatibility (deprecated)
@dataclass
class AttackPlan:
    """
    Legacy AttackPlan dataclass (deprecated).
    Use TroopComposition instead.
    """
    selected_defenses: List[Defense]
    threat_neutralized: float
    attack_count: int
    objective_value: float


def solve_attack_plan(
    defenses: List[Defense],
    attack_budget: int,
    threat_scores: Optional[Dict[int, float]] = None
) -> AttackPlan:
    """
    Legacy function (deprecated).
    Use solve_troop_composition instead.
    """
    # Convert to new formulation for backward compatibility
    # This is a simplified version that doesn't use troop composition
    if not defenses:
        return AttackPlan(
            selected_defenses=[],
            threat_neutralized=0.0,
            attack_count=0,
            objective_value=0.0
        )
    
    if threat_scores is None:
        threat_scores = {defense.id: defense.base_weight for defense in defenses}
    
    # Simple selection: pick top defenses by weight
    sorted_defenses = sorted(
        defenses,
        key=lambda d: threat_scores.get(d.id, 0.0),
        reverse=True
    )
    
    selected = sorted_defenses[:attack_budget]
    total_threat = sum(threat_scores.get(d.id, 0.0) for d in selected)
    
    return AttackPlan(
        selected_defenses=selected,
        threat_neutralized=total_threat,
        attack_count=len(selected),
        objective_value=total_threat
    )


def solve_attack_plan_with_weights(
    defenses: List[Defense],
    attack_budget: int,
    graph,
    base_clustering_factor: float = 0.2,
    max_bonus: float = 2.0
) -> AttackPlan:
    """
    Legacy function (deprecated).
    Use solve_troop_composition_with_weights instead.
    """
    from optimization.weights import assign_weights_to_defenses
    
    threat_scores = assign_weights_to_defenses(
        defenses,
        graph,
        base_clustering_factor,
        max_bonus
    )
    
    return solve_attack_plan(defenses, attack_budget, threat_scores)


def get_attack_priority_list(attack_plan: AttackPlan) -> List[Tuple[int, str, float]]:
    """
    Legacy function (deprecated).
    """
    priority_list = []
    for defense in attack_plan.selected_defenses:
        priority_list.append((
            defense.id,
            defense.type,
            defense.base_weight
        ))
    return priority_list


def print_attack_plan(attack_plan: AttackPlan) -> None:
    """
    Legacy function (deprecated).
    """
    print("=" * 60)
    print("ATTACK PLAN (LEGACY)")
    print("=" * 60)
    print(f"Total Defenses Selected: {attack_plan.attack_count}")
    print(f"Total Threat Neutralized: {attack_plan.threat_neutralized:.2f}")
    print(f"Objective Value: {attack_plan.objective_value:.2f}")
    print()
    print("Priority Order:")
    print("-" * 60)
    for idx, defense in enumerate(attack_plan.selected_defenses, 1):
        print(f"{idx}. Defense ID {defense.id} ({defense.type})")
        print(f"   Threat Score: {defense.base_weight:.2f}")
        print(f"   Position: ({defense.x:.3f}, {defense.y:.3f})")
        print()
    print("=" * 60)
