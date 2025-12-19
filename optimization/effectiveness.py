"""
Effectiveness matrix module for BASELINE.

Defines heuristic effectiveness coefficients e[i,j] representing how effective
troop type i is against defense type j. Values are in [0, 1] where:
- 1.0 = maximum effectiveness
- 0.0 = no effectiveness

These are deterministic heuristics, not physics-based calculations.
Air/ground interactions are automatically handled: if a defense cannot attack
a troop type (e.g., ground-only defense vs air troop), the troop is highly
effective (effectiveness = 1.0).
"""

from typing import Dict, Tuple
from data.schema import get_defense_attack_type
from data.troops import get_troop_type, get_hero_type


# Effectiveness matrix: e[troop_type, defense_type] -> effectiveness [0, 1]
# Higher values indicate better matchup
EFFECTIVENESS_MATRIX: Dict[Tuple[str, str], float] = {
    # Barbarians: Good against ground defenses, weak against splash
    ('barbarian', 'cannon'): 0.7,
    ('barbarian', 'archer_tower'): 0.6,
    ('barbarian', 'crusher'): 0.3,  # Weak against melee defenses
    ('barbarian', 'giant_cannon'): 0.4,  # Weak against splash
    ('barbarian', 'multi_mortar'): 0.3,
    ('barbarian', 'double_cannon'): 0.5,
    ('barbarian', 'guard_post'): 0.8,
    
    # Archers: Good range, weak health
    ('archer', 'cannon'): 0.8,
    ('archer', 'archer_tower'): 0.7,
    ('archer', 'crusher'): 0.9,  # Can attack from range
    ('archer', 'giant_cannon'): 0.5,
    ('archer', 'multi_mortar'): 0.4,
    ('archer', 'double_cannon'): 0.6,
    ('archer', 'guard_post'): 0.7,
    
    # Giants: High health tanks, moderate damage
    ('giant', 'cannon'): 0.9,
    ('giant', 'archer_tower'): 0.8,
    ('giant', 'crusher'): 0.4,  # Weak against melee
    ('giant', 'giant_cannon'): 0.6,
    ('giant', 'multi_mortar'): 0.5,
    ('giant', 'double_cannon'): 0.7,
    ('giant', 'guard_post'): 0.8,
    
    # Minions: Air troops, good against ground-only defenses
    ('minion', 'cannon'): 0.9,
    ('minion', 'archer_tower'): 0.6,  # Can target air
    ('minion', 'crusher'): 1.0,  # Cannot target air
    ('minion', 'giant_cannon'): 0.7,
    ('minion', 'multi_mortar'): 0.5,
    ('minion', 'double_cannon'): 0.8,
    ('minion', 'guard_post'): 0.7,
    
    # Bombers: High damage, good against walls and buildings
    ('bomber', 'cannon'): 0.8,
    ('bomber', 'archer_tower'): 0.7,
    ('bomber', 'crusher'): 0.5,
    ('bomber', 'giant_cannon'): 0.6,
    ('bomber', 'multi_mortar'): 0.5,
    ('bomber', 'double_cannon'): 0.7,
    ('bomber', 'guard_post'): 0.8,
    
    # Cannon Cart: High DPS, good range
    ('cannon_cart', 'cannon'): 0.9,
    ('cannon_cart', 'archer_tower'): 0.8,
    ('cannon_cart', 'crusher'): 0.7,
    ('cannon_cart', 'giant_cannon'): 0.7,
    ('cannon_cart', 'multi_mortar'): 0.6,
    ('cannon_cart', 'double_cannon'): 0.8,
    ('cannon_cart', 'guard_post'): 0.9,
    
    # Night Witch: Spawns bats, good against single targets
    ('night_witch', 'cannon'): 0.8,
    ('night_witch', 'archer_tower'): 0.7,
    ('night_witch', 'crusher'): 0.6,
    ('night_witch', 'giant_cannon'): 0.6,
    ('night_witch', 'multi_mortar'): 0.5,
    ('night_witch', 'double_cannon'): 0.7,
    ('night_witch', 'guard_post'): 0.8,
    
    # Drop Ship: Air transport, good mobility
    ('drop_ship', 'cannon'): 0.8,
    ('drop_ship', 'archer_tower'): 0.6,
    ('drop_ship', 'crusher'): 1.0,  # Cannot target air
    ('drop_ship', 'giant_cannon'): 0.7,
    ('drop_ship', 'multi_mortar'): 0.5,
    ('drop_ship', 'double_cannon'): 0.7,
    ('drop_ship', 'guard_post'): 0.7,
    
    # Super Pekka: Very high DPS and health
    ('super_pekka', 'cannon'): 1.0,
    ('super_pekka', 'archer_tower'): 0.9,
    ('super_pekka', 'crusher'): 0.7,
    ('super_pekka', 'giant_cannon'): 0.8,
    ('super_pekka', 'multi_mortar'): 0.7,
    ('super_pekka', 'double_cannon'): 0.9,
    ('super_pekka', 'guard_post'): 1.0,
    
    # Hog Glider: Fast, high damage
    ('hog_glider', 'cannon'): 0.9,
    ('hog_glider', 'archer_tower'): 0.8,
    ('hog_glider', 'crusher'): 0.6,
    ('hog_glider', 'giant_cannon'): 0.7,
    ('hog_glider', 'multi_mortar'): 0.6,
    ('hog_glider', 'double_cannon'): 0.8,
    ('hog_glider', 'guard_post'): 0.9,
    ('hog_glider', 'x_bow'): 0.6,
    ('hog_glider', 'mega_tesla'): 0.5,
    ('hog_glider', 'lava_launcher'): 0.5,
    ('hog_glider', 'roaster'): 0.5,
    
    # New defensive buildings effectiveness (x_bow, mega_tesla, lava_launcher, roaster)
    # Barbarians vs new defenses
    ('barbarian', 'x_bow'): 0.3,
    ('barbarian', 'mega_tesla'): 0.3,
    ('barbarian', 'lava_launcher'): 0.3,
    ('barbarian', 'roaster'): 0.3,
    
    # Archers vs new defenses
    ('archer', 'x_bow'): 0.5,
    ('archer', 'mega_tesla'): 0.4,
    ('archer', 'lava_launcher'): 0.4,
    ('archer', 'roaster'): 0.4,
    
    # Giants vs new defenses
    ('giant', 'x_bow'): 0.6,
    ('giant', 'mega_tesla'): 0.5,
    ('giant', 'lava_launcher'): 0.5,
    ('giant', 'roaster'): 0.5,
    
    # Minions vs new defenses
    ('minion', 'x_bow'): 0.7,
    ('minion', 'mega_tesla'): 0.6,
    ('minion', 'lava_launcher'): 0.5,
    ('minion', 'roaster'): 0.5,
    
    # Bombers vs new defenses
    ('bomber', 'x_bow'): 0.6,
    ('bomber', 'mega_tesla'): 0.5,
    ('bomber', 'lava_launcher'): 0.5,
    ('bomber', 'roaster'): 0.5,
    
    # Cannon Cart vs new defenses
    ('cannon_cart', 'x_bow'): 0.7,
    ('cannon_cart', 'mega_tesla'): 0.6,
    ('cannon_cart', 'lava_launcher'): 0.6,
    ('cannon_cart', 'roaster'): 0.6,
    
    # Night Witch vs new defenses
    ('night_witch', 'x_bow'): 0.6,
    ('night_witch', 'mega_tesla'): 0.5,
    ('night_witch', 'lava_launcher'): 0.5,
    ('night_witch', 'roaster'): 0.5,
    
    # Drop Ship vs new defenses
    ('drop_ship', 'x_bow'): 0.7,
    ('drop_ship', 'mega_tesla'): 0.6,
    ('drop_ship', 'lava_launcher'): 0.5,
    ('drop_ship', 'roaster'): 0.5,
    
    # Super Pekka vs new defenses
    ('super_pekka', 'x_bow'): 0.8,
    ('super_pekka', 'mega_tesla'): 0.7,
    ('super_pekka', 'lava_launcher'): 0.7,
    ('super_pekka', 'roaster'): 0.7,
    
    # Heroes: High DPS and health, generally effective against all defenses
    ('battle_machine', 'cannon'): 1.0,
    ('battle_machine', 'archer_tower'): 0.9,
    ('battle_machine', 'crusher'): 0.8,
    ('battle_machine', 'giant_cannon'): 0.9,
    ('battle_machine', 'multi_mortar'): 0.8,
    ('battle_machine', 'double_cannon'): 0.9,
    ('battle_machine', 'guard_post'): 1.0,
    
    ('builder_king', 'cannon'): 0.95,
    ('builder_king', 'archer_tower'): 0.9,
    ('builder_king', 'crusher'): 0.75,
    ('builder_king', 'giant_cannon'): 0.85,
    ('builder_king', 'multi_mortar'): 0.8,
    ('builder_king', 'double_cannon'): 0.9,
    ('builder_king', 'guard_post'): 0.95,
    ('builder_king', 'x_bow'): 0.85,
    ('builder_king', 'mega_tesla'): 0.8,
    ('builder_king', 'lava_launcher'): 0.8,
    ('builder_king', 'roaster'): 0.8,
    
    # Heroes vs new defenses
    ('battle_machine', 'x_bow'): 0.9,
    ('battle_machine', 'mega_tesla'): 0.85,
    ('battle_machine', 'lava_launcher'): 0.85,
    ('battle_machine', 'roaster'): 0.85,
    
    # Resource and utility buildings (non-defensive, all troops effective)
    # These buildings don't attack, so troops are generally effective (0.7-0.85)
    # Melee troops (barbarian, giant, super_pekka, heroes) - 0.8
    ('barbarian', 'gold_mine'): 0.8,
    ('barbarian', 'elixir_collector'): 0.8,
    ('barbarian', 'gem_mine'): 0.8,
    ('barbarian', 'gold_storage'): 0.8,
    ('barbarian', 'elixir_storage'): 0.8,
    ('barbarian', 'barracks'): 0.8,
    ('barbarian', 'army_camp'): 0.8,
    ('barbarian', 'laboratory'): 0.8,
    ('barbarian', 'clock_tower'): 0.8,
    ('barbarian', 'healing_hut'): 0.8,
    
    ('giant', 'gold_mine'): 0.8,
    ('giant', 'elixir_collector'): 0.8,
    ('giant', 'gem_mine'): 0.8,
    ('giant', 'gold_storage'): 0.8,
    ('giant', 'elixir_storage'): 0.8,
    ('giant', 'barracks'): 0.8,
    ('giant', 'army_camp'): 0.8,
    ('giant', 'laboratory'): 0.8,
    ('giant', 'clock_tower'): 0.8,
    ('giant', 'healing_hut'): 0.8,
    
    ('super_pekka', 'gold_mine'): 0.85,
    ('super_pekka', 'elixir_collector'): 0.85,
    ('super_pekka', 'gem_mine'): 0.85,
    ('super_pekka', 'gold_storage'): 0.85,
    ('super_pekka', 'elixir_storage'): 0.85,
    ('super_pekka', 'barracks'): 0.85,
    ('super_pekka', 'army_camp'): 0.85,
    ('super_pekka', 'laboratory'): 0.85,
    ('super_pekka', 'clock_tower'): 0.85,
    ('super_pekka', 'healing_hut'): 0.85,
    
    # Ranged troops (archer, minion, cannon_cart) - 0.75
    ('archer', 'gold_mine'): 0.75,
    ('archer', 'elixir_collector'): 0.75,
    ('archer', 'gem_mine'): 0.75,
    ('archer', 'gold_storage'): 0.75,
    ('archer', 'elixir_storage'): 0.75,
    ('archer', 'barracks'): 0.75,
    ('archer', 'army_camp'): 0.75,
    ('archer', 'laboratory'): 0.75,
    ('archer', 'clock_tower'): 0.75,
    ('archer', 'healing_hut'): 0.75,
    
    ('minion', 'gold_mine'): 0.75,
    ('minion', 'elixir_collector'): 0.75,
    ('minion', 'gem_mine'): 0.75,
    ('minion', 'gold_storage'): 0.75,
    ('minion', 'elixir_storage'): 0.75,
    ('minion', 'barracks'): 0.75,
    ('minion', 'army_camp'): 0.75,
    ('minion', 'laboratory'): 0.75,
    ('minion', 'clock_tower'): 0.75,
    ('minion', 'healing_hut'): 0.75,
    
    ('cannon_cart', 'gold_mine'): 0.75,
    ('cannon_cart', 'elixir_collector'): 0.75,
    ('cannon_cart', 'gem_mine'): 0.75,
    ('cannon_cart', 'gold_storage'): 0.75,
    ('cannon_cart', 'elixir_storage'): 0.75,
    ('cannon_cart', 'barracks'): 0.75,
    ('cannon_cart', 'army_camp'): 0.75,
    ('cannon_cart', 'laboratory'): 0.75,
    ('cannon_cart', 'clock_tower'): 0.75,
    ('cannon_cart', 'healing_hut'): 0.75,
    
    # Bombers excel against buildings - 0.85
    ('bomber', 'gold_mine'): 0.85,
    ('bomber', 'elixir_collector'): 0.85,
    ('bomber', 'gem_mine'): 0.85,
    ('bomber', 'gold_storage'): 0.85,
    ('bomber', 'elixir_storage'): 0.85,
    ('bomber', 'barracks'): 0.85,
    ('bomber', 'army_camp'): 0.85,
    ('bomber', 'laboratory'): 0.85,
    ('bomber', 'clock_tower'): 0.85,
    ('bomber', 'healing_hut'): 0.85,
    
    # Specialized troops (night_witch, drop_ship, hog_glider) - 0.7
    ('night_witch', 'gold_mine'): 0.7,
    ('night_witch', 'elixir_collector'): 0.7,
    ('night_witch', 'gem_mine'): 0.7,
    ('night_witch', 'gold_storage'): 0.7,
    ('night_witch', 'elixir_storage'): 0.7,
    ('night_witch', 'barracks'): 0.7,
    ('night_witch', 'army_camp'): 0.7,
    ('night_witch', 'laboratory'): 0.7,
    ('night_witch', 'clock_tower'): 0.7,
    ('night_witch', 'healing_hut'): 0.7,
    
    ('drop_ship', 'gold_mine'): 0.7,
    ('drop_ship', 'elixir_collector'): 0.7,
    ('drop_ship', 'gem_mine'): 0.7,
    ('drop_ship', 'gold_storage'): 0.7,
    ('drop_ship', 'elixir_storage'): 0.7,
    ('drop_ship', 'barracks'): 0.7,
    ('drop_ship', 'army_camp'): 0.7,
    ('drop_ship', 'laboratory'): 0.7,
    ('drop_ship', 'clock_tower'): 0.7,
    ('drop_ship', 'healing_hut'): 0.7,
    
    ('hog_glider', 'gold_mine'): 0.7,
    ('hog_glider', 'elixir_collector'): 0.7,
    ('hog_glider', 'gem_mine'): 0.7,
    ('hog_glider', 'gold_storage'): 0.7,
    ('hog_glider', 'elixir_storage'): 0.7,
    ('hog_glider', 'barracks'): 0.7,
    ('hog_glider', 'army_camp'): 0.7,
    ('hog_glider', 'laboratory'): 0.7,
    ('hog_glider', 'clock_tower'): 0.7,
    ('hog_glider', 'healing_hut'): 0.7,
    
    # Heroes - 0.8-0.85
    ('battle_machine', 'gold_mine'): 0.85,
    ('battle_machine', 'elixir_collector'): 0.85,
    ('battle_machine', 'gem_mine'): 0.85,
    ('battle_machine', 'gold_storage'): 0.85,
    ('battle_machine', 'elixir_storage'): 0.85,
    ('battle_machine', 'barracks'): 0.85,
    ('battle_machine', 'army_camp'): 0.85,
    ('battle_machine', 'laboratory'): 0.85,
    ('battle_machine', 'clock_tower'): 0.85,
    ('battle_machine', 'healing_hut'): 0.85,
    
    ('builder_king', 'gold_mine'): 0.8,
    ('builder_king', 'elixir_collector'): 0.8,
    ('builder_king', 'gem_mine'): 0.8,
    ('builder_king', 'gold_storage'): 0.8,
    ('builder_king', 'elixir_storage'): 0.8,
    ('builder_king', 'barracks'): 0.8,
    ('builder_king', 'army_camp'): 0.8,
    ('builder_king', 'laboratory'): 0.8,
    ('builder_king', 'clock_tower'): 0.8,
    ('builder_king', 'healing_hut'): 0.8,
}


def get_effectiveness(troop_type: str, defense_type: str) -> float:
    """
    Get effectiveness coefficient for a troop-defense matchup.
    
    This function accounts for air/ground interactions:
    - If defense can only attack ground and troop is air -> troop is immune (effectiveness = 1.0)
    - If defense can only attack air and troop is ground -> troop is immune (effectiveness = 1.0)
    - Otherwise, uses the effectiveness matrix
    
    Args:
        troop_type: Troop type name
        defense_type: Defense type name
    
    Returns:
        Effectiveness coefficient in [0, 1]. Default: 0.5 if not found.
    """
    # Get defense attack capability
    defense_attack_type = get_defense_attack_type(defense_type)
    
    # Get troop movement type (try troop first, then hero)
    try:
        troop = get_troop_type(troop_type)
        troop_movement_type = troop.movement_type
    except KeyError:
        try:
            hero = get_hero_type(troop_type)
            troop_movement_type = hero.movement_type
        except KeyError:
            # Unknown troop type, use matrix lookup
            key = (troop_type.lower(), defense_type.lower())
            return EFFECTIVENESS_MATRIX.get(key, 0.5)
    
    # Check if defense can attack this troop type
    if defense_attack_type == 'ground' and troop_movement_type == 'air':
        # Defense can only attack ground, but troop is air -> troop is immune
        return 1.0
    elif defense_attack_type == 'air' and troop_movement_type == 'ground':
        # Defense can only attack air, but troop is ground -> troop is immune
        return 1.0
    else:
        # Defense can attack this troop type (both, or matching type)
        # Use the effectiveness matrix
        key = (troop_type.lower(), defense_type.lower())
        return EFFECTIVENESS_MATRIX.get(key, 0.5)  # Default moderate effectiveness


def get_effectiveness_matrix() -> Dict[Tuple[str, str], float]:
    """
    Get the complete effectiveness matrix.
    
    Returns:
        Dictionary mapping (troop_type, defense_type) -> effectiveness
    """
    return EFFECTIVENESS_MATRIX.copy()

