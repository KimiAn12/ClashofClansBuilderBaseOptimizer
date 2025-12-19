"""
Spatial graph construction module for BASELINE.

Converts defense positions into graph-based representation for optimization.
"""

from .base_graph import (
    build_defense_graph,
    visualize_graph,
    get_graph_stats,
    euclidean_distance
)

__all__ = [
    'build_defense_graph',
    'visualize_graph',
    'get_graph_stats',
    'euclidean_distance'
]

