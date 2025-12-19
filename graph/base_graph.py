"""
Base graph construction module for BASELINE.

Constructs weighted undirected graphs from defense positions using NetworkX.
"""

import networkx as nx
import math
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data.schema import Defense


def euclidean_distance(def1: Defense, def2: Defense) -> float:
    """
    Calculate Euclidean distance between two defenses in normalized coordinate space.
    
    Args:
        def1: First Defense object
        def2: Second Defense object
    
    Returns:
        Euclidean distance in normalized coordinate space [0, sqrt(2)]
    """
    dx = def1.x - def2.x
    dy = def1.y - def2.y
    return math.sqrt(dx * dx + dy * dy)


def build_defense_graph(
    defenses: List[Defense],
    connection_radius: float = 0.2
) -> nx.Graph:
    """
    Build a weighted undirected graph from defense positions.
    
    Connects defenses that are within the specified radius. Edge weights
    are set to the Euclidean distance between connected defenses.
    
    Args:
        defenses: List of Defense objects
        connection_radius: Maximum distance for defenses to be connected.
                          Default: 0.2 (20% of normalized image size)
    
    Returns:
        NetworkX Graph object with:
            - Nodes: Defense IDs with attributes (type, x, y, base_weight)
            - Edges: Connections between nearby defenses with weight=distance
    
    Raises:
        ValueError: If connection_radius is not positive
    """
    if connection_radius <= 0:
        raise ValueError(f"connection_radius must be positive, got {connection_radius}")
    
    # Create empty undirected graph
    G = nx.Graph()
    
    # Add nodes (defenses) with their attributes
    for defense in defenses:
        G.add_node(
            defense.id,
            type=defense.type,
            x=defense.x,
            y=defense.y,
            base_weight=defense.base_weight
        )
    
    # Add edges between defenses within connection radius
    for i, def1 in enumerate(defenses):
        for def2 in defenses[i + 1:]:  # Avoid duplicate pairs
            distance = euclidean_distance(def1, def2)
            
            if distance <= connection_radius:
                # Add edge with weight equal to distance
                G.add_edge(def1.id, def2.id, weight=distance)
    
    return G


def visualize_graph(
    G: nx.Graph,
    title: str = "Defense Graph",
    figsize: tuple[int, int] = (10, 10),
    node_size: int = 300,
    font_size: int = 8,
    show_labels: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the defense graph using matplotlib.
    
    Args:
        G: NetworkX Graph object to visualize
        title: Plot title. Default: "Defense Graph"
        figsize: Figure size (width, height). Default: (10, 10)
        node_size: Size of nodes in the plot. Default: 300
        font_size: Font size for node labels. Default: 8
        show_labels: Whether to show node ID labels. Default: True
        save_path: Optional path to save the figure. If None, displays the plot.
                   Default: None
    """
    if G.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node positions from graph attributes
    pos = {}
    node_colors = []
    node_types = {}
    
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        pos[node_id] = (node_data['x'], node_data['y'])
        node_types[node_id] = node_data['type']
        
        # Assign colors based on defense type (simple hash-based coloring)
        type_hash = hash(node_data['type']) % 10
        node_colors.append(plt.cm.tab10(type_hash))
    
    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.5,
        width=1.0,
        edge_color='gray'
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.8
    )
    
    # Draw labels if requested
    if show_labels:
        labels = {node_id: f"{node_id}\n{node_types[node_id]}" for node_id in G.nodes()}
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            ax=ax,
            font_size=font_size
        )
    
    # Set plot properties
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Normalized X Coordinate", fontsize=10)
    ax.set_ylabel("Normalized Y Coordinate", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Create legend for defense types
    unique_types = set(node_types.values())
    legend_elements = [
        mpatches.Patch(
            facecolor=plt.cm.tab10(hash(def_type) % 10),
            label=def_type,
            alpha=0.8
        )
        for def_type in sorted(unique_types)
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graph visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_graph_stats(G: nx.Graph) -> dict:
    """
    Get basic statistics about the graph.
    
    Args:
        G: NetworkX Graph object
    
    Returns:
        Dictionary with graph statistics:
            - num_nodes: Number of nodes
            - num_edges: Number of edges
            - density: Graph density
            - avg_degree: Average node degree
            - avg_edge_weight: Average edge weight
    """
    if G.number_of_nodes() == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0.0,
            'avg_degree': 0.0,
            'avg_edge_weight': 0.0
        }
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Calculate density (for undirected graph)
    max_edges = num_nodes * (num_nodes - 1) / 2
    density = num_edges / max_edges if max_edges > 0 else 0.0
    
    # Calculate average degree
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0.0
    
    # Calculate average edge weight
    edge_weights = [data.get('weight', 0.0) for _, _, data in G.edges(data=True)]
    avg_edge_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0.0
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': avg_degree,
        'avg_edge_weight': avg_edge_weight
    }

