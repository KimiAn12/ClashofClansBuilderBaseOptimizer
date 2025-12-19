"""
Postprocessing module for BASELINE.

This module handles clustering of duplicate detections using simple
Euclidean distance-based grouping. Removes redundant detections by
keeping only the highest-confidence detection per cluster.
"""

import math
from typing import List, Dict


def euclidean_distance(det1: Dict, det2: Dict) -> float:
    """
    Calculate Euclidean distance between two detections in normalized coordinate space.
    
    Args:
        det1: First detection dictionary with 'x_center' and 'y_center'
        det2: Second detection dictionary with 'x_center' and 'y_center'
    
    Returns:
        Euclidean distance in normalized coordinate space [0, sqrt(2)]
    """
    dx = det1['x_center'] - det2['x_center']
    dy = det1['y_center'] - det2['y_center']
    return math.sqrt(dx * dx + dy * dy)


def cluster_detections(
    detections: List[Dict],
    distance_threshold: float = 0.05,
    same_type_only: bool = True
) -> List[List[Dict]]:
    """
    Group detections into clusters based on Euclidean distance.
    
    Uses a simple greedy clustering algorithm:
    - For each detection, find the nearest existing cluster
    - If within threshold, add to that cluster
    - Otherwise, create a new cluster
    - Optionally only cluster detections of the same defense type
    
    Args:
        detections: List of detection dictionaries
        distance_threshold: Maximum distance for detections to be in same cluster.
                           Default: 0.05 (5% of normalized image size)
        same_type_only: If True, only cluster detections of the same defense type. Default: True
    
    Returns:
        List of clusters, where each cluster is a list of detection dictionaries
    """
    if not detections:
        return []
    
    clusters = []
    
    # Process each detection
    for detection in detections:
        # Find the nearest cluster
        nearest_cluster_idx = None
        min_distance = float('inf')
        
        for cluster_idx, cluster in enumerate(clusters):
            # Check if same type required
            if same_type_only:
                cluster_type = cluster[0]['defense_type']
                if detection['defense_type'] != cluster_type:
                    continue
            
            # Calculate distance to cluster center (mean of all points in cluster)
            cluster_center = {
                'x_center': sum(d['x_center'] for d in cluster) / len(cluster),
                'y_center': sum(d['y_center'] for d in cluster) / len(cluster)
            }
            
            distance = euclidean_distance(detection, cluster_center)
            
            if distance < min_distance:
                min_distance = distance
                nearest_cluster_idx = cluster_idx
        
        # Add to nearest cluster if within threshold, otherwise create new cluster
        if nearest_cluster_idx is not None and min_distance <= distance_threshold:
            clusters[nearest_cluster_idx].append(detection)
        else:
            clusters.append([detection])
    
    return clusters


def keep_best_detection_per_cluster(clusters: List[List[Dict]]) -> List[Dict]:
    """
    From each cluster, keep only the detection with the highest match_score.
    
    Args:
        clusters: List of clusters, where each cluster is a list of detections
    
    Returns:
        List of best detections, one per cluster
    """
    best_detections = []
    
    for cluster in clusters:
        if not cluster:
            continue
        
        # Find detection with highest match_score
        best_detection = max(cluster, key=lambda d: d['match_score'])
        best_detections.append(best_detection)
    
    return best_detections


def deduplicate_detections(
    detections: List[Dict],
    distance_threshold: float = 0.05,
    same_type_only: bool = True
) -> List[Dict]:
    """
    Remove duplicate detections by clustering nearby detections and keeping
    only the highest-confidence detection per cluster.
    
    This function performs the complete postprocessing pipeline:
    1. Groups detections into clusters based on Euclidean distance
    2. Keeps only the best (highest confidence) detection from each cluster
    
    Args:
        detections: List of detection dictionaries, each containing:
            - defense_type: Name of the defense (str)
            - x_center: Normalized x-coordinate [0, 1] (float)
            - y_center: Normalized y-coordinate [0, 1] (float)
            - match_score: Match confidence score [0, 1] (float)
        distance_threshold: Maximum Euclidean distance (in normalized coordinates)
                           for detections to be considered duplicates.
                           Default: 0.05 (5% of image size)
        same_type_only: If True, only cluster detections of the same defense type. Default: True
    
    Returns:
        List of unique detection dictionaries, one per cluster, sorted by
        match_score (highest first)
    
    Example:
        >>> detections = [
        ...     {'defense_type': 'cannon', 'x_center': 0.5, 'y_center': 0.5, 'match_score': 0.9},
        ...     {'defense_type': 'cannon', 'x_center': 0.51, 'y_center': 0.51, 'match_score': 0.85},
        ...     {'defense_type': 'archer', 'x_center': 0.2, 'y_center': 0.2, 'match_score': 0.8}
        ... ]
        >>> unique = deduplicate_detections(detections, distance_threshold=0.05)
        >>> # Returns 2 detections: the cannon with score 0.9 and the archer
    """
    if not detections:
        return []
    
    # Validate distance threshold
    if distance_threshold <= 0:
        raise ValueError(f"distance_threshold must be positive, got {distance_threshold}")
    
    # Group detections into clusters
    clusters = cluster_detections(detections, distance_threshold, same_type_only=same_type_only)
    
    # Keep only the best detection from each cluster
    unique_detections = keep_best_detection_per_cluster(clusters)
    
    # Sort by match_score (highest first) for consistent output
    unique_detections.sort(key=lambda d: d['match_score'], reverse=True)
    
    return unique_detections

