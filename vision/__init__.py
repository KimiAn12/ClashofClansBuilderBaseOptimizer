"""
Computer vision module for BASELINE.

Handles image preprocessing, template matching, and postprocessing operations.
"""

from .preprocess import preprocess_image
from .template_matcher import detect_defenses, load_templates, match_template
from .postprocess import deduplicate_detections, cluster_detections, keep_best_detection_per_cluster

__all__ = [
    'preprocess_image',
    'detect_defenses',
    'load_templates',
    'match_template',
    'deduplicate_detections',
    'cluster_detections',
    'keep_best_detection_per_cluster'
]

