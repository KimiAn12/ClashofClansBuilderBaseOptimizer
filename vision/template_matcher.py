"""
Template matching module for BASELINE.

This module handles detection of defense structures using OpenCV template matching.
Uses normalized cross-correlation (TM_CCOEFF_NORMED) for robust matching.
Improved with multi-scale matching and non-maximum suppression.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def preprocess_template(template: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Preprocess a template image to match the preprocessing applied to input images.
    
    Args:
        template: Grayscale template image
        target_size: Optional target size. If None, keeps original size.
    
    Returns:
        Preprocessed template image
    """
    # Convert to grayscale if needed
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Resize if target size specified
    if target_size is not None:
        template = cv2.resize(template, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Apply same blur as input preprocessing
    template = cv2.GaussianBlur(template, (5, 5), sigmaX=0)
    
    return template


def load_templates(templates_dir: str, preprocess: bool = True) -> Dict[str, np.ndarray]:
    """
    Load all template images from the templates directory and subdirectories.
    
    Template files should be named with the defense type (e.g., "cannon.png", 
    "archer_tower.jpg"). The defense type is derived from the filename without extension.
    Searches recursively in subdirectories (e.g., templates/defensive_type/).
    
    Args:
        templates_dir: Path to directory containing template images
        preprocess: If True, preprocess templates to match input preprocessing. Default: True
    
    Returns:
        Dictionary mapping defense_type (str) to template image (grayscale numpy array)
    
    Raises:
        FileNotFoundError: If templates directory does not exist
        ValueError: If no template images are found
    """
    templates_path = Path(templates_dir)
    
    if not templates_path.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
    
    if not templates_path.is_dir():
        raise ValueError(f"Path is not a directory: {templates_dir}")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    templates = {}
    
    # Load all image files from the directory and subdirectories
    for file_path in templates_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # Extract defense type from filename (without extension)
            defense_type = file_path.stem
            
            # Handle common naming variations
            # mutli_mortar -> multi_mortar (fix typo)
            if defense_type == 'mutli_mortar':
                defense_type = 'multi_mortar'
            # firecracker -> firecrackers (standardize)
            elif defense_type == 'firecracker':
                defense_type = 'firecrackers'
            
            # Load template image
            template = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            
            if template is None:
                # Skip files that couldn't be loaded as images
                continue
            
            # Preprocess template if requested
            if preprocess:
                template = preprocess_template(template)
            
            # If duplicate defense_type found, keep the first one
            if defense_type not in templates:
                templates[defense_type] = template
    
    if not templates:
        raise ValueError(f"No valid template images found in {templates_dir}")
    
    return templates


def non_maximum_suppression(
    matches: List[Tuple[int, int, float]],
    template_width: int,
    template_height: int,
    overlap_threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """
    Apply non-maximum suppression to remove overlapping detections.
    
    Args:
        matches: List of (x, y, score) tuples
        template_width: Width of the template
        template_height: Height of the template
        overlap_threshold: IoU threshold for considering detections as overlapping. Default: 0.5
    
    Returns:
        Filtered list of matches after NMS
    """
    if not matches:
        return []
    
    # Sort by score (highest first)
    matches = sorted(matches, key=lambda m: m[2], reverse=True)
    
    suppressed = []
    
    for match in matches:
        x, y, score = match
        
        # Check overlap with already accepted matches
        overlap = False
        for accepted_x, accepted_y, _ in suppressed:
            # Calculate bounding boxes
            box1 = (x, y, x + template_width, y + template_height)
            box2 = (accepted_x, accepted_y, accepted_x + template_width, accepted_y + template_height)
            
            # Calculate IoU
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = template_width * template_height
                area2 = template_width * template_height
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > overlap_threshold:
                    overlap = True
                    break
        
        if not overlap:
            suppressed.append(match)
    
    return suppressed


def match_template_multi_scale(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.75,
    scales: List[float] = None,
    min_template_size: int = 20,
    max_template_size: int = 500
) -> List[Tuple[int, int, float]]:
    """
    Match a template against an image using multi-scale template matching.
    
    Args:
        image: Grayscale input image (numpy array)
        template: Grayscale template image (numpy array)
        threshold: Minimum match score (0.0 to 1.0). Default: 0.75
        scales: List of scale factors to try. If None, uses [0.8, 0.9, 1.0, 1.1, 1.2]
        min_template_size: Minimum template size in pixels. Default: 20
        max_template_size: Maximum template size in pixels. Default: 500
    
    Returns:
        List of (x, y, score) tuples for all matches above threshold.
        x, y are pixel coordinates of the top-left corner of the match.
    """
    if scales is None:
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    image_height, image_width = image.shape[:2]
    template_height, template_width = template.shape[:2]
    
    all_matches = []
    
    for scale in scales:
        # Calculate scaled template size
        scaled_width = int(template_width * scale)
        scaled_height = int(template_height * scale)
        
        # Skip if template too small or too large
        if scaled_width < min_template_size or scaled_height < min_template_size:
            continue
        if scaled_width > max_template_size or scaled_height > max_template_size:
            continue
        if scaled_width > image_width or scaled_height > image_height:
            continue
        
        # Resize template
        scaled_template = cv2.resize(template, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        
        # Perform template matching
        result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
        
        # Find matches above threshold
        locations = np.where(result >= threshold)
        
        for y, x in zip(locations[0], locations[1]):
            score = result[y, x]
            all_matches.append((x, y, float(score)))
    
    # Apply non-maximum suppression
    if all_matches:
        # Use average template size for NMS
        avg_width = int(template_width * np.mean(scales))
        avg_height = int(template_height * np.mean(scales))
        all_matches = non_maximum_suppression(all_matches, avg_width, avg_height)
    
    return all_matches


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.75,
    use_multi_scale: bool = True
) -> List[Tuple[int, int, float]]:
    """
    Match a single template against an image using normalized cross-correlation.
    
    Args:
        image: Grayscale input image (numpy array)
        template: Grayscale template image (numpy array)
        threshold: Minimum match score (0.0 to 1.0). Default: 0.75
        use_multi_scale: If True, use multi-scale matching. Default: True
    
    Returns:
        List of (x, y, score) tuples for all matches above threshold.
        x, y are pixel coordinates of the top-left corner of the match.
    """
    if use_multi_scale:
        return match_template_multi_scale(image, template, threshold)
    else:
        # Original single-scale matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        
        matches = []
        for y, x in zip(locations[0], locations[1]):
            score = result[y, x]
            matches.append((x, y, float(score)))
        
        # Apply NMS even for single-scale
        template_height, template_width = template.shape[:2]
        matches = non_maximum_suppression(matches, template_width, template_height)
        
        return matches


def detect_defenses(
    image: np.ndarray,
    templates_dir: str = "templates",
    confidence_threshold: float = 0.75,
    use_multi_scale: bool = True,
    defense_specific_thresholds: Dict[str, float] = None
) -> List[Dict[str, float]]:
    """
    Detect all defense structures in an image using template matching.
    
    Args:
        image: Preprocessed grayscale image (numpy array)
        templates_dir: Path to directory containing template images. Default: "templates"
        confidence_threshold: Minimum match score (0.0 to 1.0). Default: 0.75
        use_multi_scale: If True, use multi-scale matching. Default: True
        defense_specific_thresholds: Optional dict mapping defense_type to custom threshold.
                                    Overrides confidence_threshold for specific defenses.
    
    Returns:
        List of detection dictionaries, each containing:
            - defense_type: Name of the defense (str)
            - x_center: Normalized x-coordinate of center [0, 1] (float)
            - y_center: Normalized y-coordinate of center [0, 1] (float)
            - match_score: Match confidence score [0, 1] (float)
    
    Raises:
        FileNotFoundError: If templates directory does not exist
        ValueError: If no templates found or threshold is invalid
    """
    # Validate threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")
    
    # Get image dimensions for coordinate normalization
    image_height, image_width = image.shape[:2]
    
    # Load all templates
    templates = load_templates(templates_dir, preprocess=True)
    
    if defense_specific_thresholds is None:
        defense_specific_thresholds = {}
    
    all_detections = []
    
    # Match each template against the image
    for defense_type, template in templates.items():
        # Use defense-specific threshold if available, otherwise use default
        threshold = defense_specific_thresholds.get(defense_type, confidence_threshold)
        
        # Get template dimensions
        template_height, template_width = template.shape[:2]
        
        # Skip if template is larger than image
        if template_width > image_width or template_height > image_height:
            continue
        
        # Find all matches for this template
        matches = match_template(image, template, threshold, use_multi_scale=use_multi_scale)
        
        # Convert matches to detections with normalized coordinates
        for x_top_left, y_top_left, score in matches:
            # Calculate center coordinates in pixels
            x_center_px = x_top_left + (template_width / 2.0)
            y_center_px = y_top_left + (template_height / 2.0)
            
            # Normalize coordinates to [0, 1]
            x_center_norm = x_center_px / image_width
            y_center_norm = y_center_px / image_height
            
            # Clamp to [0, 1] to handle edge cases
            x_center_norm = max(0.0, min(1.0, x_center_norm))
            y_center_norm = max(0.0, min(1.0, y_center_norm))
            
            detection = {
                'defense_type': defense_type,
                'x_center': x_center_norm,
                'y_center': y_center_norm,
                'match_score': score
            }
            
            all_detections.append(detection)
    
    return all_detections

