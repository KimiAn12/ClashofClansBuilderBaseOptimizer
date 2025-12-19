"""
Image preprocessing module for BASELINE.

This module handles basic image preprocessing operations:
- Loading images
- Resizing to fixed resolution
- Grayscale conversion
- Gaussian blur application
"""

import cv2
import numpy as np


def preprocess_image(
    image_path: str,
    target_size: tuple[int, int] = (1024, 1024),
    blur_kernel_size: int = 5
) -> np.ndarray:
    """
    Preprocess a raw image for template matching.
    
    Args:
        image_path: Path to the input image file
        target_size: Target resolution as (width, height). Default: (1024, 1024)
        blur_kernel_size: Size of Gaussian blur kernel (must be odd). Default: 5
    
    Returns:
        Preprocessed grayscale image as NumPy array
    
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If blur_kernel_size is even
    """
    # Validate blur kernel size (must be odd for Gaussian blur)
    if blur_kernel_size % 2 == 0:
        raise ValueError(f"blur_kernel_size must be odd, got {blur_kernel_size}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Resize to target resolution
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to grayscale
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    image_blurred = cv2.GaussianBlur(
        image_gray,
        (blur_kernel_size, blur_kernel_size),
        sigmaX=0
    )
    
    return image_blurred

