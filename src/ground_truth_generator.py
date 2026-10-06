"""
Ground truth generation from polygon masks
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, binary_dilation


def boundary_from_mask(mask, method='canny', dilate_iter=1):
    """
    Extract boundary from polygon mask
    
    Args:
        mask: Binary mask (height, width)
        method: 'canny', 'edge', or 'morphological'
        dilate_iter: Dilation iterations for boundary thickening
    
    Returns:
        boundary: Binary boundary mask
    """
    mask = mask.astype(np.uint8)
    
    if method == 'canny':
        # Canny edge detection
        boundary = cv2.Canny(mask * 255, 50, 150)
        boundary = (boundary > 0).astype(np.uint8)
    
    elif method == 'edge':
        # Sobel edge detection
        sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
        boundary = np.sqrt(sobelx**2 + sobely**2) > 0
        boundary = boundary.astype(np.uint8)
    
    elif method == 'morphological':
        # Morphological edge detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask, kernel)
        boundary = (mask - eroded).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Dilate boundary for better supervision
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary = cv2.dilate(boundary, kernel, iterations=dilate_iter)
    
    return boundary.astype(np.uint8)


def generate_distance_map(mask):
    """
    Generate distance transform for boundary weighting
    
    Args:
        mask: Binary mask (height, width)
    
    Returns:
        distance_map: Distance from each pixel to nearest boundary
    """
    boundary = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    distance_map = distance_transform_edt(1 - (boundary > 0))
    return distance_map.astype(np.float32)


def generate_multi_class_labels(field_mask, boundary_dilate=2):
    """
    Generate multi-class labels: background (0), field interior (1), boundary (2)
    
    Args:
        field_mask: Binary field mask
        boundary_dilate: Dilation for boundary class
    
    Returns:
        labels: Multi-class label map (0, 1, 2)
    """
    labels = np.zeros_like(field_mask, dtype=np.uint8)
    
    # Field interior
    labels[field_mask > 0] = 1
    
    # Boundary (erode field to get interior only)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    interior = cv2.erode(field_mask, kernel, iterations=1)
    boundary_mask = (field_mask - interior).astype(np.uint8)
    
    if boundary_dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=boundary_dilate)
    
    labels[boundary_mask > 0] = 2
    
    return labels


def create_weighted_mask(field_mask, boundary_weight=2.0):
    """
    Create sample weights for loss function
    Boundary pixels get higher weight
    
    Args:
        field_mask: Binary field mask
        boundary_weight: Weight multiplier for boundary pixels
    
    Returns:
        weights: Weight map (1.0 for field, boundary_weight for boundary)
    """
    boundary = boundary_from_mask(field_mask, method='canny', dilate_iter=1)
    weights = np.ones_like(field_mask, dtype=np.float32)
    weights[boundary > 0] = boundary_weight
    return weights
