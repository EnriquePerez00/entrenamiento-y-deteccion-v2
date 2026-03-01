# -*- coding: utf-8 -*-
"""
color_classifier.py – Post-Inference LEGO Color Classifier.

Given a YOLO crop of a detected LEGO piece, this module:
  1. Extracts the foreground by removing the background (HSV thresholding).
  2. Finds the dominant color via K-Means clustering.
  3. Matches it against the official LDraw color table using Delta E (CIE76).

Designed for a fixed camera/lighting setup where color consistency is high.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class LegoColorClassifier:
    """
    Classifies the official LEGO color from a cropped piece image.
    
    Usage:
        classifier = LegoColorClassifier()
        result = classifier.classify(crop_bgr)
        print(result)  
        # {'color_id': 4, 'color_name': 'Red', 'confidence': 0.95, 'delta_e': 3.2, 'dominant_rgb': (180, 0, 0)}
    """
    
    def __init__(self):
        # Lazy import to avoid circular dependency
        from src.logic.lego_colors import LDRAW_COLORS, classify_color as _classify
        self._colors = LDRAW_COLORS
        self._classify_fn = _classify
    
    def classify(self, crop_image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Classify the LEGO color from a BGR crop image.
        
        Args:
            crop_image: BGR numpy array (from OpenCV / YOLO crop).
            mask: Optional binary mask (255=foreground, 0=background).
                  If None, the mask is auto-generated via HSV thresholding.
        
        Returns:
            Dict with keys: color_id, color_name, confidence, delta_e, dominant_rgb.
        """
        if crop_image is None or crop_image.size == 0:
            return self._empty_result()
        
        # 1. Generate foreground mask if not provided
        if mask is None:
            mask = self._generate_mask(crop_image)
        
        # 2. Extract foreground pixels
        fg_pixels = crop_image[mask > 0]
        
        if len(fg_pixels) < 10:
            # Too few foreground pixels – fallback to full image
            fg_pixels = crop_image.reshape(-1, 3)
        
        # 3. Find dominant color via K-Means
        dominant_bgr = self._dominant_color(fg_pixels, k=3)
        
        # Convert BGR → RGB for classification
        dominant_rgb = (int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0]))
        
        # 4. Classify against LDraw table
        matches = self._classify_fn(dominant_rgb, top_k=3)
        
        if not matches:
            return self._empty_result()
        
        best = matches[0]
        return {
            "color_id": best["color_id"],
            "color_name": best["name"],
            "confidence": best["confidence"],
            "delta_e": round(best["delta_e"], 2),
            "dominant_rgb": dominant_rgb,
            "alternatives": matches[1:] if len(matches) > 1 else [],
        }
    
    def classify_batch(self, crops: list) -> list:
        """Classify a batch of crops. Returns list of result dicts."""
        return [self.classify(crop) for crop in crops]
    
    def _generate_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Generate a foreground mask using HSV thresholding.
        Assumes the background is a neutral surface (grey/white/black).
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Strategy: foreground is anything with reasonable saturation OR 
        # is very dark (black pieces) or has distinct value from background.
        # For controlled lighting, we use a combination:
        
        # Mask 1: Saturated pixels (colored pieces)
        sat_mask = (s > 25).astype(np.uint8) * 255
        
        # Mask 2: Very dark pixels (black pieces, even with low saturation)
        dark_mask = (v < 60).astype(np.uint8) * 255
        
        # Mask 3: Very bright pixels that are NOT the white background
        # (white pieces are tricky – we use edge detection as fallback)
        
        # Combine
        combined = cv2.bitwise_or(sat_mask, dark_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # If mask is too small (< 5% of image), use the full image
        if combined.sum() < 0.05 * 255 * combined.size:
            combined = np.ones_like(combined) * 255
        
        return combined
    
    def _dominant_color(self, pixels: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Find the dominant color in a set of pixels using K-Means.
        Returns the BGR center of the largest cluster.
        """
        if len(pixels) < k:
            return np.mean(pixels, axis=0).astype(np.uint8)
        
        # Subsample for speed (max 5000 pixels)
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]
        
        pixels_float = pixels.astype(np.float32)
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            kmeans.fit(pixels_float)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_idx = labels[np.argmax(counts)]
            dominant_center = kmeans.cluster_centers_[dominant_idx]
        except ImportError:
            # Fallback: simple median-based dominant color (no sklearn needed)
            dominant_center = np.median(pixels_float, axis=0)
        
        return dominant_center.astype(np.uint8)
    
    def _empty_result(self) -> Dict:
        """Return a safe empty result."""
        return {
            "color_id": -1,
            "color_name": "Unknown",
            "confidence": 0.0,
            "delta_e": 999.0,
            "dominant_rgb": (0, 0, 0),
            "alternatives": [],
        }
