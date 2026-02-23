
import time
from contextlib import contextmanager
import cv2
import numpy as np

class Benchmark:
    def __init__(self):
        self.metrics = {}
    
    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            if name in self.metrics:
                self.metrics[name] += duration_ms
            else:
                self.metrics[name] = duration_ms

    def get_metrics(self):
        # Round metrics for display
        return {k: round(v, 2) for k, v in self.metrics.items()}

    def reset(self):
        self.metrics = {}

def draw_masks(image_path, masks_data, boxes_data=None):
    """
    Draws masks and bounding boxes on the image.
    args:
        image_path: str or numpy array of the image
        masks_data: list of masks (binary numpy arrays)
        boxes_data: list of boxes [x1, y1, x2, y2]
    returns:
        annotated_image: numpy array
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path.copy()
    
    # Draw boxes
    if boxes_data is not None:
        for box in boxes_data:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
    # Draw masks with transparency
    if masks_data is not None and len(masks_data) > 0:
        mask_overlay = image.copy()
        for mask in masks_data:
            # Color for masks (e.g., Green)
            color = np.array([0, 255, 0], dtype=np.uint8)
            
            # Mask acts as alpha
            # Ensure mask is boolean or 0/1
            if mask.dtype != bool:
                mask = mask > 0.5
                
            mask_overlay[mask] = color
            
        alpha = 0.5
        image = cv2.addWeighted(mask_overlay, alpha, image, 1 - alpha, 0)
        
    return image
