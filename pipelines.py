
import numpy as np
import torch
from abc import ABC, abstractmethod
from ultralytics import YOLO
from utils import Benchmark

# Mock SAM 2 if not present, but assume it is installed as per instructions
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("SAM 2 not found, using MockSAM2")
    SAM2_AVAILABLE = False
    
    class SAM2ImagePredictor:
        def __init__(self, model):
            self.model = model
            
        def set_image(self, image):
            self.image_shape = image.shape[:2]
            
        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            # Mock prediction returning a square mask inside the box
            masks = []
            scores = []
            
            if box is not None:
                # Handle batches of boxes if necessary, but assume single for simplicity in mock unless batched
                # box shape might be (1, 4)
                if len(box.shape) == 1:
                    boxes = [box]
                else:
                    boxes = box
                
                for b in boxes:
                    x1, y1, x2, y2 = map(int, b)
                    mask = np.zeros(self.image_shape, dtype=bool)
                    mask[y1:y2, x1:x2] = True
                    masks.append(mask)
                    scores.append(0.95)
            
            return np.array(masks), np.array(scores), np.array([])

class PipelineStrategy(ABC):
    def __init__(self, model_name, sam_checkpoint):
        self.benchmark = Benchmark()
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        print(f"Loading YOLO Model {model_name} on {self.device}...")
        self.yolo_model = YOLO(model_name)
        
        self.sam_available = SAM2_AVAILABLE
        if self.sam_available:
            print(f"Loading SAM 2 Model {sam_checkpoint} on {self.device}...")
            # We assume config names match standard SAM2 configs or are just basenames
            # For simplicity, we use a mapping or pass config directly if file exists
            # Here we assume the checkpoint file implies the config or user provides paths.
            # In a real app we'd map 'sam2_hiera_base.pt' -> 'sam2_hiera_b.yaml'
            
            # Simple mapping for demo purposes (adjust paths as needed)
            config_mapping = {
                'sam2_hiera_base.pt': 'configs/sam2/sam2_hiera_b+.yaml',
                'sam2_hiera_large.pt': 'configs/sam2/sam2_hiera_l.yaml',
                'sam2_hiera_tiny.pt': 'configs/sam2/sam2_hiera_t.yaml',
                'sam2_hiera_small.pt': 'configs/sam2/sam2_hiera_s.yaml',
            }
            
            # If checkpoint filename is just the name, assume standard
            cfg = config_mapping.get(sam_checkpoint, 'configs/sam2/sam2_hiera_l.yaml')
            
            try:
                self.sam_model = build_sam2(cfg, sam_checkpoint, device=self.device)
                self.sam_predictor = SAM2ImagePredictor(self.sam_model)
            except Exception as e:
                 print(f"Failed to load SAM2 real model: {e}. Falling back to mock.")
                 self.sam_available = False
                 self.sam_predictor = SAM2ImagePredictor(None)
        else:
            self.sam_predictor = SAM2ImagePredictor(None)

    @abstractmethod
    def run(self, image, config):
        pass
        
    def get_metrics(self):
        return self.benchmark.get_metrics()
        
    def reset_metrics(self):
        self.benchmark.reset()

class StrategyA(PipelineStrategy):
    """
    Balanced Efficiency: YOLO Small + SAM 2 Base (No TTA, No Refinement)
    """
    def run(self, image, config):
        self.reset_metrics()
        self.benchmark.reset()
        
        # 0. Preprocess / Set Image for SAM
        with self.benchmark.measure('SAM2 Encoding'):
            self.sam_predictor.set_image(image)
            
        # 1. Detect
        conf_threshold = config.get('conf', 0.25)
        iou_threshold = config.get('iou', 0.45)
        
        with self.benchmark.measure('YOLO Detection'):
            results = self.yolo_model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
        if len(boxes) == 0:
            return [], boxes
           
        # 2. Segment (Now looping for stability across SAM 2 versions)
        masks_final = []
        with self.benchmark.measure('SAM2 Decoding'):
            with torch.inference_mode():
                for box in boxes:
                    masks, scores, _ = self.sam_predictor.predict(
                        box=box,
                        multimask_output=False
                    )
                    # masks is (1, H, W) or (3, H, W) if multimask=True
                    masks_final.append(masks[0])
            
        return np.array(masks_final), boxes

class StrategyB(PipelineStrategy):
    """
    Ground Truth Precision: YOLO X (TTA) + SAM 2 Huge + Iterative Refinement
    """
    def run(self, image, config):
        self.reset_metrics()
        
        # 0. Preprocess
        with self.benchmark.measure('SAM2 Encoding'):
            self.sam_predictor.set_image(image)
            
        # 1. Detect with TTA
        conf = config.get('conf', 0.25)
        iou = config.get('iou', 0.45)
        use_tta = config.get('tta', True)
        
        with self.benchmark.measure('YOLO Detection (TTA)'):
            results = self.yolo_model(image, conf=conf, iou=iou, augment=use_tta, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
        if len(boxes) == 0:
            return [], boxes
            
        masks_final = []
        
        # 2. Refinement Loop per Object
        # Note: Batching refinement is harder, doing per-object loop for clarity/precision as requested
        max_iters = config.get('refine_iters', 1)
        with self.benchmark.measure('Refinement Loop'):
            with torch.inference_mode():
                for i, box in enumerate(boxes):
                    # Initial Pred
                    mask_preds, scores, _ = self.sam_predictor.predict(
                        box=box,
                        multimask_output=True # We want diverse masks to pick best
                    )
                    
                    # Pick best based on score
                    best_idx = np.argmax(scores)
                    current_mask = mask_preds[best_idx]
                    current_score = scores[best_idx]
                    
                    # Iterative Refinement
                    for _ in range(max_iters):
                        y_indices, x_indices = np.where(current_mask)
                        if len(y_indices) > 0:
                            centroid_y = np.mean(y_indices)
                            centroid_x = np.mean(x_indices)
                            
                            point_coords = np.array([[centroid_x, centroid_y]])
                            point_labels = np.array([1]) # 1 = positive
                            
                            # Re-predict with box + point
                            mask_preds_ref, scores_ref, _ = self.sam_predictor.predict(
                                box=box,
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=True
                            )
                            
                            # Update if score improved
                            best_idx_ref = np.argmax(scores_ref)
                            if scores_ref[best_idx_ref] > current_score:
                                 current_mask = mask_preds_ref[best_idx_ref]
                                 current_score = scores_ref[best_idx_ref]
                        else:
                            break # Empty mask, stop
                    
                    masks_final.append(current_mask)
                
        return np.array(masks_final), boxes
