
import torch
import numpy as np
from pipelines import StrategyA, StrategyB
import time

def reproduce():
    print("Testing Strategy A and B with a dummy image...")
    # Create a 1024x1024 dummy image
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    try:
        print("Initializing Strategy A...")
        sa = StrategyA(model_name='yolo11n.pt', sam_checkpoint='sam2_hiera_base.pt')
        print("Running Strategy A...")
        masks_a, boxes_a = sa.run(image, {'conf': 0.25, 'iou': 0.45})
        print(f"Strategy A done. Boxes: {len(boxes_a)}")
    except Exception as e:
        print(f"Strategy A failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\nInitializing Strategy B...")
        sb = StrategyB(model_name='yolo11x.pt', sam_checkpoint='sam2_hiera_large.pt')
        print("Running Strategy B...")
        # Add a dummy box to force SAM execution in Strategy B
        # Since YOLO won't find anything on a black image
        sa.yolo_model.predict = lambda *args, **kwargs: type('obj', (object,), {'boxes': type('obj', (object,), {'xyxy': torch.tensor([[100, 100, 200, 200]], device=sa.device)})()})()
        
        # Monkey patch YOLO in Strategy B to return a dummy box
        sb.yolo_model = sa.yolo_model 
        
        masks_b, boxes_b = sb.run(image, {'conf': 0.25, 'iou': 0.45, 'tta': False, 'refine_iters': 1})
        print(f"Strategy B done. Boxes: {len(boxes_b)}")
    except Exception as e:
        print(f"Strategy B failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
