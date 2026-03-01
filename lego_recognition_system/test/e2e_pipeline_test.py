import os
import json
import shutil
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("."))

from src.logic.generator import SyntheticGenerator
from src.logic.trainer import ModelTrainer
from src.logic.build_reference_index import build_index
from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Config
SET_ID = "75078_E2E"
TEST_DIR = Path("test/e2e_workspace")
NUM_IMAGES = 20 # Small number for quick verification
EPOCHS = 2 # Minimum for verification

# Test Parts (from set 75078-1)
TEST_PARTS = [
    {"ldraw_id": "3004", "name": "Brick 1 x 2", "color_id": 71},   # Light Bluish Gray
    {"ldraw_id": "3710", "name": "Plate 1 x 4", "color_id": 71}    # Light Bluish Gray
]

def run_e2e():
    print(f"🚀 Starting E2E Pipeline Verification for Set {SET_ID}")
    
    # Load config
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    blender_path = config.get("blender_executable_path")
    ldraw_path = config.get("ldraw_library_path")
    
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- PHASE 1: SYNTHETIC GENERATION ---
    print("\n--- PHASE 1: Synthetic Generation ---")
    gen = SyntheticGenerator(
        blender_path=blender_path,
        ldraw_path=ldraw_path,
        output_dir=str(TEST_DIR / "datasets")
    )
    
    # Force Universal Mode
    os.environ["UNIVERSAL_DETECTOR"] = "1"
    
    print(f"Generating {NUM_IMAGES} images for parts: {[p['ldraw_id'] for p in TEST_PARTS]}")
    for progress, status in gen.generate_dataset(SET_ID, TEST_PARTS, num_images=NUM_IMAGES):
        print(f"   [{progress}%] {status}")
        
    dataset_path = TEST_DIR / "datasets" / SET_ID
    if not (dataset_path / "images").exists() or not (dataset_path / "image_meta.jsonl").exists():
        print("❌ Error: Phase 1 FAILED - Dataset not generated correctly.")
        return

    # --- PHASE 2: YOLO TRAINING ---
    print("\n--- PHASE 2: YOLO Training ---")
    trainer = ModelTrainer(model_dir=str(TEST_DIR / "models"))
    
    # We use very few epochs and a small model for speed
    trainer.train_model(SET_ID, str(dataset_path), epochs=EPOCHS)
    
    model_path = trainer.get_model_path(SET_ID)
    if not model_path or not os.path.exists(model_path):
        print("❌ Error: Phase 2 FAILED - Model not trained/saved.")
        return
    print(f"✅ Phase 2 SUCCESS: Model saved at {model_path}")

    # --- PHASE 3: VECTOR INDEXING ---
    print("\n--- PHASE 3: Vector Indexing ---")
    index_dir = TEST_DIR / "indices"
    build_index(str(dataset_path), str(index_dir), unified=True)
    
    master_index_path = index_dir / "lego.index"
    if not master_index_path.exists():
        print("❌ Error: Phase 3 FAILED - FAISS index not created.")
        return
    print(f"✅ Phase 3 SUCCESS: Index created at {master_index_path}")

    # --- PHASE 4: RECOGNITION SIMULATION ---
    print("\n--- PHASE 4: Recognition Simulation ---")
    
    # 1. Load Components
    print("   Loading YOLO model...")
    detector = YOLO(model_path)
    
    print("   Loading Vector Index...")
    v_index = VectorIndex(index_path=str(master_index_path))
    
    print("   Initializing Feature Extractor...")
    extractor = FeatureExtractor()
    
    # 2. Pick a random test image (from the generated ones)
    test_images = list((dataset_path / "images").glob("*.png"))
    if not test_images:
        print("❌ Error: No images to test.")
        return
    
    sample_img_path = test_images[0]
    print(f"   Testing on image: {sample_img_path.name}")
    
    # Load metadata for ground truth
    ground_truth = {}
    with open(dataset_path / "image_meta.jsonl", 'r') as f:
        for line in f:
            entry = json.loads(line)
            ground_truth[entry['img']] = entry['ids']
    
    real_ids = ground_truth.get(sample_img_path.name, [])
    print(f"   Ground Truth (LDraw IDs): {real_ids}")
    
    # 3. Detection
    results = detector(str(sample_img_path), device='mps')
    
    img_cv = cv2.imread(str(sample_img_path))
    h, w, _ = img_cv.shape
    
    print(f"   Detected {len(results[0].boxes)} objects.")
    
    match_count = 0
    for i, box in enumerate(results[0].boxes):
        # Localize
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Crop & Buffer
        crop = img_cv[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0: continue
        
        # Letterboxing (as done in build_index)
        ch, cw, _ = crop.shape
        max_dim = max(ch, cw)
        canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        canvas[(max_dim-ch)//2:(max_dim-ch)//2+ch, (max_dim-cw)//2:(max_dim-cw)//2+cw] = crop
        
        # Feature Extraction
        crop_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        embedding = extractor.get_embedding(crop_pil)
        
        # Search
        search_results = v_index.search(embedding, k=1)
        
        if search_results:
            pred_id = search_results[0]['metadata']['ldraw_id']
            similarity = search_results[0]['similarity']
            
            # Simple check (order might correspond if 1 object, but let's just see if it's in GT)
            is_correct = pred_id in real_ids
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            if is_correct: match_count += 1
            
            print(f"   [Obj {i}] Predicted: {pred_id} (Sim: {similarity:.4f}) -> {status}")

    print(f"\n--- FINAL RESULTS ---")
    print(f"Success Rate: {match_count}/{len(results[0].boxes)} detections matched ground truth.")
    print(f"E2E Pipeline Verification: {'PASSED' if match_count > 0 else 'FAILED'}")

if __name__ == "__main__":
    run_e2e()
