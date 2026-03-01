import sys
import os
import json
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

from src.logic.generator import SyntheticGenerator

def test_fix():
    blender_path = "/Applications/Blender.app/Contents/MacOS/Blender" # Adjust if necessary
    if not os.path.exists(blender_path):
        # Try to find it in path
        import shutil
        blender_path = shutil.which("blender")
        if not blender_path:
            print("Blender not found. Skipping execution test, will only verify logic if possible.")
            return

    ldraw_path = "/Applications/Studio 2.0/ldraw" 
    assets_dir = "./assets"
    output_dir = "./data/test_datasets"
    
    gen = SyntheticGenerator(blender_path, ldraw_path, assets_dir, output_dir)
    
    set_id = "test_set"
    part_list = [
        {"ldraw_id": "3001", "name": "Brick 2 x 4"},
        {"ldraw_id": "3002", "name": "Brick 2 x 3"}
    ]
    
    # Force Universal Detector mode for testing
    os.environ["UNIVERSAL_DETECTOR"] = "1"
    
    print("Testing SyntheticGenerator.generate_dataset...")
    for progress, msg in gen.generate_dataset(set_id, part_list, num_images=4):
        if msg: print(f"  {msg}")
        else: print(f"  Progress: {progress}%")
        
    # Verification
    final_dir = Path(output_dir) / set_id
    images_dir = final_dir / "images"
    labels_dir = final_dir / "labels"
    meta_path = final_dir / "image_meta.jsonl"
    yaml_path = final_dir / "data.yaml"
    
    print("\nVerifying outputs:")
    print(f"Images count: {len(list(images_dir.glob('*.png')))} (Expected 4)")
    print(f"Labels count: {len(list(labels_dir.glob('*.txt')))} (Expected 4)")
    print(f"Metadata file exists: {meta_path.exists()}")
    print(f"Data.yaml exists: {yaml_path.exists()}")
    
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            lines = f.readlines()
            print(f"Metadata entries count: {len(lines)} (Expected 4 if no empty backgrounds)")
            if lines:
                print(f"First meta entry: {lines[0].strip()}")
                
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            print(f"Data.yaml content:\n{f.read()}")

if __name__ == "__main__":
    test_fix()
