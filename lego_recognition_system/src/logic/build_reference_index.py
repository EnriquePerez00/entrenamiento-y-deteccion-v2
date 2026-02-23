import os
import json
import cv2
from PIL import Image
from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex
from pathlib import Path

def build_index(dataset_dir, output_folder):
    print(f"🛠️ Building Per-Piece Reference Indices in {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    
    extractor = FeatureExtractor()
    
    # We will group embeddings by ldraw_id to save separate files
    piece_data = {} # ldraw_id -> {'embeddings': [], 'metadata': []}
    
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    names_map = []
    if os.path.exists(data_yaml_path):
        import yaml
        with open(data_yaml_path, 'r') as f:
            y_data = yaml.safe_load(f)
            names_map = y_data.get('names', [])

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    processed_count = 0
    for label_file in label_files:
        img_file = label_file.replace('.txt', '.png')
        img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            
            # OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
            if len(parts) == 9:
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]
                
                # To extract the piece, we find the AABB of the OBB
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                x1 = int(min_x * w)
                y1 = int(min_y * h)
                x2 = int(max_x * w)
                y2 = int(max_y * h)
            else:
                # Fallback for old AABB just in case
                x_c, y_c, bw, bh = map(float, parts[1:5])
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
            
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0: continue
            
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            embedding = extractor.get_embedding(crop_pil)
            
            ldraw_id = names_map[class_id] if class_id < len(names_map) else "unknown"
            
            if ldraw_id not in piece_data:
                piece_data[ldraw_id] = {'embeddings': [], 'metadata': []}
            
            piece_data[ldraw_id]['embeddings'].append(embedding)
            piece_data[ldraw_id]['metadata'].append({'ldraw_id': ldraw_id})
            processed_count += 1
            
        if processed_count % 100 == 0:
            print(f"   Processed {processed_count} instances...")

    # Save individuals
    for ldraw_id, data in piece_data.items():
        out_path = os.path.join(output_folder, f"{ldraw_id}.pkl")
        temp_index = VectorIndex()
        temp_index.embeddings = data['embeddings']
        temp_index.metadata = data['metadata']
        temp_index.save(out_path)
        print(f"   ✅ Saved index: {ldraw_id}.pkl ({len(data['embeddings'])} views)")

    print(f"🎉 Finished splitting batch into {len(piece_data)} individual piece files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    build_index(args.dataset, args.output)
