import pickle
import os

meta_path = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/models/piezas_vectores/lego_meta.pkl"
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta_dict = pickle.load(f)
    
    metadata = meta_dict.get('metadata', [])
    print(f"Total entries in metadata: {len(metadata)}")
    
    ids = set()
    for m in metadata:
        ids.add(m.get('ldraw_id'))
    
    print(f"Total unique IDs: {len(ids)}")
    if '3200' in ids: print("✅ Found 3200")
    if '32000' in ids: print("✅ Found 32000")
    if not '3200' in ids and not '32000' in ids:
        print(f"Example IDs: {list(ids)[:20]}")
