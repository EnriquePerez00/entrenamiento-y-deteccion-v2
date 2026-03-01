#!/usr/bin/env python3
"""
🚀 Local Render Engine v1.0 (Mac Pro M4 Optimized)
==================================================
This script orchestrates the local Blender rendering process, mirroring the strategy of 
the cloud notebooks but optimized for Apple Silicon (METAL rendering).
"""

import os
import sys
import json
import time
import shutil
import zipfile
import subprocess
import concurrent.futures
import multiprocessing
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
RENDER_LOCAL_DIR = PROJECT_ROOT / "render_local"
IMAGES_DIR = RENDER_LOCAL_DIR / "images"
LABELS_DIR = RENDER_LOCAL_DIR / "labels"
LOGS_DIR = RENDER_LOCAL_DIR / "logs"
CONFIGS_DIR = RENDER_LOCAL_DIR / "configs"

# Paths
LDRAW_PATH = PROJECT_ROOT / "assets" / "ldraw"
ADDON_PATH = PROJECT_ROOT / "src" / "blender_scripts"
SCENE_SETUP_PY = PROJECT_ROOT / "src" / "blender_scripts" / "scene_setup.py"
# Default path for Blender on macOS
BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"

# If blender is not in Applications, try to find it in PATH
if not os.path.exists(BLENDER_PATH):
    try:
        BLENDER_PATH = subprocess.check_output(["which", "blender"]).decode().strip()
    except:
        pass

TIER_CONFIG = {
    'TIER_1': {'imgs': 30, 'res': 640, 'engine': 'EEVEE', 'epochs': 50},
    'TIER_2': {'imgs': 80, 'res': 1280, 'engine': 'EEVEE', 'epochs': 150},
    'TIER_3': {'imgs': 150, 'res': 2048, 'engine': 'CYCLES', 'epochs': 300}, # 2K para piezas complejas
}

def get_piece_tier(part_id):
    """Dynamic tiering logic (Simplified for local setup, using common IDs)."""
    # Minifig parts or complex IDs usually deserve TIER 3
    if part_id.startswith('sw') or len(part_id) > 5:
        return 'TIER_3'
    # Medium complexity
    if part_id in ['32054', '3795']:
        return 'TIER_3'
    return 'TIER_2'

def setup_structure(piece_id=None):
    """Initializes the local render workspace or piece-specific subfolders."""
    base = RENDER_LOCAL_DIR if not piece_id else RENDER_LOCAL_DIR / piece_id
    
    subdirs = {
        'images': base / "images",
        'labels': base / "labels",
        'logs': base / "logs",
        'configs': base / "configs"
    }
    
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return subdirs

def run_render_worker(worker_id, piece_id, chunks_for_worker):
    """Execution function for a single Blender process (now piece-specific)."""
    if not chunks_for_worker: return
    
    piece_dirs = setup_structure(piece_id)
    
    worker_cfg = {
        'worker_id': str(worker_id),
        'set_id': piece_id, 
        'pieces_config': chunks_for_worker, 
        'output_base': str(RENDER_LOCAL_DIR / piece_id),
        'assets_dir': str(PROJECT_ROOT / 'assets'),
        'ldraw_path': str(LDRAW_PATH),
        'addon_path': str(ADDON_PATH)
    }
    
    cfg_path = piece_dirs['configs'] / f'render_cfg_{worker_id}.json'
    with open(cfg_path, 'w') as f:
        json.dump(worker_cfg, f, indent=4)
    
    log_file = piece_dirs['logs'] / f'worker_{worker_id}.log'
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['UNIVERSAL_DETECTOR'] = '1'
    
    print(f"  ↳ [{piece_id} | Worker {worker_id}] Starting render (caffeinated)...")
    
    with open(log_file, 'w') as f_out:
        # Wrap blender in 'caffeinate' to keep the Mac awake
        subprocess.run([
            'caffeinate', '-i', BLENDER_PATH, '--background', '--python', str(SCENE_SETUP_PY),
            '--', str(cfg_path)
        ], stdout=f_out, stderr=subprocess.STDOUT, env=env)
    
    return worker_id

def main(target_parts, render_settings=None):
    start_time = time.time()
    
    # 1. Resolve Tiers and Minifig Components
    from src.logic.resolve_minifig import MinifigResolver
    resolver = MinifigResolver(ldraw_path=LDRAW_PATH)
    
    full_pieces_config = []
    minifig_groups = {} # To store chunks per minifig ID
    
    for p_id in target_parts:
        tier_key = p_id.get('part_id', p_id.get('ldraw_id', str(p_id))) if isinstance(p_id, dict) else str(p_id)
        tier = get_piece_tier(tier_key)
        cfg = TIER_CONFIG[tier].copy() # copy to avoid global mutation
        
        # Override with manual settings if provided (e.g. from UI)
        if render_settings:
            if render_settings.get('num_images') is not None:
                cfg['imgs'] = render_settings['num_images']
            if render_settings.get('engine'):
                cfg['engine'] = render_settings['engine']
            if render_settings.get('resolution_x'):
                cfg['res'] = render_settings['resolution_x']

        # Support color_id: target_parts can be a list of strings (part_id only)
        # or a list of dicts ({"part_id": ..., "color_id": ...})
        if isinstance(p_id, dict):
            raw_color = p_id.get('color_id', 15)
            p_id_str = p_id.get('part_id', p_id.get('ldraw_id', str(p_id)))
        else:
            raw_color = 15  # Default: White
            p_id_str = str(p_id)
        
        piece_entry = {
            'ldraw_id': p_id_str, 
            'color_id': int(raw_color), 
            'name': p_id_str
        }
        
        is_minifig = p_id_str.startswith('sw') or p_id_str.startswith('fig')
        if is_minifig:
            print(f"🔍 Resolving components for Minifig: {p_id_str}")
            components = resolver.get_minifig_parts(p_id_str)
            
            if not components:
                # 🛡️ Generic/Specific Fallbacks for local testing without API key
                print(f"⚠️ API failed for {p_id_str}. Using internal fallback...")
                if p_id_str == "sw0578": # Stormtrooper
                    components = [
                        {"rb_id": "973pb1672c01", "ldraw_id": "973ps9", "name": "Torso", "type": "Torso"},
                        {"rb_id": "970pb0536", "ldraw_id": "970", "name": "Legs", "type": "Legs"},
                        {"rb_id": "3626cpb1126", "ldraw_id": "3626c", "name": "Head", "type": "Head"},
                        {"rb_id": "11110", "ldraw_id": "30408", "name": "Helmet", "type": "Hat"}
                    ]
                else: # Generic Battle Droid or Humanoid fallback
                    components = [
                        {"rb_id": "30375", "ldraw_id": "30375", "name": "Torso", "type": "Torso"},
                        {"rb_id": "30376", "ldraw_id": "30376", "name": "Legs", "type": "Legs"},
                        {"rb_id": "30377", "ldraw_id": "30377", "name": "Head", "type": "Head"}
                    ]

            if components:
                print(f"✅ Found {len(components)} components for {p_id}.")
                minifig_groups[p_id] = [
                    {
                        'part': piece_entry,
                        'minifig_components': components,
                        'tier': 'TIER_3', # Minifigs always TIER_3
                        'imgs': cfg['imgs'],
                        'engine': 'CYCLES', # Minifigs always CYCLES
                        'res': cfg['res']
                    }
                ]
            else:
                print(f"❌ Could not resolve components for {p_id}. Skipping.")
        else:
            config_item = {
                'part': piece_entry,
                'tier': tier,
                'imgs': cfg['imgs'],
                'engine': cfg['engine'],
                'res': cfg['res']
            }
            full_pieces_config.append(config_item)

    # 2. Execution Strategy
    # We'll run standard pieces together, and Minifigs in their own folders
    total_imgs = sum(p['imgs'] for p in full_pieces_config) + sum(p[0]['imgs'] for p in minifig_groups.values())
    
    if total_imgs == 0:
        print("📭 No images to render. Check if pieces/minifigs have valid components.")
        return

    print(f"📊 Total Render Plan: {total_imgs} images", flush=True)

    # M4 Pro Optimization: Balanced workload to allow OS responsiveness.
    num_cores = multiprocessing.cpu_count()
    max_workers = max(1, num_cores - 2)
    print(f"🚀 M4 Balanced Parallelism: Using {max_workers} concurrent workers (Cores: {num_cores})", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Standard pieces (Parallelized by splitting total count if fewer pieces than workers)
        if len(full_pieces_config) == 1 and max_workers > 1:
            item = full_pieces_config[0]
            p_id = item['part']['ldraw_id']
            cnt = item['imgs']
            chunk = cnt // max_workers
            rem = cnt % max_workers
            for i in range(max_workers):
                c = chunk + (1 if i < rem else 0)
                if c == 0: continue
                w_item = item.copy()
                w_item['imgs'] = c
                futures.append(executor.submit(run_render_worker, str(i), p_id, [w_item]))
        else:
            for item in full_pieces_config:
                p_id = item['part']['ldraw_id']
                futures.append(executor.submit(run_render_worker, "0", p_id, [item]))

        # Minifigures (Parallelized similarly if few minifigs)
        if len(minifig_groups) == 1 and max_workers > 1:
            m_id = list(minifig_groups.keys())[0]
            m_cfg = minifig_groups[m_id]
            # m_cfg is a list of components, but usually we just want to split the 'imgs' count
            # Simplified: just use 1 worker for now if logic is complex, or split count
            futures.append(executor.submit(run_render_worker, "0", m_id, m_cfg))
        else:
            for m_id, m_cfg in minifig_groups.items():
                futures.append(executor.submit(run_render_worker, "0", m_id, m_cfg))

        # Monitor
        while any(f.running() for f in futures):
            # Only count images for parts in the current request to avoid global cache pollution in progress
            curr_imgs = 0
            for p_id_entry in target_parts:
                p_id_str = p_id_entry.get('part_id', p_id_entry.get('ldraw_id', str(p_id_entry))) if isinstance(p_id_entry, dict) else str(p_id_entry)
                p_img_dir = RENDER_LOCAL_DIR / p_id_str / "images"
                if p_img_dir.exists():
                    curr_imgs += len(list(p_img_dir.glob("*.jpg")))
            
            p = (curr_imgs / total_imgs) * 100 if total_imgs > 0 else 0
            # Cap at 100% and show correct current count
            print(f"📈 Progress: {min(total_imgs, curr_imgs)}/{total_imgs} images ({min(100.0, p):.1f}%)", flush=True)
            time.sleep(2)

    # 4. Post-Processing: Similarity Filter + data.yaml
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ All renders completed in {duration/60:.1f} min.")
    
    # 4a. Similarity Filter (remove near-duplicate images, cosine > 0.98)
    print("🔍 Running similarity filter on rendered images...")
    total_deleted = 0
    try:
        from src.logic.feature_extractor import FeatureExtractor
        from PIL import Image
        import numpy as np
        
        extractor = FeatureExtractor()
        THRESHOLD = 0.98
        
        for piece_dir in RENDER_LOCAL_DIR.iterdir():
            if not piece_dir.is_dir() or piece_dir.name.startswith('.'): continue
            images_dir = piece_dir / "images"
            labels_dir = piece_dir / "labels"
            if not images_dir.exists(): continue
            
            img_paths = sorted(list(images_dir.glob("*.jpg")))
            if len(img_paths) < 2: continue
            
            last_embedding = None
            deleted_in_piece = 0
            
            for img_path in img_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    embedding = extractor.get_embedding(img)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    if last_embedding is not None:
                        similarity = float(np.dot(embedding, last_embedding))
                        if similarity > THRESHOLD:
                            img_path.unlink()
                            label_path = labels_dir / img_path.name.replace('.jpg', '.txt')
                            if label_path.exists():
                                label_path.unlink()
                            deleted_in_piece += 1
                            continue
                    
                    last_embedding = embedding
                except Exception as e:
                    print(f"  ⚠️ Filter error on {img_path.name}: {e}")
            
            if deleted_in_piece > 0:
                print(f"  🗑️ {piece_dir.name}: removed {deleted_in_piece} near-duplicates")
                total_deleted += deleted_in_piece
    except ImportError:
        print("  ⚠️ FeatureExtractor not available. Skipping similarity filter.")
    
    print(f"✅ Filter complete. Removed {total_deleted} duplicates total.")
    
    # 4b. Generate data.yaml for each piece/minifig subfolder
    print("📝 Generating data.yaml files...")
    piece_manifest = []
    
    for piece_dir in RENDER_LOCAL_DIR.iterdir():
        if not piece_dir.is_dir() or piece_dir.name.startswith('.'): continue
        images_dir = piece_dir / "images"
        labels_dir = piece_dir / "labels"
        if not images_dir.exists(): continue
        
        img_count = len(list(images_dir.glob("*.jpg")))
        lbl_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
        
        if img_count == 0: continue
        
        # Write data.yaml (YOLO format, Universal Detector = 1 class)
        data_yaml_path = piece_dir / "data.yaml"
        with open(data_yaml_path, 'w') as f:
            f.write(f"path: .\n")
            f.write(f"train: images\n")
            f.write(f"val: images\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['lego']\n")
        
        piece_manifest.append({
            "piece_id": piece_dir.name,
            "images": img_count,
            "labels": lbl_count,
            "data_yaml": str(data_yaml_path.relative_to(RENDER_LOCAL_DIR)),
        })
        print(f"  ✅ {piece_dir.name}: {img_count} imgs, {lbl_count} labels → data.yaml")
    
    # 4c. Global manifest
    manifest_path = RENDER_LOCAL_DIR / "dataset_manifest.json"
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "duration_minutes": round(duration / 60, 1),
        "total_pieces": len(piece_manifest),
        "total_images": sum(p['images'] for p in piece_manifest),
        "duplicates_removed": total_deleted,
        "pieces": piece_manifest,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Manifest: {manifest_path}")
    
    # 5. ZIP for Lightning AI (includes dataset + source code needed for training)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"lightning_dataset_{ts}.zip"
    zip_path = PROJECT_ROOT / zip_name
    
    print(f"📦 Creating Lightning AI package: {zip_name}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # A. Dataset (render_local/ contents)
        for file in RENDER_LOCAL_DIR.rglob("*"):
            if file.is_file() and not file.name.startswith('.'):
                arcname = Path("render_local") / file.relative_to(RENDER_LOCAL_DIR)
                zipf.write(file, arcname=arcname)
        
        # B. Source code needed for training
        src_dir = PROJECT_ROOT / "src"
        for py_file in src_dir.rglob("*.py"):
            if '__pycache__' in str(py_file): continue
            arcname = Path("src") / py_file.relative_to(src_dir)
            zipf.write(py_file, arcname=arcname)
        
        # C. Config
        config_path = PROJECT_ROOT / "config_train.json"
        if config_path.exists():
            zipf.write(config_path, arcname="config_train.json")
        
        # D. Credentials for Drive sync (if available)
        for cred_file in ["credentials.json", "token_1973.pickle"]:
            cred_path = PROJECT_ROOT / cred_file
            if cred_path.exists():
                zipf.write(cred_path, arcname=cred_file)
        
        # E. Existing models (for incremental training)
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            for model_file in models_dir.rglob("*"):
                if model_file.is_file() and not model_file.name.startswith('.'):
                    arcname = Path("models") / model_file.relative_to(models_dir)
                    zipf.write(model_file, arcname=arcname)
                    
        # F. Notebooks (NEW: include generated Lightning AI notebooks)
        notebooks_dir = PROJECT_ROOT / "notebooks"
        if notebooks_dir.exists():
            # Search for LightningAI notebooks
            for nb_file in notebooks_dir.rglob("*.ipynb"):
                if 'LightningAI_' in nb_file.name:
                    # We only include the most recent one to keep ZIP small, or all? 
                    # Usually, including the subfolder structure is safer
                    arcname = Path("notebooks") / nb_file.relative_to(notebooks_dir)
                    zipf.write(nb_file, arcname=arcname)
                    print(f"  📎 Including notebook: {nb_file.name}")
    
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"🌟 Lightning package ready: {zip_name} ({zip_size_mb:.1f} MB)")
    print(f"📊 Contents: {manifest['total_images']} images across {manifest['total_pieces']} pieces")
    print(f"⏱️ Total pipeline time: {duration/60:.1f} min")

if __name__ == "__main__":
    # Get pieces from config_train.json if exists, else defaults
    parts = []
    render_settings = {}
    config_path = PROJECT_ROOT / "config_train.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
            parts = data.get('target_parts', [])
            render_settings = data.get('render_settings', {})
            
    if not parts:
        parts = ["3022", "32054", "3795", "4073", "sw0578"]
        
    main(parts, render_settings=render_settings)
