
import os
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(os.getcwd())
sys.path.append(str(PROJECT_ROOT))

from src.logic.resolve_minifig import MinifigResolver
import run_local_render

def dry_run_minifig(minifig_id, num_test_images=5):
    print(f"🚀 Starting End-to-End Trial for Minifig: {minifig_id}")
    
    ldraw_path = PROJECT_ROOT / "assets" / "ldraw"
    resolver = MinifigResolver(ldraw_path=ldraw_path)
    
    # 1. Resolve components
    print(f"📥 Fetching components for {minifig_id}...")
    components = resolver.get_minifig_parts(minifig_id)
    if not components:
        print("⚠️ Rebrickable API failed, using fallback component list for sw0578 test...")
        components = [
            {"rb_id": "973pb1672c01", "ldraw_id": "973", "name": "Torso", "type": "Torso"},
            {"rb_id": "970pb0536", "ldraw_id": "970", "name": "Legs", "type": "Legs"},
            {"rb_id": "3626cpb1126", "ldraw_id": "3626c", "name": "Head", "type": "Head"}
        ]

    print(f"✅ Found {len(components)} components.")
    for c in components:
        print(f"   - {c['ldraw_id']} ({c['name']})")

    # 2. Download Assets
    print("📦 Downloading missing LDraw assets...")
    from download_assets import download_part, setup_dirs
    setup_dirs()
    for c in components:
        download_part(c['ldraw_id'])

    # 3. Create Temporary Config for 5 images
    print(f"🎬 Preparing local render (Target: {num_test_images} images)...")
    
    # Overriding TIER_CONFIG for this test to force 5 images
    test_tier = {
        'imgs': num_test_images,
        'res': 1024,
        'engine': 'CYCLES'
    }
    
    # We call run_local_render logic manually with the test override
    run_local_render.TIER_CONFIG['TIER_TEST'] = test_tier
    
    # Standardize run_local_render calls
    run_local_render.main([minifig_id])

if __name__ == "__main__":
    dry_run_minifig("sw0578", num_test_images=5)
