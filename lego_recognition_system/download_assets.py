
import os
import requests
from pathlib import Path

import json

# Config
# Use environment variable if available (for cloud environments)
LDRAW_DIR = Path(os.environ.get("LDRAW_PATH", "./assets/ldraw")) 
PARTS_DIR = LDRAW_DIR / "parts"
P_DIR = LDRAW_DIR / "p"

# Target Parts (Read from config_train.json)
TARGETS = []
config_path = Path("config_train.json")
if config_path.exists():
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            TARGETS = config.get("target_parts", [])
            print(f"Loaded target parts from config: {TARGETS}")
    except Exception as e:
        print(f"Error reading config: {e}")
        TARGETS = ["15391", "3004", "51739"] # Fallback
else:
    TARGETS = ["15391", "3004", "51739"] # Fallback

def setup_dirs():
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    P_DIR.mkdir(parents=True, exist_ok=True)
    
def download_part(part_id):
    urls = [
        f"https://library.ldraw.org/library/official/parts/{part_id}.dat",
        f"https://library.ldraw.org/library/official/p/{part_id}.dat",
        f"https://library.ldraw.org/library/unofficial/parts/{part_id}.dat"
    ]
    
    save_path = PARTS_DIR / f"{part_id}.dat"
    
    for url in urls:
        try:
            print(f"Trying {url}...")
            r = requests.get(url)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
                print(f"✅ Downloaded {part_id}")
                return True
        except Exception as e:
            print(f"Error: {e}")
            
    print(f"❌ Failed to download {part_id}")
    return False

if __name__ == "__main__":
    setup_dirs()
    print("Downloading required LDraw parts...")
    for part in TARGETS:
        download_part(part)

