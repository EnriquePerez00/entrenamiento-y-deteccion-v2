import os
import requests
import json
import pandas as pd
from pathlib import Path

class InventoryManager:
    def __init__(self, api_key, cache_dir="./data/inventory", ldraw_dir="./assets/ldraw"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.ldraw_dir = Path(ldraw_dir)
        self.base_url = "https://rebrickable.com/api/v3/lego"
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ldraw_dir.mkdir(parents=True, exist_ok=True)

    def inventory_exists(self, set_num):
        """Check if inventory exists in local cache."""
        return (self.cache_dir / f"{set_num}.json").exists()

    def get_set_inventory(self, set_num):
        """Fetch inventory from Rebrickable or local cache."""
        if "-" not in set_num:
             set_num = f"{set_num}-1"
             
        cache_file = self.cache_dir / f"{set_num}.json"
        
        if cache_file.exists():
            print(f"Loading inventory from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                return pd.DataFrame(json.load(f))
        
        print(f"Fetching from Rebrickable: {set_num}")
        url = f"{self.base_url}/sets/{set_num}/parts/"
        params = {'key': self.api_key, 'page_size': 1000}
        
        all_parts = []
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            all_parts.extend(data.get('results', []))
            
            while data.get('next'):
                response = requests.get(data['next'])
                response.raise_for_status()
                data = response.json()
                all_parts.extend(data.get('results', []))
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching inventory: {e}")
            return pd.DataFrame()

        # Simplify Data
        processed_data = []
        for part in all_parts:
            # Map Rebrickable Part ID to LDraw ID directly if available
            # Often 'part.part_num' is the LDraw ID, but sometimes there are variants.
            # We will use 'part_num' as primary key.
            processed_data.append({
                'part_num': part['part']['part_num'],
                'name': part['part']['name'],
                'color_id': part['color']['id'],
                'color_name': part['color']['name'],
                'quantity': part['quantity'],
                'ldraw_id': part['part'].get('external_ids', {}).get('LDraw', [part['part']['part_num']])[0] 
                # Fallback to part_num if external_ids not present (Rebrickable usually has this)
            })
            
        df = pd.DataFrame(processed_data)
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(processed_data, f, indent=4)
            
        return df

    def check_ldraw_assets(self, inventory_df):
        """Verify if LDraw .dat files exist for all parts."""
        missing_parts = []
        
        if inventory_df.empty:
            return missing_parts

        for _, row in inventory_df.iterrows():
            part_id = row['ldraw_id']
            
            # Check standard LDraw locations
            # Usually in 'parts', sometimes primitives in 'p'
            path_parts = self.ldraw_dir / "parts" / f"{part_id}.dat"
            path_p = self.ldraw_dir / "p" / f"{part_id}.dat"
            path_root = self.ldraw_dir / f"{part_id}.dat" # Fallback/Custom
            
            if not path_parts.exists() and not path_p.exists() and not path_root.exists():
                missing_parts.append(part_id)
        
        return list(set(missing_parts)) # Unique missing IDs

    def download_ldraw_part(self, part_id):
        """Download missing .dat file from LDraw library."""
        # Generic URL structure for LDraw library
        # Try both 'parts' and 'p' (primitives) folders if needed, but start with parts
        base_urls = [
            f"https://library.ldraw.org/library/official/parts/{part_id}.dat",
            f"https://library.ldraw.org/library/official/p/{part_id}.dat",
            f"https://library.ldraw.org/library/unofficial/parts/{part_id}.dat" 
        ]
        
        target_path = self.ldraw_dir / f"{part_id}.dat"
        
        for url in base_urls:
            try:
                print(f"Attempting download: {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(target_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {part_id}.dat")
                    return True
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                
        print(f"Could not find LDraw file for part: {part_id}")
        return False

    def get_all_colors(self):
        """Fetch and cache all LEGO colors from Rebrickable."""
        cache_file = self.cache_dir / "colors.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return {int(k): v for k, v in json.load(f).items()}
        
        print("Fetching colors from Rebrickable...")
        url = f"{self.base_url}/colors/"
        params = {'key': self.api_key, 'page_size': 1000}
        
        colors_map = {}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for color in data.get('results', []):
                # map ID -> RGB
                colors_map[color['id']] = color['rgb']
            
            with open(cache_file, 'w') as f:
                json.dump(colors_map, f)
                
            return colors_map
            
        except Exception as e:
            print(f"Error fetching colors: {e}")
            return {}

def get_set_inventory(set_num):
    """Helper function for standalone usage."""
    # Try to load from environment first
    api_key = os.environ.get("REBRICKABLE_API_KEY")
    
    # Fallback to config.json
    if not api_key:
        # Check relative to this script's directory first (project root)
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir.parent.parent / "config.json"
        
        if not config_path.exists():
            # Fallback to current working directory
            config_path = Path("config.json")
            
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get("rebrickable_api_key")
    
    # Final fallback to dummy
    if not api_key:
        api_key = "a52f6e7e9cb8c225d1339dcfda8b6ae7"

    im = InventoryManager(api_key=api_key)
    df = im.get_set_inventory(set_num)
    if not df.empty:
         return df.to_dict('records')
    return []

if __name__ == "__main__":
    # Test with a dummy key or ensure config is loaded
    # df = get_set_inventory("42115-1")
    # print(df)
    pass
