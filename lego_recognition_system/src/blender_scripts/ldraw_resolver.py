import os
import json

class LDrawResolver:
    def __init__(self, ldraw_path_base=None):
        self.ldraw_path_base = ldraw_path_base
        self.studio_path = "/Applications/Studio 2.0/ldraw"
        self.missing_parts = []

    def find_part(self, part_id):
        # Ensure part_id doesn't have .dat for internal matching if needed, 
        # but the _check_path adds it. Let's make it robust.
        clean_id = str(part_id).replace(".dat", "").replace(".DAT", "")
        
        # Priority 1: Configured LDraw Path
        if self.ldraw_path_base:
            path = self._check_path(self.ldraw_path_base, clean_id)
            if path: return path

        # Priority 2: Studio 2.0 Path
        path = self._check_path(self.studio_path, part_id)
        if path:
            print(f"Fallback used for {part_id}: Found in Studio 2.0 library.")
            return path

        # Not found
        self.missing_parts.append(part_id)
        return None

    def _check_path(self, base_path, part_id):
        # Check standard locations: parts/ and p/
        # Also check root for flat libraries
        candidates = [
            os.path.join(base_path, "parts", f"{part_id}.dat"),
            os.path.join(base_path, "p", f"{part_id}.dat"),
            os.path.join(base_path, f"{part_id}.dat")
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def save_report(self, output_dir):
        if self.missing_parts:
            report_path = os.path.join(output_dir, "missing_parts.json")
            with open(report_path, 'w') as f:
                json.dump({"missing": self.missing_parts}, f, indent=2)
            print(f"Missing parts report saved to {report_path}")
