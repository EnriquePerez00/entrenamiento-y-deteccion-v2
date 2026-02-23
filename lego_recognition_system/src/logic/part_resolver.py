"""
part_resolver.py
Resolves a set ID or single piece ID into a standardized list of parts.
Loads from local inventory JSON first, falls back to Rebrickable API.
"""
import os
import json
import random
import logging

logger = logging.getLogger("LegoVision")

REBRICKABLE_API_BASE = "https://rebrickable.com/api/v3/lego"
INVENTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "inventory")


def _load_local_inventory(set_id: str) -> list | None:
    """Try to load inventory from local JSON file."""
    path = os.path.join(INVENTORY_DIR, f"{set_id}.json")
    if os.path.exists(path):
        logger.info(f"📂 Loaded local inventory for set {set_id}")
        with open(path, "r") as f:
            return json.load(f)
    return None


def _fetch_rebrickable_inventory(set_id: str) -> list | None:
    """Fetch set inventory from Rebrickable API (no auth needed for basic queries)."""
    try:
        import urllib.request
        url = f"{REBRICKABLE_API_BASE}/sets/{set_id}/parts/?page_size=500"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        parts = []
        for item in data.get("results", []):
            p = item.get("part", {})
            parts.append({
                "part_num": p.get("part_num", ""),
                "name": p.get("name", ""),
                "color_id": item.get("color", {}).get("id", 0),
                "color_name": item.get("color", {}).get("name", "Unknown"),
                "quantity": item.get("quantity", 1),
                "ldraw_id": p.get("part_num", ""),
            })
        if parts:
            logger.info(f"🌐 Fetched {len(parts)} parts from Rebrickable for set {set_id}")
            # Cache locally
            os.makedirs(INVENTORY_DIR, exist_ok=True)
            with open(os.path.join(INVENTORY_DIR, f"{set_id}.json"), "w") as f:
                json.dump(parts, f, indent=4)
        return parts
    except Exception as e:
        logger.warning(f"⚠️ Rebrickable API failed: {e}")
        return None


def resolve_set(set_id: str, max_parts: int = None) -> list:
    """
    Given a set ID, return a list of unique parts (by ldraw_id).
    If max_parts is specified, picks a random sample.
    Returns list of dicts: {ldraw_id, name, part_num}
    """
    raw = _load_local_inventory(set_id) or _fetch_rebrickable_inventory(set_id)
    if not raw:
        raise ValueError(f"❌ Could not resolve set {set_id}. Check the set ID or add a JSON file at data/inventory/{set_id}.json")

    # Deduplicate by ldraw_id
    seen = {}
    for part in raw:
        lid = part.get("ldraw_id") or part.get("part_num")
        if lid and lid not in seen:
            seen[lid] = {"ldraw_id": lid, "name": part.get("name", lid), "part_num": part.get("part_num", lid)}

    unique_parts = list(seen.values())
    logger.info(f"🧩 Set {set_id} has {len(unique_parts)} unique part types")

    if max_parts and max_parts < len(unique_parts):
        selected = random.sample(unique_parts, max_parts)
        logger.info(f"🎲 Randomly selected {max_parts} parts from {len(unique_parts)} total")
        return selected
    return unique_parts


def resolve_piece(ldraw_id: str) -> list:
    """Returns a single-part list for a specific piece ID."""
    return [{"ldraw_id": ldraw_id, "name": ldraw_id, "part_num": ldraw_id}]
