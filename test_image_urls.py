import urllib.request
import json

part_id = "3001" # 2x4 brick

urls_to_test = [
    f"https://cdn.rebrickable.com/media/parts/ldraw/14/{part_id}.png",
    f"https://cdn.rebrickable.com/media/parts/elements/{part_id}.jpg",
    f"https://img.bricklink.com/ItemImage/PN/0/{part_id}.png",
    f"https://img.bricklink.com/ItemImage/PN/11/{part_id}.png"
]

for url in urls_to_test:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        print(f"✅ SUCCESS: {url} (Status: {response.status})")
    except Exception as e:
        print(f"❌ FAILED: {url} - {e}")

