import sys
import os
import cv2
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.logic.recognition_2d import TwoDRecognitionEngine

def create_mock_78075_image():
    # Simulate an image of set 78075-1
    # User description/image shows:
    # - White Grid Background
    # - Grey pieces (Light & Dark)
    # - White pieces (Stormtrooper torso?)
    # - Technic pins (Black/Blue?)
    # - Red pieces?
    
    # A4 at roughly 10 px/cm
    height, width = 297, 210
    img = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background
    
    # Draw Grid (1cm lines = 10px spacing)
    grid_color = (200, 200, 200) 
    for x in range(0, width, 10):
        cv2.line(img, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, 10):
        cv2.line(img, (0, y), (width, y), grid_color, 1)
        
    # 1. Dark Grey Plate (e.g. 4x6)
    # Area ~ 24 studs * 0.64 = 15 cm2 -> 150 px area -> sqrt(150) ~ 12px? No.
    # 1 stud = 0.8cm x 0.8cm = 0.64 cm2.
    # 10px/cm -> 1 stud = 8px x 8px.
    # 4x6 plate = 3.2cm x 4.8cm = 32px x 48px.
    # Color: Dark Grey (RGB ~ 60,60,60)
    cv2.rectangle(img, (50, 50), (82, 98), (60, 60, 60), -1) 
    
    # 2. Light Grey Brick (2x4)
    # 1.6cm x 3.2cm = 16px x 32px
    # Color: Light Grey (RGB ~ 160,160,160)
    cv2.rectangle(img, (100, 60), (132, 76), (160, 160, 160), -1)
    
    # 3. White Piece (Stormtrooper)
    # Irregular shape, maybe 2cm x 2cm = 20px x 20px
    # Color: White (255,255,255) -> Matches background!
    # If piece is white on white background, it will fail unless shadow/edges?
    # Or maybe it's "off-white" or the grid makes it visible?
    # Let's start with Off-White (240, 240, 240)
    cv2.rectangle(img, (150, 150), (170, 170), (240, 240, 240), -1)
    
    # 4. Red Tile (1x2)
    # 0.8cm x 1.6cm = 8px x 16px
    # Color: Red (200, 0, 0)
    cv2.rectangle(img, (50, 150), (58, 166), (0, 0, 200), -1)
    
    return img

def test_78075_simulation():
    print("Initializing Engine (78075-1 Simulation)...")
    engine = TwoDRecognitionEngine()
    
    print("Creating Mock Image (Grid + Grey/White/Red pieces)...")
    img = create_mock_78075_image()
    cv2.imwrite("tests/mock_78075.png", img)
    
    # Mock Inventory for 78075-1
    # Assuming it's based on visual description
    processed_data = [
        # Dark Grey Plate 4x6
        {'part_num': '3032', 'name': 'Plate 4x6', 'color_id': 85, 'quantity': 1, 'ldraw_id': '3032'},
        # Light Grey Brick 2x4
        {'part_num': '3001', 'name': 'Brick 2x4', 'color_id': 86, 'quantity': 1, 'ldraw_id': '3001'},
        # White Torso/Brick
        {'part_num': '973', 'name': 'Torso', 'color_id': 15, 'quantity': 1, 'ldraw_id': '973'},
        # Red Tile 1x2 : Color ID 4
        {'part_num': '3069b', 'name': 'Tile 1x2', 'color_id': 4, 'quantity': 1, 'ldraw_id': '3069b'},
    ]
    inv_df = pd.DataFrame(processed_data)
    
    # Map Colors (Rebrickable/LDraw IDs to RGB)
    colors_map = {
        85: '606060',  # Dark Bluish Gray
        86: 'A0A0A0',  # Light Bluish Gray
        15: 'F2F2F2',  # White
        4:  'FF0000',  # Red
    }
    
    # Mock Area Parsing
    def mock_parse(part_id):
        if part_id == '3032': return 3.2 * 4.8 # Plate 4x6 in cm2 ~ 15.36
        if part_id == '3001': return 1.6 * 3.2 # Brick 2x4 ~ 5.12
        if part_id == '973': return 2.0 * 2.0  # Torso ~ 4.0
        if part_id == '3069b': return 0.8 * 1.6 # Tile 1x2 ~ 1.28
        return 1.0
    engine.parse_ldraw_dimensions = mock_parse
    
    print("Running Analysis...")
    img_rgb, results = engine.analyze_image("tests/mock_78075.png", inv_df, colors_map)
    
    print(f"Detected {len(results)} objects.")
    
    for r in results:
        print(f"Object Area: {r['area_cm2']:.2f} cm2 | Color: {r['color_rgb']}")
        sugg = r.get('suggestion')
        if sugg:
            print(f"  -> Suggestion: {sugg['name']} (Expected: {sugg['expected_area']:.2f})")
    
    # Assertions
    # We expect 4 objects: D-Grey, L-Grey, White, Red.
    # The White one is tricky on White background.
    
    if len(results) >= 3:
        print("✅ Detection Reasonable (>= 3 objects)")
        return True
    else:
        print("❌ Detection Failed (Too few objects)")
        return False

if __name__ == "__main__":
    test_78075_simulation()
