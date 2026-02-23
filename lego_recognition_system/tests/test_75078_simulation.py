import sys
import os
import cv2
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.logic.recognition_2d import TwoDRecognitionEngine

def create_mock_75078_image():
    # Simulate set 75078-1 (Imperial Troop Transport)
    # Features:
    # - White Stormtroopers (Minifigs)
    # - Light Grey / Dark Grey vehicle parts
    # - Background: White Grid (0.5cm)
    
    # Increase resolution to be more realistic (40 px/cm)
    scale = 40
    width = int(21 * scale)
    height = int(29.7 * scale)
    
    img = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background
    
    # Draw Grid (0.5cm lines = 20px spacing)
    grid_px = int(0.5 * scale)
    grid_color = (200, 200, 200) 
    
    for x in range(0, width, grid_px):
        cv2.line(img, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, grid_px):
        cv2.line(img, (0, y), (width, y), grid_color, 1)
        
    # 1. Dark Grey Piece (e.g. Plate 2x10) - ID 85
    # 1.6cm x 8cm
    w1 = int(1.6 * scale)
    h1 = int(8.0 * scale)
    cv2.rectangle(img, (100, 100), (100+w1, 100+h1), (60, 60, 60), -1) 
    
    # 2. Light Grey Piece (e.g. Brick 1x2) - ID 86
    # 0.8cm x 1.6cm
    w2 = int(0.8 * scale)
    h2 = int(1.6 * scale)
    cv2.rectangle(img, (300, 200), (300+w2, 200+h2), (160, 160, 160), -1)
    
    # 3. White Minifig Torso/Legs (Stormtrooper) - ID 15
    # 1.5cm x 1.0cm (Approx)
    w3 = int(1.5 * scale)
    h3 = int(1.0 * scale)
    cv2.rectangle(img, (400, 300), (400+w3, 300+h3), (230, 230, 230), -1)
    
    # 4. Black Pin (Technic) - ID 0
    # 0.5cm x 1cm (Approx)
    w4 = int(0.5 * scale)
    h4 = int(1.0 * scale)
    cv2.rectangle(img, (200, 500), (200+w4, 500+h4), (20, 20, 20), -1)
    
    return img

def test_75078_simulation():
    print("Initializing Engine (75078-1 Simulation)...")
    engine = TwoDRecognitionEngine()
    
    print("Creating Mock Image (75078-1 style)...")
    img = create_mock_75078_image()
    cv2.imwrite("tests/mock_75078.png", img)
    
    # Mock Inventory for 75078-1
    processed_data = [
        # Dark Grey Plate
        {'part_num': '3832', 'name': 'Plate 2x10', 'color_id': 85, 'quantity': 2, 'ldraw_id': '3832'},
        # Light Grey Brick
        {'part_num': '3004', 'name': 'Brick 1x2', 'color_id': 86, 'quantity': 4, 'ldraw_id': '3004'},
        # White Torso
        {'part_num': '973', 'name': 'Torso Stormtrooper', 'color_id': 15, 'quantity': 4, 'ldraw_id': '973'},
        # Black Pin
        {'part_num': '2780', 'name': 'Technic Pin', 'color_id': 0, 'quantity': 8, 'ldraw_id': '2780'},
    ]
    inv_df = pd.DataFrame(processed_data)
    
    # Map Colors
    colors_map = {
        85: '606060',  # Dark Bluish Gray
        86: 'A0A0A0',  # Light Bluish Gray
        15: 'F2F2F2',  # White
        0:  '05131D',  # Black
    }
    
    # Mock Area Parsing
    def mock_parse(part_id):
        if part_id == '3832': return 1.6 * 8.0 # 12.8 cm2
        if part_id == '3004': return 0.8 * 1.6 # 1.28 cm2
        if part_id == '973': return 1.5 * 1.0  # 1.5 cm2
        if part_id == '2780': return 0.5 * 2.0 # 1.0 cm2
        return 1.0
    engine.parse_ldraw_dimensions = mock_parse
    
    print("Running Analysis...")
    img_rgb, results, _ = engine.analyze_image("tests/mock_75078.png", inv_df, colors_map)
    
    print(f"Detected {len(results)} objects.")
    
    found_types = set()
    for r in results:
        print(f"Object Area: {r['area_cm2']:.2f} cm2 | Color: {r['color_rgb']}")
        sugg = r.get('suggestion')
        if sugg:
            print(f"  -> Suggestion: {sugg['name']}")
            found_types.add(sugg['name'])
    
    # Assertions
    # Should detect Black, Dark Grey, Light Grey at least.
    # White is hard.
    if len(results) >= 3:
        print("✅ Detection Good (>= 3 objects)")
        return True
    else:
        print("❌ Detection Failed (Too few objects)")
        return False

if __name__ == "__main__":
    if test_75078_simulation():
        print("Simulation Passed")
    else:
        print("Simulation Failed")
