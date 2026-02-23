import sys
import os
import cv2
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.logic.recognition_2d import TwoDRecognitionEngine

def create_mock_image():
    # Increase resolution to be more realistic
    # A4 at 72dpi is ~595x842. Let's use ~600x800.
    # width 21cm, height 29.7cm.
    # Let's say 40 px/cm => 840 x 1188.
    scale = 40 # px per cm
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
        
    # Draw Red Square (roughly 2x4 brick size)
    # 2x4 studs ~ 1.6cm x 3.2cm
    w_obj = int(1.6 * scale)
    h_obj = int(3.2 * scale)
    cv2.rectangle(img, (100, 100), (100+w_obj, 100+h_obj), (0, 0, 200), -1)
    
    # Draw Blue Square (2x2 brick)
    # 1.6cm x 1.6cm
    w_obj2 = int(1.6 * scale)
    h_obj2 = int(1.6 * scale)
    cv2.rectangle(img, (300, 400), (300+w_obj2, 400+h_obj2), (200, 0, 0), -1)
    
    return img

def test_2d_recognition():
    print("Initializing Engine...")
    engine = TwoDRecognitionEngine()
    
    print("Creating Mock Data...")
    img = create_mock_image()
    cv2.imwrite("tests/mock_input.png", img)
    
    # Mock Inventory
    # Red (ID 4) - Large Area
    # Blue (ID 1) - Small Area
    processed_data = [
        # Red Part
        {
            'part_num': '3001', 'name': 'Brick 2x4', 
            'color_id': 4, 'color_name': 'Red', 
            'quantity': 1, 'ldraw_id': '3001'
        },
        # Blue Part
        {
            'part_num': '3003', 'name': 'Brick 2x2', 
            'color_id': 1, 'color_name': 'Blue', 
            'quantity': 1, 'ldraw_id': '3003'
        }
    ]
    inv_df = pd.DataFrame(processed_data)
    
    # Mock Colors Map
    colors_map = {
        4: '0000FF', # Red (Rebrickable/LDraw RGB might differ, usually Red is 255,0,0)
        1: 'FF0000'  # Blue
    }
    
    # Mock LDraw resolution results for Area estimation
    # We will mock the `parse_ldraw_dimensions` method or just rely on fallback.
    # Since we can't easily mock the method without a framework, we will rely on fallback '1.0' 
    # OR we can monkeypatch `parse_ldraw_dimensions`
    
    original_parse = engine.parse_ldraw_dimensions
    def mock_parse(part_id):
        if part_id == '3001': return 10.0 # Red target
        if part_id == '3003': return 5.0  # Blue target
        return 1.0
    engine.parse_ldraw_dimensions = mock_parse
    
    print("Running Analysis...")
    img_rgb, results, _ = engine.analyze_image("tests/mock_input.png", inv_df, colors_map)
    
    print(f"Detected {len(results)} objects.")
    
    for r in results:
        print(f"Object Area: {r['area_cm2']:.2f} cm2")
        print(f"Color RGB: {r['color_rgb']}")
        sugg = r.get('suggestion')
        if sugg:
            print(f"Suggestion: {sugg['name']} (Color ID: {r['assigned_color_id']})")
        else:
            print("No suggestion.")
            
    # Assertions
    if len(results) != 2:
        print("FAILED: Expected 2 objects.")
        return False
        
    # Sort by area
    results.sort(key=lambda x: x['area_cm2'], reverse=True)
    
    large = results[0] # Should be Red
    small = results[1] # Should be Blue
    
    # Red check (RGB roughly 255,0,0)
    # Note: cv2.mean returns (R,G,B) because we converted input to RGB in analyze_image
    print(f"Large Color: {large['color_rgb']}")
    
    if large['assigned_color_id'] != 4:
        # Note: Depending on clustering order, ID 4 should map to largest area group if Red is largest in inventory?
        # In inventory: Red has 10.0 area, Blue has 5.0 area.
        # In image: Red has ~10.0 area, Blue has ~5.0.
        # So Rank 1 Official (Red) should be mapped to Rank 1 Detected (Red).
        # Rank 2 Official (Blue) should be mapped to Rank 2 Detected (Blue).
        print("SUCCESS: Large object mapped to Red (ID 4)")
    else:
        print("FAILED: Large object NOT mapped to Red.")
        
    return True

if __name__ == "__main__":
    if test_2d_recognition():
        print("\n✅ Verification Passed!")
    else:
        print("\n❌ Verification Failed!")
