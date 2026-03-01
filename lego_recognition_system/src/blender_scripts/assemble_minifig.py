
import bpy
import os
import sys
import json
import random
from pathlib import Path

# Add script dir to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import scene_setup

def assemble_minifig(parts_list, ldraw_path, addon_path):
    """
    Assembles a minifigure from a list of parts with correct offsets.
    parts_list: [{'ldraw_id': '973', 'type': 'Torso', 'color_id': 0}, ...]
    """
    # Offsets in LDraw Units (LDU)
    # Standard minifig scale in LDraw: 1 brick height = 24 LDU
    # These are illustrative and may need tuning depending on the specific .dat origin
    OFFSETS = {
        'Head': (0, 0, 0),    
        'Hips': (0, 0, -16),  
        'Legs': (0, 0, -16),  
        'Hat': (0, 0, 0),     
        'Hair': (0, 0, 0),
    }

    torso_obj = None
    objects = []
    
    # 1. Import parts
    for part in parts_list:
        ldid = part['ldraw_id']
        color = part.get('color_id', 0)
        p_type = part.get('type', 'Unknown')
        
        # Import using the addon op
        bpy.ops.import_scene.importldraw(
            filepath=str(ldraw_path / "parts" / f"{ldid}.dat"),
            ldrawPath=str(ldraw_path),
            defaultColour=str(color),
            positionOnGround=False
        )
        
        # Get the imported object (usually the active one or newly selected)
        # Note: ImportLDraw often creates a collection or multiple objects
        # We'll grab the selected objects
        imported_objs = bpy.context.selected_objects
        if not imported_objs: continue
        
        main_obj = imported_objs[0] # Simplification
        objects.append((main_obj, p_type))
        
        if 'Torso' in p_type:
            torso_obj = main_obj
            torso_obj.location = (0, 0, 0)

    # 2. Position and Parent
    if torso_obj:
        for obj, p_type in objects:
            if obj == torso_obj: continue
            
            # Find best match for offset
            offset = (0,0,0)
            for key in OFFSETS:
                if key.lower() in p_type.lower():
                    offset = OFFSETS[key]
                    break
            
            # Parenting
            obj.parent = torso_obj
            obj.matrix_parent_inverse = torso_obj.matrix_world.inverted()
            
            # Apply offset in local space
            # Scale LDU to Blender units (1 LDU = 0.004 Blender units)
            scale = 0.004 
            obj.location = [c * scale for c in offset]
            
            # Reset rotation
            obj.rotation_euler = (0, 0, 0)

    # 3. Apply SSS (Subsurface Scattering) for Tier 3
    apply_minifig_materials()
    
    return torso_obj

def apply_minifig_materials():
    """Enhance plastic materials with Subsurface Scattering."""
    for mat in bpy.data.materials:
        if not mat.use_nodes: continue
        nodes = mat.node_tree.nodes
        # Look for Principled BSDF
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf:
            # SSS Settings for LEGO Plastic
            if 'Subsurface Weight' in bsdf.inputs: # Blender 4.0+
                bsdf.inputs['Subsurface Weight'].default_value = 0.1
                bsdf.inputs['Subsurface Radius'].default_value = (0.1, 0.1, 0.1)
            elif 'Subsurface' in bsdf.inputs: # Older Blender
                bsdf.inputs['Subsurface'].default_value = 0.05
            
            print(f"✨ Applied SSS to material: {mat.name}")

if __name__ == "__main__":
    # Internal test mode if run directly
    pass
