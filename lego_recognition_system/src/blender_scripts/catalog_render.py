import bpy
import sys
import json
import os
import random
import math
from mathutils import Vector, Euler
import addon_utils

# Add current dir to path to find local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from scene_setup import LDrawResolver, import_ldraw_part, setup_render_engine, clean_scene, setup_lighting, register_ldraw_addon

def setup_catalog_camera():
    """Setup a tight zenithal camera for single parts."""
    cam_data = bpy.data.cameras.new("CatalogCam")
    cam_obj = bpy.data.objects.new("CatalogCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Position: Slightly higher than pile camera but with narrower zoom
    cam_obj.location = (0, 0, 0.4) 
    cam_obj.rotation_euler = (0, 0, 0)
    cam_data.lens = 85 # Telephoto for less perspective distortion
    return cam_obj

def main():
    # Usage: blender --background --python catalog_render.py -- config.json
    argv = sys.argv
    try:
        idx = argv.index("--")
        data_file = argv[idx + 1]
    except (ValueError, IndexError):
        print("No data file passed.")
        return

    with open(data_file, 'r') as f:
        data = json.load(f)

    parts = data['parts'] 
    output_base = data.get('output_base', '/content/catalog')
    assets_dir = data.get('assets_dir')
    ldraw_path_base = data.get('ldraw_path')
    
    os.makedirs(output_base, exist_ok=True)
    images_dir = os.path.join(output_base, "images")
    os.makedirs(images_dir, exist_ok=True)

    register_ldraw_addon(data.get('addon_path'))
    clean_scene()
    setup_render_engine(data.get('render_engine', 'CYCLES'))
    cam = setup_catalog_camera()
    setup_lighting()
    
    resolver = LDrawResolver(ldraw_path_base)
    
    # Optional: Add a simple white/grey ground for consistency
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,-0.001))
    ground = bpy.context.object
    mat = bpy.data.materials.new(name="CatalogGround")
    mat.use_nodes = True
    mat.node_tree.nodes.get("Principled BSDF").inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
    ground.data.materials.append(mat)

    for part in parts:
        ldraw_id = part['ldraw_id']
        part_path = resolver.find_part(ldraw_id)
        
        if not part_path:
            continue
            
        print(f"📸 Rendering Catalog Image for: {ldraw_id}")
        
        # 1. Import Part
        obj = import_ldraw_part(part_path, ldraw_path_base)
        if not obj: continue
        
        # 2. Center and Fit
        obj.location = (0, 0, 0)
        # Random rotation for catalog? Usually, 1-3 fixed angles are better.
        # Let's do a 45/45 degree isometric-ish view
        obj.rotation_euler = (math.radians(30), 0, math.radians(45))
        
        # 3. Render
        render_path = os.path.join(images_dir, f"{ldraw_id}.png")
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        
        # 4. Clean up for next part
        # Delete the part hierarchy
        to_delete = [obj] + list(obj.children_recursive)
        for o in to_delete:
            bpy.data.objects.remove(o, do_unlink=True)
        
        # Clear unused meshes/materials
        for block in bpy.data.meshes:
            if block.users == 0: bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0: bpy.data.materials.remove(block)

    print("✅ Catalog rendering complete.")

if __name__ == "__main__":
    main()
