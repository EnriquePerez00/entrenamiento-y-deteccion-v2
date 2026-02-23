import bpy
import sys
import json
import os
import random
import math
from mathutils import Vector, Euler

# Add current dir to path to find local modules if needed (optional)
import addon_utils

# Add current dir to path to find local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

def setup_render_engine():
    """Configure render engine. Uses CYCLES with OPTIX/CUDA hardware ray tracing."""
    scene = bpy.context.scene
    print("🛠️ Configuring Render Engine: CYCLES")
    
    # Common render settings
    scene.render.resolution_x = 960
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    
    # CYCLES: Force OptiX (>30% faster than CUDA on T4 Turing). 
    # CUDA_VISIBLE_DEVICES is set per-worker in the calling subprocess.
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()
    
    device_configured = False
    # OptiX first (hardware ray tracing), then CUDA, then METAL
    for device_type in ['OPTIX', 'CUDA', 'METAL']:
        try:
            cprefs.compute_device_type = device_type
            for device in cprefs.devices:
                if device.type != 'CPU':
                    device.use = True
                    device_configured = True
            if device_configured:
                print(f"  🔬 CYCLES: {device_type} hardware ray-tracing activated.")
                break
        except Exception:
            pass
    
    if not device_configured:
        print("  ⚠️ CYCLES: No GPU found, falling back to CPU.")
        scene.cycles.device = 'CPU'
    
    # Eco-Mode: fast rendering, try AI denoising if supported by this Blender version
    scene.cycles.samples = 64  # Increased for higher quality without sacrificing much time on dual GPUs
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.1
    
    # Try to enable AI denoising — Blender 3.x and 4.x have different APIs
    denoiser_set = False
    for denoiser_name in ['OPTIX', 'OPENIMAGEDENOISE']:
        try:
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = denoiser_name
            print(f"  🧠 Denoiser: {denoiser_name} active.")
            denoiser_set = True
            break
        except (TypeError, AttributeError):
            continue
    if not denoiser_set:
        scene.cycles.use_denoising = False
        print("  ℹ️ Denoising not supported on this Blender version. Disabled.")
    
    # Minimal bounces for speed
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_bounces = 2
    scene.cycles.transmission_bounces = 0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.render.use_persistent_data = True
    print("  ⚡ Cycles Eco-Mode: 8 samples + OptiX denoising.")

def register_ldraw_addon(addon_path=None):
    """Register the ImportLDraw addon from a dynamic path or local script dir."""
    print("🚀 Running register_ldraw_addon (CODE VERSION 2.2)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not addon_path:
        # Try to find ImportLDraw folder next to this script
        potential_path = os.path.join(script_dir, "ImportLDraw")
        if os.path.isdir(potential_path):
            addon_path = script_dir
            print(f"📦 Found local ImportLDraw addon at: {potential_path}")

    if addon_path and os.path.exists(addon_path):
        if addon_path not in sys.path:
            sys.path.append(addon_path)
        print(f"📦 Added addon path to sys.path: {addon_path}")

    # Try to import and register. Support both 'ImportLDraw' and 'io_scene_importldraw' (standard)
    success = False
    for module_name in ['ImportLDraw', 'io_scene_importldraw']:
        try:
            mod = __import__(module_name)
            mod.register()
            print(f"✅ Addon registered successfully via module: {module_name}")
            success = True
            break
        except Exception:
            continue

    if not success:
        print("⚠️ Failed to register ImportLDraw addon via ImportLDraw or io_scene_importldraw.")
        print("   Checking if it's already registered via standard installation...")
        if "importldraw" not in dir(bpy.ops.import_scene):
            raise Exception("❌ CRITICAL: ImportLDraw addon is NOT available. Aborting.")

def clean_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_camera():
    """Create and return a camera object."""
    bpy.ops.object.camera_add(location=(0, 0, 10))
    cam = bpy.context.object
    cam.data.lens = 50
    bpy.context.scene.camera = cam
    return cam

def setup_lighting():
    """Create random lighting setup."""
    # Delete existing lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()

    # Create Key Light
    bpy.ops.object.light_add(type='AREA', location=(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(5, 10)))
    key_light = bpy.context.object
    key_light.data.energy = random.uniform(500, 2500) # Increased dynamic range (was 800-1500)
    key_light.data.color = (random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)) # Stronger color tints
    key_light.scale = (random.uniform(2, 6), random.uniform(2, 6), 1)

    # Fill Light
    bpy.ops.object.light_add(type='POINT', location=(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(2, 5)))
    fill_light = bpy.context.object
    fill_light.data.energy = random.uniform(50, 800)
    
    return key_light, fill_light


def import_ldraw_part(filepath, ldraw_lib=None):
    """Import LDraw part using ImportLDraw addon."""
    try:
        # Explicitly pass the library path to the operator
        # Deselect all to ensure we only get the new object
        bpy.ops.object.select_all(action='DESELECT')

        if ldraw_lib:
            bpy.ops.import_scene.importldraw(filepath=filepath, ldrawPath=ldraw_lib, addEnvironment=False, positionCamera=False)
        else:
            bpy.ops.import_scene.importldraw(filepath=filepath, addEnvironment=False, positionCamera=False)
        
        # The imported object is usually selected
        # ImportLDraw often imports a hierarchy (Empty -> Mesh). 
        # We need the root object that contains the mesh or valid dimensions.
        selected = bpy.context.selected_objects
        if not selected:
             return None
             
        # Ideally, we want the parent object if it exists
        obj = selected[0]
        while obj.parent:
            obj = obj.parent
            
        return obj
    except Exception as e:
        print(f"Failed to import {filepath}: {e}")
        return None

def get_hierarchy_corners(obj):
    """Recursively find all bounding box corners in an object hierarchy in world space."""
    corners = []
    if obj.type == 'MESH':
        mw = obj.matrix_world
        corners.extend([mw @ Vector(corner) for corner in obj.bound_box])
    for child in obj.children:
        corners.extend(get_hierarchy_corners(child))
    return corners

def get_hierarchy_vertices(obj):
    """Recursively find all vertices in an object hierarchy in world space."""
    verts = []
    if obj.type == 'MESH':
        mw = obj.matrix_world
        verts.extend([mw @ v.co for v in obj.data.vertices])
    for child in obj.children:
        verts.extend(get_hierarchy_vertices(child))
    return verts

def get_convex_hull(points):
    """Computes the convex hull of a set of 2D points using Graham scan."""
    points = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(points) <= 1:
        return points
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
        
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
        
    return lower[:-1] + upper[:-1]

def get_mabr(hull):
    """Computes the Minimum Area Bounding Rectangle (OBB) for a convex hull."""
    import math
    min_area = float('inf')
    best_rect = None
    n = len(hull)
    if n < 3:
        # Fallback for degenerate flat hulls (just make a straight box)
        if n == 0: return None
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        
    for i in range(n):
        p1 = hull[i]
        p2 = hull[(i+1)%n]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length == 0: continue
        
        ux, uy = dx / length, dy / length
        vx, vy = -uy, ux
        
        min_u, max_u = float('inf'), float('-inf')
        min_v, max_v = float('inf'), float('-inf')
        
        for p in hull:
            u = p[0] * ux + p[1] * uy
            v = p[0] * vx + p[1] * vy
            if u < min_u: min_u = u
            if u > max_u: max_u = u
            if v < min_v: min_v = v
            if v > max_v: max_v = v
            
        area = (max_u - min_u) * (max_v - min_v)
        if area < min_area:
            min_area = area
            c1 = (min_u * ux + min_v * vx, min_u * uy + min_v * vy)
            c2 = (max_u * ux + min_v * vx, max_u * uy + min_v * vy)
            c3 = (max_u * ux + max_v * vx, max_u * uy + max_v * vy)
            c4 = (min_u * ux + max_v * vx, min_u * uy + max_v * vy)
            best_rect = [c1, c2, c3, c4]
            
    return best_rect

# ... (existing imports)

# Import LDrawResolver from local module
try:
    from ldraw_resolver import LDrawResolver
except ImportError:
    # If running from Blender, verify sys.path
    pass


def main():
    # Get arguments passed after "--"
    argv = sys.argv
    try:
        idx = argv.index("--")
        data_file = argv[idx + 1]
    except (ValueError, IndexError):
        print("No data file passed to Blender script.")
        return

    with open(data_file, 'r') as f:
        data = json.load(f)

    set_id = data['set_id']
    parts = data['parts'] # List of {ldraw_id, color_id, ...}
    num_images = data['num_images']
    output_base = data.get('output_base', '/content/dataset')
    assets_dir = data.get('assets_dir')
    ldraw_path_base = data.get('ldraw_path') 
    addon_path = data.get('addon_path') # Path to ImportLDraw folder parent

    print(f"🚀 Blender started for Set {set_id}")
    print(f"📍 Output path: {output_base}")

    # Register Addon from dynamic path or local script dir
    register_ldraw_addon(data.get('addon_path'))

    # Initialize variables to avoid NameError if logic is skipped
    unique_meshes = []
    physics_objects = []
    total_spawned = 0

    # Validate and Auto-Fix LDraw Path
    if ldraw_path_base:
        # Check root
        p_root = os.path.join(ldraw_path_base, "p")
        parts_root = os.path.join(ldraw_path_base, "parts")
        
        if not (os.path.isdir(p_root) and os.path.isdir(parts_root)):
            # Try to find a subfolder that contains them (case insensitive ldraw/LDraw)
            found_inner = False
            for d in os.listdir(ldraw_path_base):
                inner_path = os.path.join(ldraw_path_base, d)
                if os.path.isdir(inner_path):
                    if os.path.isdir(os.path.join(inner_path, "p")) and os.path.isdir(os.path.join(inner_path, "parts")):
                        ldraw_path_base = inner_path
                        found_inner = True
                        print(f"📂 Auto-detected LDraw subfolder: {ldraw_path_base}")
                        break
            
            if not found_inner:
                msg = f"❌ CRITICAL: LDraw path '{ldraw_path_base}' is invalid (missing 'p' or 'parts' subfolders)."
                print(msg)
                raise Exception(msg)
    else:
        print("⚠️ No LDraw path provided. Expecting global library setup.")

    # Apply to Addon Preferences
    if ldraw_path_base:
        try:
            bpy.context.preferences.addons['import_scene_importldraw'].preferences.ldrawPath = ldraw_path_base
        except Exception:
            try:
                bpy.context.preferences.addons['ImportLDraw'].preferences.ldrawPath = ldraw_path_base
            except Exception:
                pass

    os.makedirs(output_base, exist_ok=True)
    images_dir = os.path.join(output_base, "images")
    labels_dir = os.path.join(output_base, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    clean_scene()
    setup_render_engine() # Default and only engine is CYCLES
    cam = setup_camera()
    key_light, fill_light = setup_lighting()
    
    # Setup Color Management to avoid "Colorspace not found" and ensure sRGB standard
    # effective for training data to match standard camera input
    try:
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.view_settings.look = 'None' 
        bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'
    except Exception as e:
        print(f"Warning: Could not set color management: {e}")

    # Create Ground Plane
    # In Blender >= 2.8, 'size' is the total width. We want a 50x50cm surface.
    bpy.ops.mesh.primitive_plane_add(size=0.5, location=(0, 0, -0.001))
    ground = bpy.context.object
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    
    # Material with Texture Support
    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    
    # Check for background image
    bg_path = os.path.join(assets_dir, "backgrounds", "background.jpg")
    
    # 1. Add NOISE TEXTURE to make the surface "dirty/rough/complex" to lower mAP artificially
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = random.uniform(50, 250)
    noise.inputs['Detail'].default_value = 15
    
    # Mix RGB Node to multiply dirt over the background image
    mix_rgb = nodes.new('ShaderNodeMixRGB')
    mix_rgb.blend_type = 'MULTIPLY'
    mix_rgb.inputs['Fac'].default_value = random.uniform(0.1, 0.45) # 10-45% dirt intensity
    
    links.new(noise.outputs['Color'], mix_rgb.inputs[2])

    if os.path.exists(bg_path):
        print(f"🖼️ Loading background texture: {bg_path}")
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = bpy.data.images.load(bg_path)
        links.new(tex_node.outputs['Color'], mix_rgb.inputs[1])
    else:
        print("💡 No background image found at assets/backgrounds/background.jpg. Using random color.")
        rgb_node = nodes.new('ShaderNodeRGB')
        rgb_node.outputs['Color'].default_value = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), 1)
        links.new(rgb_node.outputs['Color'], mix_rgb.inputs[1])

    links.new(mix_rgb.outputs['Color'], bsdf.inputs['Base Color'])

    # Extreme variation in surface reflectivity per image
    bsdf.inputs['Roughness'].default_value = random.uniform(0.3, 0.9)
    if 'Specular IOR Level' in bsdf.inputs: # For latest Blender 4.0 BSDF
        bsdf.inputs['Specular IOR Level'].default_value = random.uniform(0.0, 0.8)
    elif 'Specular' in bsdf.inputs: # For older Blender
        bsdf.inputs['Specular'].default_value = random.uniform(0.0, 0.8)
        
    ground.data.materials.append(mat)


    # --- CONFIGURATION FOR HIGH DENSITY ---
    # User Request: 50x50cm area, 70cm height camera. 
    # Prototype: 3 parts, but we need high density (up to 500 total).
    
    PARTS_PER_TYPE = data.get('parts_per_type', 20) # Default to 20 if not specified
    
    # 1. Setup Camera (Fixed Zenithal Centered)
    # Surface is 50x50cm centered at 0,0.
    cam.location = (0, 0, 0.70) 
    cam.rotation_euler = (0, 0, 0) 
    cam.data.lens = 50 

    # Initialize Resolver
    resolver = LDrawResolver(ldraw_path_base)
    print(f"DEBUG: Resolver initialized. Methods: {dir(resolver)}")
    


    for i, part in enumerate(parts):
        ldraw_id = part['ldraw_id']
        part_name = f"{ldraw_id}.dat"
        
        # Strategy C: Universal Detector mode forces all class IDs to 0
        if os.environ.get("UNIVERSAL_DETECTOR", "0") == "1":
            class_id = 0
            if i == 0: print("🛠️ Universal Detector Mode: All pieces will be labeled as class 0.")
        else:
            class_id = i
        
        part_path = resolver.find_part(ldraw_id)
        if not part_path:
            continue
            
        obj = import_ldraw_part(part_path, ldraw_path_base)
        if obj:
            unique_meshes.append({'obj': obj, 'id': class_id})
            obj.location = (0, 0, -10) # move template out of sight
            
    # Spawn Copies

    for template in unique_meshes:
        template_obj = template['obj']
        class_id = template['id']
        
        for n in range(PARTS_PER_TYPE):
            # Duplicate
            new_obj = template_obj.copy()
            if template_obj.data:
                new_obj.data = template_obj.data.copy() # Full copy to be safe with modifiers
            bpy.context.collection.objects.link(new_obj)
            # Position Randomly in a pile area (40x40cm center to keep away from the 50cm edges)
            # The surface is strictly [-0.25, 0.25] in XY. 
            # We spawn tightly within [-0.20, 0.20] so pieces 100% land on the mat and never fall off.
            new_obj.location = (
                random.uniform(-0.20, 0.20), 
                random.uniform(-0.20, 0.20), 
                random.uniform(0.05, 0.35)  # Drop from higher up
            )
            new_obj.rotation_euler = (random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14))
            
            # Physics
            bpy.context.view_layer.objects.active = new_obj
            bpy.ops.rigidbody.object_add()
            new_obj.rigid_body.type = 'ACTIVE'
            new_obj.rigid_body.mass = 0.05 # Lighter for Lego
            new_obj.rigid_body.friction = 0.8 # Plastic friction
            new_obj.rigid_body.collision_shape = 'CONVEX_HULL'
            # Damping to stop them jiggling forever
            new_obj.rigid_body.linear_damping = 0.5 
            new_obj.rigid_body.angular_damping = 0.5
            
            new_obj['class_id'] = class_id
            physics_objects.append(new_obj)
            total_spawned += 1
            
    # VALIDATION: Check if objects were imported
    if total_spawned == 0:
        msg = "❌ CRITICAL ERROR: No LEGO pieces were imported or spawned into the scene. Aborting render."
        print(msg)
        raise Exception(msg)
    
    print(f"✅ Successfully spawned {total_spawned} pieces ( {len(unique_meshes)} unique types).")
            
    # Hide templates again or delete?
    # Keeping them far away is fine.

    # Save missing parts report
    resolver.save_report(output_base)

    # Simulation Loop
    import bpy_extras
    
    scene = bpy.context.scene
    
    # 3. Physics Simulation (The Drop)
    SETTLE_FRAMES = 40 # Reduced from 60/80 for speed
    scene.frame_end = SETTLE_FRAMES + 10
    
    offset_idx = data.get('offset_idx', 0)
    
    # We will do 2 variations (lighting/jitter) per physical drop to save CPU time
    # Total physical drops = num_images // 2
    num_drops = max(1, num_images // 2)
    
    for drop_idx in range(num_drops):
        # 4a. Reset Positions (Shuffle)
        scene.frame_set(1)
        bpy.ops.ptcache.free_bake_all()
        
        for obj in physics_objects:
            obj.location = (random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), random.uniform(0.1, 0.4))
            obj.rotation_euler = (random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14))
        
        # 4b. Run Settle Simulation
        scene.frame_set(SETTLE_FRAMES)
        bpy.context.view_layer.update()
        
        # 4c. Render 2 variations per drop (Lighting diversity)
        for var_idx in range(2):
            local_idx = (drop_idx * 2) + var_idx
            if local_idx >= num_images: break
            
            img_idx = local_idx + offset_idx
            
            # Sub-variation: Randomize lights, color tint, and slight camera jitter
            key_light.location = (random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(3, 8))
            key_light.data.energy = random.uniform(500, 2500)
            key_light.data.color = (random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0))
            fill_light.data.energy = random.uniform(50, 1000)
            
            # Render
            render_path = os.path.join(images_dir, f"img_{img_idx:04d}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)
        
            # 5. Generate Labels (Bounding Boxes - OBB format)
            label_path = os.path.join(labels_dir, f"img_{img_idx:04d}.txt")
            
            with open(label_path, 'w') as lf:
                for obj in physics_objects: # Iterate over all spawned physics objects
                    camera = scene.camera
                    
                    # For OBB, using vertices is much more accurate than the 8 bounding box corners
                    # because the real shape limits the tight OBB fit.
                    verts_3d = get_hierarchy_vertices(obj)
                    if not verts_3d:
                        continue
                    
                    # Project vertices to 2D
                    coords_2d = [bpy_extras.object_utils.world_to_camera_view(scene, camera, coord) for coord in verts_3d]
                    
                    # Filter visible coords and clamp to 0-1 bounds
                    visible_points = []
                    for c in coords_2d:
                        if c.z > 0: # In front of camera
                            cx, cy = max(0.0, min(1.0, c.x)), max(0.0, min(1.0, c.y))
                            visible_points.append((cx, cy))
                            
                    if not visible_points: continue
                    
                    # Calculate Minimum Area Bounding Rectangle (MABR)
                    hull = get_convex_hull(visible_points)
                    mabr = get_mabr(hull)
                    
                    if not mabr: continue
                    
                    # Check area size to filter noise
                    xs = [p[0] for p in mabr]
                    ys = [p[1] for p in mabr]
                    if (max(xs) - min(xs)) < 0.002 or (max(ys) - min(ys)) < 0.002:
                        continue
                        
                    # Write YOLO OBB format (class x1 y1 x2 y2 x3 y3 x4 y4)
                    if 'class_id' in obj:
                        # Note: Blender Y goes up, YOLO Y goes down. We invert Y.
                        # Also YOLO-OBB expects x1 y1 x2 y2 x3 y3 x4 y4 normalized (0-1)
                        pts = []
                        for corner in mabr:
                            px = max(0.0, min(1.0, corner[0]))
                            py = max(0.0, min(1.0, 1.0 - corner[1])) # Invert Y for YOLO
                            pts.extend([f"{px:.6f}", f"{py:.6f}"])
                            
                        lf.write(f"{obj['class_id']} {' '.join(pts)}\n")

    # Create data.yaml for YOLO training
    data_yaml_path = os.path.join(output_base, "data.yaml")
    # Path to images should be absolute for training (or relative to data.yaml)
    # Since we move things in the notebook sometimes, let's use relative paths from data.yaml
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_base)}\n")
        f.write("train: images\n")
        f.write("val: images\n") # Use same for val in prototype
        f.write("\n")
        f.write(f"nc: {len(parts)}\n")
        f.write(f"names: {[p['ldraw_id'] for p in parts]}\n")

    print(f"✅ Created data.yaml at {data_yaml_path}")
    print("Blender script finished.")

if __name__ == "__main__":
    main()
