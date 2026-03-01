import bpy

def check_input_enum():
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = getattr(scene, "node_tree", None) or getattr(scene, "compositing_node_group", None)
    if tree is None and hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
        tree = scene.compositing_node_group
    
    node = tree.nodes.new('CompositorNodeScale')
    inp = node.inputs['Type']
    print(f"INPUT Type Type: {type(inp)}")
    print(f"INPUT Type dir: {dir(inp)}")
    if hasattr(inp, "default_value"):
        print(f"INPUT Type default_value: {inp.default_value}")
    
    # Try to set it
    try:
        inp.default_value = 'RENDER_SIZE'
        print("Set default_value to RENDER_SIZE success")
    except Exception as e:
        print(f"Set default_value to RENDER_SIZE failed: {e}")

    try:
        # In some versions it might be an index
        inp.default_value = 3 # Render Size is often the 4th item
        print("Set default_value to 3 success")
    except Exception as e:
        print(f"Set default_value to 3 failed: {e}")

if __name__ == "__main__":
    check_input_enum()
