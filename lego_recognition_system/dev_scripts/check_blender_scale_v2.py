import bpy

def check_everything():
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = getattr(scene, "node_tree", None) or getattr(scene, "compositing_node_group", None)
    if tree is None and hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
        tree = scene.compositing_node_group
    
    node = tree.nodes.new('CompositorNodeScale')
    print(f"NODE: {node.name} ({node.bl_idname})")
    
    for p in node.bl_rna.properties:
        if p.type == 'ENUM':
            items = [o.identifier for o in p.enum_items]
            print(f"PROP ENUM {p.identifier}: {items}")
            
    for i in node.inputs:
        print(f"INPUT {i.name} ({i.identifier}, {i.type}): value={getattr(i, 'default_value', 'N/A')}")
        if hasattr(i, "enum_items"):
             print(f"  INPUT ENUM items: {[o.identifier for o in i.enum_items]}")
        # Some sockets in 5.0 might have their own properties or sub-enums
        
    # Check if there's a new node type for this
    print("ALL COMPOSITOR NODES:")
    # print([n for n in dir(bpy.types) if n.startswith("CompositorNode")])

if __name__ == "__main__":
    check_everything()
