import bpy

def check_nodes():
    # Make sure we are in a clean state or have a scene
    scene = bpy.context.scene
    scene.use_nodes = True
    
    # In some versions, use_nodes=True automatically creates Render Layers and Composite nodes
    tree = getattr(scene, "node_tree", None) or getattr(scene, "compositing_node_group", None)
    
    if tree:
        print(f"TREE NODES: {[n.bl_idname for n in tree.nodes]}")
        for n in tree.nodes:
            print(f"  NODE: {n.name}, ID: {n.bl_idname}, Label: {n.bl_label}")
    else:
        print("NO TREE FOUND even after use_nodes=True")

if __name__ == "__main__":
    check_nodes()
