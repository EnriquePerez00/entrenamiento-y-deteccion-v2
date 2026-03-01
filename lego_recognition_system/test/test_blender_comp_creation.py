import bpy

def test_creation():
    scene = bpy.context.scene
    print(f"Initial comp group: {scene.compositing_node_group}")
    
    # Create new tree
    new_tree = bpy.data.node_groups.new("MyComp", "CompositorNodeTree")
    scene.compositing_node_group = new_tree
    
    print(f"Assigned comp group: {scene.compositing_node_group}")
    print(f"Tree nodes: {list(new_tree.nodes)}")
    
    # Try common output node names
    for name in ['Composite', 'Output', 'Viewer', 'OutputFile']:
        try:
            node = new_tree.nodes.new('CompositorNode' + name)
            print(f"SUCCESS creating CompositorNode{name}: {node.bl_idname}")
        except Exception as e:
            print(f"FAILED creating CompositorNode{name}: {e}")

if __name__ == "__main__":
    test_creation()
