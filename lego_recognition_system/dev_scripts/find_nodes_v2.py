import bpy

def find_nodes():
    print("SEARCHING FOR COMPOSITE IN BPY.TYPES")
    for k in dir(bpy.types):
        if 'Composite' in k:
            print(f"TYPE: {k}")
    
    # Check what nodes are available in a new tree
    scene = bpy.context.scene
    nt = bpy.data.node_groups.new("Test", "CompositorNodeTree")
    
    # Try all prefixes
    prefixes = ['CompositorNode', 'Node', '']
    names = ['Composite', 'Output', 'Final', 'RenderOutput', 'Result']
    
    for p in prefixes:
        for n in names:
            try:
                node = nt.nodes.new(p + n)
                print(f"SUCCESS: {p+n} -> {node.bl_idname}")
            except:
                pass

if __name__ == "__main__":
    find_nodes()
