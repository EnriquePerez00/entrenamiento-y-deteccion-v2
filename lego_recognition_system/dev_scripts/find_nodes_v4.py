import bpy

def find_output_node():
    scene = bpy.context.scene
    nt = bpy.data.node_groups.new("Test", "CompositorNodeTree")
    
    # Try every possible CompositorNode* type from dir(bpy.types)
    possible_types = [n for n in dir(bpy.types) if 'CompositorNode' in n]
    
    for t in possible_types:
        try:
            # The bl_idname is usually what's needed for nodes.new()
            # but sometimes it differs.
            # Let's try to get it from the class
            cls = getattr(bpy.types, t)
            if hasattr(cls, 'bl_rna'):
                # Try to create it
                try:
                    node = nt.nodes.new(t)
                    print(f"CREATED: {t}")
                    # Check if it has an output-like label
                    if 'Composite' in node.bl_label or 'Output' in node.bl_label:
                        print(f"  FOUND POTENTIAL OUTPUT: {t} (Label: {node.bl_label})")
                except:
                     # Try with the bl_idname if it exists and is different
                     if hasattr(cls, 'bl_idname') and cls.bl_idname != t:
                         try:
                             node = nt.nodes.new(cls.bl_idname)
                             print(f"CREATED via ID: {cls.bl_idname}")
                         except: pass
        except:
            pass

if __name__ == "__main__":
    find_output_node()
