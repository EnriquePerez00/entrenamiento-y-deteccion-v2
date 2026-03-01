import bpy
import sys

def check_scale_node():
    scene = bpy.context.scene
    scene.use_nodes = True
    
    comp_tree = getattr(scene, "node_tree", None)
    if comp_tree is None and hasattr(scene, "compositing_node_group"):
        if scene.compositing_node_group is None:
            scene.compositing_node_group = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
        comp_tree = scene.compositing_node_group

    if not comp_tree:
        print("COULD NOT FIND COMP TREE")
        return

    node = comp_tree.nodes.new('CompositorNodeScale')
    print(f"NODE TYPE: {node.type}")
    print(f"PROPS: {[p.identifier for p in node.bl_rna.properties if not p.is_readonly]}")
    print(f"INPUTS: {[i.name for i in node.inputs]}")
    
    # Try to find something related to RENDER_SIZE
    for p in node.bl_rna.properties:
        if p.type == 'ENUM':
            print(f"  ENUM {p.identifier}: {[o.identifier for o in p.enum_items]}")

if __name__ == "__main__":
    check_scale_node()
