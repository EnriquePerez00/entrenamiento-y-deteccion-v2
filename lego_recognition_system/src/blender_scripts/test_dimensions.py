import bpy
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, -0.001))
ground = bpy.context.object
ground.scale = (0.25, 0.25, 1)
bpy.context.view_layer.update()
print("DIMENSIONS:", ground.dimensions)
