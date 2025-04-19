import open3d as o3d


mesh_path = "./pings_experiments/test_ipbcar_gs_ours_ipb_car__2025-04-19_18-28-46/mesh/mesh_15cm.ply"

mesh = o3d.io.read_triangle_mesh(mesh_path)
print(mesh)

mesh.compute_vertex_normals()

mesh.paint_uniform_color([0.7, 0.7, 0.7])  # light gray

o3d.visualization.draw_geometries([mesh])