import open3d as o3d
import copy
import numpy as np

# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
# mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
# print(f'Center of mesh: {mesh.get_center()}')
# print(f'Center of mesh tx: {mesh_tx.get_center()}')
# print(f'Center of mesh ty: {mesh_ty.get_center()}')
# o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])



# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# mesh_mv = copy.deepcopy(mesh).translate((2, 2, 2), relative=False)
# print(f'Center of mesh: {mesh.get_center()}')
# print(f'Center of translated mesh: {mesh_mv.get_center()}')
# o3d.visualization.draw_geometries([mesh, mesh_mv])


# 카메라 내부 매개 변수
image_resolution = (2208, 1242)
fx = 1048.80224609375  # focal length in pixels
fy = 1048.80224609375
cx = 1104.6346435546875  # principal point in pixels
cy = 621.6849975585938

# 카메라 위치 및 방향
camera_position = np.array([-1.3044313379641588, 0.02674518353635602, 0.9266801808276479])
camera_rotation_euler = np.array([-1.9760258764979506, 0.003413526105369158, -1.5848662024427904])


world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

R = camera_frame.get_rotation_matrix_from_xyz(camera_rotation_euler)
camera_frame.rotate(R, center=(0, 0, 0))
camera_frame.translate(translation=camera_position)


v3d = o3d.utility.Vector3dVector(np.asarray([[0, -0.5, 0], [0, 0.5, 0]]))
cloud = o3d.geometry.PointCloud(v3d)
cloud.paint_uniform_color([1, 0, 0])


o3d.visualization.draw_geometries([camera_frame, world_frame, cloud])
