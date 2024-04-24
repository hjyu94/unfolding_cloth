import open3d as o3d
import numpy as np


def show_open3d_pcd(pcd, show_origin=True, origin_size=3, show_grid=True):
    cloud = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector

    if isinstance(pcd, type(cloud)):
        pass
    elif isinstance(pcd, np.ndarray):
        cloud.points = v3d(pcd)

    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))

    # set front, lookat, up, zoom to change initial view
    o3d.visualization.draw_geometries([cloud, coord])


points = np.asarray([[0, -0.5, 0], [0, 0.5, 0]])
show_open3d_pcd(points)


