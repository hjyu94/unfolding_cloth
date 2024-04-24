import os
import open3d as o3d
import numpy as np
import cal_camera_vec
from datetime import datetime

MIN = -4
MAX = 4


if __name__ == '__main__':
    root_dir = f"/home/hjeong/code/unfolding_cloth/datasets/temp/"
    input_ply_path = root_dir + "point_cloud.ply"
    output_ply_path = root_dir + f"crop_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.ply"

    front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec_from_json(
        json_path = root_dir + "camera_pose_in_world.json"
    )


    pcd = o3d.io.read_point_cloud(input_ply_path)

    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.3, 0.2), max_bound=(0.5, 0.2, 3))
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.5, -3), max_bound=(0.5, 0.5, 0.9))
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(MIN, MIN, MIN), max_bound=(MAX, 2, MAX))


    ###

    # center = np.array([ 0.40798209, -1.36998866,  1.92668018])
    # offset = 0.3
    #
    center = np.array([0, 0, 0])
    offset = np.array([0, 1.369, 0.770])


    min_bound = center-offset
    min_bound[0] = MIN
    max_bound = center+offset
    max_bound[1] = MAX

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    cropped = pcd.crop(bbox)

    o3d.visualization.draw_geometries([cropped], zoom=0.1, front=front_vector, lookat=look_at_vector, up=up_vector)
    # o3d.io.write_point_cloud(output_ply_path, cropped)

