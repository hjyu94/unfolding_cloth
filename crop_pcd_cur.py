import os
import open3d as o3d
import numpy as np
import cal_camera_vec
from datetime import datetime

MIN = -4
MAX = 4

# 카메라 내부 매개 변수
image_resolution = (2208, 1242)
fx = 1048.80224609375  # focal length in pixels
fy = 1048.80224609375
cx = 1104.6346435546875  # principal point in pixels
cy = 621.6849975585938

# 카메라 위치 및 방향
camera_position = np.array([-1.3044313379641588, 0.02674518353635602, 0.9266801808276479])
camera_rotation_euler = np.array([-1.9760258764979506, 0.003413526105369158, -1.5848662024427904])


def img_coord_to_camera_coord(pixel_x, pixel_y, front_vector, up_vector):
    # 픽셀 좌표를 카메라 좌표계로 변환
    pixel_in_camera_coords = np.array([
        (pixel_x - cx) / fx,
        (pixel_y - cy) / fy,
        1
    ])

    x_vec = np.cross(up_vector, front_vector)
    y_vec = -up_vector

    return (x_vec * pixel_in_camera_coords[0]) + (y_vec * pixel_in_camera_coords[1])



if __name__ == '__main__':
    root_dir = f"/home/hjeong/code/unfolding_cloth/datasets/temp/"
    input_ply_path = root_dir + "point_cloud.ply"
    output_ply_path = root_dir + f"crop_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.ply"

    front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec_from_json(
        json_path = root_dir + "camera_pose_in_world.json"
    )

    pcd = o3d.io.read_point_cloud(input_ply_path)

    # # center += camera_position
    # offset = 0.2
    #
    # min_bound = center-offset
    # min_bound[0] = MIN
    # max_bound = center+offset
    # max_bound[0] = MAX

    test = img_coord_to_camera_coord(cx, cy, front_vector, up_vector)
    test1 = img_coord_to_camera_coord(cx-100, cy, front_vector, up_vector)
    test2 = img_coord_to_camera_coord(cx-200, cy, front_vector, up_vector)
    test3 = img_coord_to_camera_coord(0, cy, front_vector, up_vector)

    test4 = img_coord_to_camera_coord(cx, cy-100, front_vector, up_vector)
    test5 = img_coord_to_camera_coord(cx, cy-200, front_vector, up_vector)


    test = img_coord_to_camera_coord(0, cy, front_vector, up_vector)
    test1 = img_coord_to_camera_coord(image_resolution[0], cy, front_vector, up_vector)

    left_top = img_coord_to_camera_coord(1030, 186, front_vector, up_vector)
    right_down = img_coord_to_camera_coord(1030+194, 186+565, front_vector, up_vector)

    left_top[0] = MAX
    right_down[0] = MIN

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound = right_down, max_bound=left_top)

    cropped = pcd.crop(bbox)

    o3d.visualization.draw_geometries([cropped], zoom=0.1, front=front_vector, lookat=look_at_vector, up=up_vector)
    # o3d.io.write_point_cloud(output_ply_path, cropped)

