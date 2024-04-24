import numpy as np


def image_pixel_to_camera_coordinates(u, v, fx, fy, cx, cy, depth):
    # 이미지 상의 픽셀 좌표(u, v)를 카메라 좌표계로 변환합니다.
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return x, y, z


def camera_coordinates_to_world_coordinates(x, y, z, camera_position, camera_rotation_euler):
    # 카메라 좌표계에서 뷰 좌표계로 변환합니다.
    rotation_matrix = np.array([
        [np.cos(camera_rotation_euler[2]), -np.sin(camera_rotation_euler[2]), 0],
        [np.sin(camera_rotation_euler[2]), np.cos(camera_rotation_euler[2]), 0],
        [0, 0, 1]
    ])
    camera_coords = np.dot(rotation_matrix, np.array([x, y, z]))

    # 월드 좌표계로 변환합니다.
    world_coords = camera_coords + camera_position

    return world_coords


# 주어진 데이터
camera_intrinsic = {
    "image_resolution": {"width": 2208, "height": 1242},
    "focal_lengths_in_pixels": {"fx": 1048.80224609375, "fy": 1048.80224609375},
    "principal_point_in_pixels": {"cx": 1104.6346435546875, "cy": 621.6849975585938}
}

camera_pose = {
    "position_in_meters": {"x": -1.3044313379641588, "y": 0.02674518353635602, "z": 0.9266801808276479},
    "rotation_euler_xyz_in_radians": {"roll": -1.9760258764979506, "pitch": 0.003413526105369158,
                                      "yaw": -1.5848662024427904}
}


# 카메라 내부 매개 변수 추출
fx = camera_intrinsic["focal_lengths_in_pixels"]["fx"]
fy = camera_intrinsic["focal_lengths_in_pixels"]["fy"]
cx = camera_intrinsic["principal_point_in_pixels"]["cx"]
cy = camera_intrinsic["principal_point_in_pixels"]["cy"]


# 이미지 상의 픽셀 좌표
u = cx
v = 1242

# 이미지 상의 깊이 정보 (예시)
depth = 1.5


# 카메라 위치 및 방향 추출
camera_position = np.array([camera_pose["position_in_meters"]["x"], camera_pose["position_in_meters"]["y"],
                            camera_pose["position_in_meters"]["z"]])
camera_rotation_euler = np.array(
    [camera_pose["rotation_euler_xyz_in_radians"]["roll"], camera_pose["rotation_euler_xyz_in_radians"]["pitch"],
     camera_pose["rotation_euler_xyz_in_radians"]["yaw"]])

# 이미지 상의 (u, v) 좌표를 (x, y, z) 좌표로 변환
x, y, z = image_pixel_to_camera_coordinates(u, v, fx, fy, cx, cy, depth)

# 카메라 좌표를 월드 좌표로 변환
world_coords = camera_coordinates_to_world_coordinates(x, y, z, camera_position, camera_rotation_euler)

print("이미지 상의 픽셀 좌표 (u, v):", u, v)
print("변환된 월드 좌표 (x, y, z):", world_coords)