import numpy as np

# 카메라 내부 매개 변수
image_resolution = (2208, 1242)
fx = 1048.80224609375  # focal length in pixels
fy = 1048.80224609375
cx = 1104.6346435546875  # principal point in pixels
cy = 621.6849975585938

# 카메라 위치 및 방향
camera_position = np.array([-1.3044313379641588, 0.02674518353635602, 0.9266801808276479])
camera_rotation_euler = np.array([-1.9760258764979506, 0.003413526105369158, -1.5848662024427904])

# 픽셀 좌표 (x, y)
pixel_x = 1030
pixel_y = 186
# Camera Coordinates: [ 0.40798209 -1.36998866  1.92668018  1.        ]

# 픽셀 좌표를 카메라 좌표계로 변환
pixel_in_camera_coords = np.array([
    (pixel_x - cx) / fx,
    (pixel_y - cy) / fy,
    1
])

# 카메라 좌표계에서 뷰 좌표계로 변환
view_matrix = np.eye(4)
view_matrix[:3, 3] = -camera_position
rotation_matrix = np.array([
    [np.cos(camera_rotation_euler[2]), -np.sin(camera_rotation_euler[2]), 0],
    [np.sin(camera_rotation_euler[2]), np.cos(camera_rotation_euler[2]), 0],
    [0, 0, 1]
])
view_matrix[:3, :3] = rotation_matrix

camera_coords = np.dot(np.linalg.inv(view_matrix), np.append(pixel_in_camera_coords, 1))

# 결과를 출력
print("Camera Coordinates:", camera_coords)

# 이제 포인트 클라우드 상의 해당 좌표를 찾기 위해 필요한 단계를 따릅니다.
# 포인트 클라우드에서 카메라 좌표계로 변환하고, 이후에는 월드 좌표계로 변환합니다.