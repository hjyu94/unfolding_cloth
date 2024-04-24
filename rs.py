import numpy as np
import pyrealsense2 as rs


image_resolution = (2208, 1242)
fx = 1048.80224609375       # focal length in pixels
fy = 1048.80224609375
cx = 1104.6346435546875     # principal point in pixels
cy = 621.6849975585938

# 카메라 위치 및 방향
camera_position = np.array([-1.3044313379641588, 0.02674518353635602, 0.9266801808276479])
camera_rotation_euler = np.array([-1.9760258764979506, 0.003413526105369158, -1.5848662024427904])


intrinsic = rs.intrinsics()
intrinsic.width = image_resolution[0]
intrinsic.height = image_resolution[1]
intrinsic.ppx = cx
intrinsic.ppy = cy
intrinsic.fx = fx
intrinsic.fy = fy
intrinsic.model = rs.distortion.none

x = 1030+194
y = 186+565
depth = 1

result = rs.rs2_deproject_pixel_to_point(intrinsic, [x, y], depth)
print(result)

extrinsic = rs.extrinsics()
extrinsic.translation = camera_position

roll = camera_rotation_euler[0]
pitch = camera_rotation_euler[1]
yaw = camera_rotation_euler[2]

R_x = np.array([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]])
R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])
R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])
R = np.dot(R_z, np.dot(R_y, R_x))

extrinsic.rotation = R

from_point = [-0.07116179168224335, -0.41541194915771484, 1.0]
to_point = rs.rs2_transform_point_to_point(extrinsic, from_point)
print(to_point)

# [-0.07116179168224335, -0.41541194915771484, 1.0]
# [0.11381112039089203, 0.1232977956533432, 1.0]