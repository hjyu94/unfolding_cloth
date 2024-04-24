import numpy as np
import cal_camera_vec

camera_position = np.array([-1.3044313379641588, 0.02674518353635602, 0.9266801808276479])
camera_rotation_euler = np.array([-1.9760258764979506, 0.003413526105369158, -1.5848662024427904])

R = cal_camera_vec.euler_to_rotation_matrix(camera_rotation_euler[0], camera_rotation_euler[1], camera_rotation_euler[2])

left = np.dot(R, np.array([0, 0.5, 0])) + camera_position
right = np.dot(R, np.array([0, -0.5, 0]))

i = 1