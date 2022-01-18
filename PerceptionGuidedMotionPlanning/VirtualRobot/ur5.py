import os
import sys
import math
import time
import random
import string
import pygame
import numpy as np
import open3d as o3d
import vrep
import cv2
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
import scipy.optimize as opt


# 在机器人lua脚本中添加：
# simRemoteApi.start(20001, 1300, false, false)
class UR5Robot:
    def __init__(self):
        self.JOINT_NUM = 6
        self.JOINT_NAME = "UR5_joint"
        self.CAMERA_RGB_NAME = "kinect_rgb"
        self.CAMERA_DEPTH_NAME = "kinect_depth"
        self.__joint_handles = [0 for _ in range(6)]
        self.__current_joint_angle = [0 for _ in range(6)]

        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.__client_id = vrep.simxStart('127.0.0.1', 20001, True, True, 5000, 5)  # Connect to CoppeliaSim
        if self.__client_id == -1:
            print("[ERRO] Can not Connected to remote API server")
            return

        # Now try to retrieve data in a blocking fashion (i.e. a service call):
        res, objs = vrep.simxGetObjects(self.__client_id, vrep.sim_handle_all, vrep.simx_opmode_blocking)
        if res == vrep.simx_return_ok:
            print('[INFO] Number of objects in the scene: ', len(objs))
        else:
            print('[ERRO] Remote API function call returned with, return code: ', res)

        for i in range(self.JOINT_NUM):
            res, returnHandle = vrep.simxGetObjectHandle(self.__client_id, self.JOINT_NAME + str(i + 1),
                                                         vrep.simx_opmode_blocking)
            if res != vrep.simx_return_ok:
                print("[ERRO] get joint handle error, return code: ", res)
            self.__joint_handles[i] = returnHandle
            res, cur_pos = vrep.simxGetJointPosition(self.__client_id, self.__joint_handles[i],
                                                     vrep.simx_opmode_blocking)
            if res != vrep.simx_return_ok:
                print("[ERRO] get joint position error, return code: ", res)
            self.__current_joint_angle[i] = cur_pos

        res, self.__camera_rgb_handle = vrep.simxGetObjectHandle(self.__client_id, self.CAMERA_RGB_NAME,
                                                                 vrep.simx_opmode_blocking)
        if res != vrep.simx_return_ok:
            print("[ERRO] get rgb camera handle error, return code: ", res)

        res, self.__camera_depth_handle = vrep.simxGetObjectHandle(self.__client_id, self.CAMERA_DEPTH_NAME,
                                                                   vrep.simx_opmode_blocking)
        if res != vrep.simx_return_ok:
            print("[ERRO] get depth camera handle error, return code: ", res)

        res, self.__ur5_handle = vrep.simxGetObjectHandle(self.__client_id, "UR5_connection", vrep.simx_opmode_blocking)
        if res != vrep.simx_return_ok:
            print("[ERRO] get ur5 handle error, return code: ", res)

        self.set_streaming_mode()

    def set_streaming_mode(self):
        res, position = vrep.simxGetObjectPosition(self.__client_id, self.__joint_handles[0], self.__ur5_handle,
                                                   vrep.simx_opmode_streaming)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetObjectPosition failed, return code: {}".format(res))
        res, q = vrep.simxGetObjectQuaternion(self.__client_id, self.__joint_handles[0], self.__ur5_handle,
                                              vrep.simx_opmode_streaming)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetObjectQuaternion failed, return code: {}".format(res))
        res, position = vrep.simxGetObjectPosition(self.__client_id, self.__ur5_handle, self.__joint_handles[0],
                                                   vrep.simx_opmode_streaming)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetObjectPosition failed, return code: {}".format(res))
        res, q = vrep.simxGetObjectQuaternion(self.__client_id, self.__ur5_handle, self.__joint_handles[0],
                                              vrep.simx_opmode_streaming)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetObjectQuaternion failed, return code: {}".format(res))

        res, resolution, image_rgb = vrep.simxGetVisionSensorImage(self.__client_id, self.__camera_rgb_handle, 0,
                                                                   vrep.simx_opmode_streaming)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetVisionSensorImage rgb failed, return code: {}".format(res))

        res, resolution, image_depth = vrep.simxGetVisionSensorImage(self.__client_id, self.__camera_depth_handle, 0,
                                                                     vrep.simx_opmode_streaming)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetVisionSensorImage depth failed, return code: {}".format(res))
        res, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.__client_id,
                                                                            self.__camera_depth_handle,
                                                                            vrep.simx_opmode_streaming)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetVisionSensorDepthBuffer failed, return code: {}".format(res))

    def get_base2end_matrix(self):
        res, position = vrep.simxGetObjectPosition(self.__client_id, self.__joint_handles[0], self.__ur5_handle,
                                                   vrep.simx_opmode_buffer)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetObjectPosition failed")
        res, q = vrep.simxGetObjectQuaternion(self.__client_id, self.__joint_handles[0], self.__ur5_handle,
                                              vrep.simx_opmode_buffer)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetObjectQuaternion failed")
        rotation_matrix = self.quaternion_to_rotation_matrix(q)
        base2end = ([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], position[0]],
                     [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], position[1]],
                     [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], position[2]]])
        return base2end, rotation_matrix, position

    def get_end2base_matrix(self):
        res, position = vrep.simxGetObjectPosition(self.__client_id, self.__ur5_handle, self.__joint_handles[0],
                                                   vrep.simx_opmode_buffer)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetObjectPosition failed")
        res, q = vrep.simxGetObjectQuaternion(self.__client_id, self.__ur5_handle, self.__joint_handles[0],
                                              vrep.simx_opmode_buffer)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetObjectQuaternion failed")
        rotation_matrix = self.quaternion_to_rotation_matrix(q)
        end2base = ([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], position[0]],
                     [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], position[1]],
                     [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], position[2]],
                     [0, 0, 0, 1]])
        return end2base, rotation_matrix, position

    def get_rgb_image(self):
        res, resolution, image_rgb = vrep.simxGetVisionSensorImage(self.__client_id, self.__camera_rgb_handle, 0,
                                                                   vrep.simx_opmode_buffer)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetVisionSensorImage rgb failed")
        sensor_image = np.array(image_rgb, dtype=np.uint8)
        sensor_image.resize([resolution[1], resolution[0], 3])
        print("[INFO] rgb image resolution0: ", resolution[0], "resolution1: ", resolution[1])
        sensor_image = cv2.cvtColor(sensor_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('rgb image', sensor_image)
        return sensor_image

    def get_depth_image(self):
        res, resolution, image_depth = vrep.simxGetVisionSensorImage(self.__client_id, self.__camera_depth_handle, 0,
                                                                     vrep.simx_opmode_buffer)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetVisionSensorImage depth failed")
        sensor_image = np.array(image_depth, dtype=np.uint8)
        sensor_image.resize([resolution[1], resolution[0], 3])
        sensor_image = cv2.cvtColor(sensor_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('depth image', sensor_image)
        return sensor_image

    def __del__(self):
        if self.__client_id != -1:
            vrep.simxFinish(self.__client_id)

    def rotate_joint(self, joint_id, angle):
        res = vrep.simxSetJointTargetPosition(self.__client_id, self.__joint_handles[joint_id],
                                              (self.__current_joint_angle[joint_id] - angle) / (180 / math.pi),
                                              vrep.simx_opmode_blocking)
        if res != vrep.simx_return_ok:
            print("[ERRO] get joint position error, return code: ", res)
            return
        self.__current_joint_angle[joint_id] += angle

    def get_point_cloud(self):
        resolution_x = 640
        resolution_y = 480
        perspective_angle = 57
        far = 3.5
        near = 0.01

        res, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.__client_id,
                                                                            self.__camera_depth_handle,
                                                                            vrep.simx_opmode_buffer)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxGetVisionSensorDepthBuffer failed, return code: {}".format(res))
        # depth_buffer dtype: float64
        depth_buffer = np.array(depth_buffer)
        effective_point = 0
        for i in range(resolution_x * resolution_y):
            if (1.0001 > depth_buffer[i] > 0.9999) or (0.0001 > depth_buffer[i] > -0.0001):
                continue
            effective_point += 1
        print("[INFO] The number of valid points is: {}".format(effective_point))
        point_array = np.zeros((effective_point, 3))
        focal_x = (max(resolution_x, resolution_y) / 2) / math.tan(math.radians(perspective_angle) / 2)
        effective_point = 0
        for i in range(resolution_y):
            for j in range(resolution_x):
                if (1.0001 > depth_buffer[i * resolution_x + j] > 0.9999) or \
                        (0.0001 > depth_buffer[i * resolution_x + j] > -0.0001):
                    continue
                point_array[effective_point][2] = near + depth_buffer[i * resolution_x + j] * (far - near)
                # TODO: 确定为什么要对x轴取反
                point_array[effective_point][0] = -((j - resolution_x / 2) / focal_x) * \
                                                       point_array[effective_point][2]
                point_array[effective_point][1] = ((i - resolution_y / 2) / focal_x) * \
                                                       point_array[effective_point][2]
                effective_point += 1
        cloud_point = o3d.geometry.PointCloud()
        cloud_point.points = o3d.utility.Vector3dVector(point_array)
        self.add_color_to_point_cloud(cloud_point)
        return cloud_point

    def add_color_to_point_cloud(self, cloud_point):
        res, resolution, image_rgb = vrep.simxGetVisionSensorImage(self.__client_id, self.__camera_rgb_handle, 0,
                                                                   vrep.simx_opmode_buffer)
        rgb_image = np.array(image_rgb, dtype=np.uint8)
        rgb_image.resize([resolution[0] * resolution[1], 3])
        print("[INFO] rgb image resolution0: ", resolution[0], "resolution1: ", resolution[1])
        resolution_x = 640
        resolution_y = 480
        color_array = np.zeros((resolution_x * resolution_y, 3), dtype=np.float64)
        for i in range(resolution_x * resolution_y):
            color_array[i][0] = rgb_image[i][0] / 256.0
            color_array[i][1] = rgb_image[i][1] / 256.0
            color_array[i][2] = rgb_image[i][2] / 256.0
        cloud_point.colors = o3d.utility.Vector3dVector(color_array)

    def get_joint_angles(self):
        joint_states = []
        for i in range(self.JOINT_NUM):
            _, joint_state = vrep.simxGetJointPosition(self.__client_id, self.__joint_handles[i],
                                                       vrep.simx_opmode_blocking)
            joint_states.append(joint_state)
        return joint_states

    def get_camera2end_matrix(self):
        _, position = vrep.simxGetObjectPosition(self.__client_id, self.__camera_depth_handle, self.__ur5_handle,
                                                 vrep.simx_opmode_blocking)
        _, q = vrep.simxGetObjectQuaternion(self.__client_id, self.__camera_depth_handle, self.__ur5_handle,
                                            vrep.simx_opmode_blocking)
        rotation_matrix = self.quaternion_to_rotation_matrix(q)
        camera2end = ([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], position[0]],
                       [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], position[1]],
                       [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], position[2]]])
        return camera2end

    def quaternion_to_rotation_matrix(self, q):
        """ 四元数转旋转矩阵 """
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        # 检查四元数是否单位化
        if (x ** 2 + y ** 2 + z ** 2 + w ** 2) != 1:
            print("[WARR] Not a unit quaternion: {}".format(x ** 2 + y ** 2 + z ** 2 + w ** 2))
        # 四元数转旋转矩阵
        # https://zhuanlan.zhihu.com/p/45404840
        r11 = 1 - 2 * y * y - 2 * z * z
        r12 = 2 * x * y - 2 * w * z
        r13 = 2 * x * z + 2 * w * y
        r21 = 2 * x * y + 2 * w * z
        r22 = 1 - 2 * x * x - 2 * z * z
        r23 = 2 * y * z - 2 * w * x
        r31 = 2 * x * z - 2 * w * y
        r32 = 2 * y * z + 2 * w * x
        r33 = 1 - 2 * x * x - 2 * y * y
        return [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
