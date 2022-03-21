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

        res, self.__ur5_handle = vrep.simxGetObjectHandle(self.__client_id, "UR5_link7", vrep.simx_opmode_blocking)
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
        point_array = np.zeros((resolution_x * resolution_y, 3))

        focal_x = (max(resolution_x, resolution_y) / 2) / math.tan(math.radians(perspective_angle) / 2)
        for i in range(resolution_y):
            for j in range(resolution_x):
                point_array[i * resolution_x + j][2] = near + depth_buffer[i * resolution_x + j] * (far - near)
                # TODO: 确定为什么要对x轴取反
                point_array[i * resolution_x + j][0] = -((j - resolution_x / 2) / focal_x) * \
                                                       point_array[i * resolution_x + j][2]
                point_array[i * resolution_x + j][1] = ((i - resolution_y / 2) / focal_x) * \
                                                       point_array[i * resolution_x + j][2]

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


class CalibrationSimulation:
    def __init__(self, display_image=False):
        pygame.init()
        self.screen = pygame.display.set_mode((300, 300))
        pygame.display.set_caption("UR5 & Kinect")
        self.robot = UR5Robot()
        self.step = 5
        self.cur_point_number = 0
        # 是否显示中间过程的图像
        self.__display_image = display_image
        self.__camera2end = np.matrix([[-0.0003, 1.0000, 0.0006, 0.0960],
                                       [-1.0000, -0.0003, 0.0006, -0.0140],
                                       [0.0006, -0.0006, 1.0000, 0.0179],
                                       [0, 0, 0, 1]])
        # camera2end = self.robot.get_camera2end_matrix()
        # print("[INFO] camera2end matrix:")
        # for i in range(len(camera2end)):
        #     for j in range(len(camera2end[i])):
        #         print("{:.14f}".format(camera2end[i][j]), end=' ')
        #     print('')

        # 设置界面背景
        # try:
        #     img = pygame.image.load("background.png")
        #     screen.blit(img, (0, 0))
        #     pygame.display.update()
        # except:
        #     print('[ERRO] Unexpected error:', sys.exc_info()[0])

    def run_loop(self):
        while (True):
            key_press = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        sys.exit()
                    elif event.key == pygame.K_q:
                        self.robot.rotate_joint(0, self.step)
                    elif event.key == pygame.K_w:
                        self.robot.rotate_joint(0, -self.step)
                    elif event.key == pygame.K_a:
                        self.robot.rotate_joint(1, self.step)
                    elif event.key == pygame.K_s:
                        self.robot.rotate_joint(1, -self.step)
                    elif event.key == pygame.K_z:
                        self.robot.rotate_joint(2, self.step)
                    elif event.key == pygame.K_x:
                        self.robot.rotate_joint(2, -self.step)
                    elif event.key == pygame.K_e:
                        self.robot.rotate_joint(3, self.step)
                    elif event.key == pygame.K_r:
                        self.robot.rotate_joint(3, -self.step)
                    elif event.key == pygame.K_d:
                        self.robot.rotate_joint(4, self.step)
                    elif event.key == pygame.K_f:
                        self.robot.rotate_joint(4, -self.step)
                    elif event.key == pygame.K_c:
                        self.robot.rotate_joint(5, self.step)
                    elif event.key == pygame.K_v:
                        self.robot.rotate_joint(5, -self.step)
                    elif event.key == pygame.K_SPACE:
                        self.add_position()
                    elif event.key == pygame.K_k:
                        self.verify_rotation_matrix()
                    elif event.key == pygame.K_l:
                        self.verify_calibration_result()
                    else:
                        print("Invalid input, no corresponding function for this key!")

    def verify_rotation_matrix(self):
        """ 验证vrep输出的旋转矩阵是否合理 """
        print("------------- T matrix -------------")
        base2end, r1, p1 = self.robot.get_base2end_matrix()
        base2end.append([0, 0, 0, 1])
        base2end = np.matrix(base2end)
        base2end = np.linalg.inv(base2end)
        print(base2end)
        end2base, r2, p2 = self.robot.get_end2base_matrix()
        end2base = np.matrix(end2base)
        print(end2base)

        print("------------- sphere center -------------")
        center_zero = np.matrix([0, 0, 0, 1])
        center_zero = center_zero.reshape(4, 1)
        print(end2base * center_zero)

        print("------------- rotation matrix -------------")
        print(np.matrix(r1))
        print(np.linalg.inv(np.matrix(r2)))

    def add_position(self):
        self.cur_point_number += 1
        point_cloud = self.robot.get_point_cloud()
        if self.__display_image:
            o3d.visualization.draw_geometries([point_cloud])

        base2end_matrix, _, _ = self.robot.get_base2end_matrix()
        res, circle_center = self.find_circle_center(point_cloud)
        if res:
            with open("circle_center.txt", "a+", encoding="utf-8") as f:
                for i in range(len(circle_center)):
                    f.write("{:.14f}".format(circle_center[i]) + " ")
                f.write("\n")

            with open("base2end_matrix.txt", "a+", encoding="utf-8") as f:
                for i in range(3):
                    for j in range(4):
                        f.write(str(base2end_matrix[i][j]) + " ")
                f.write("\n")
        else:
            print("[WARN] can not get calibration ball center")

    def verify_calibration_result(self):
        """ 验证在base坐标系下球心的坐标是否合理 """
        point_cloud = self.robot.get_point_cloud()
        if self.__display_image:
            o3d.visualization.draw_geometries([point_cloud])
        res, circle_center = self.find_circle_center(point_cloud)
        if not res:
            print("[ERRO] can not get ball center")
            return
        circle_center.append(1)
        circle_center = np.matrix(circle_center)
        # numpy reshape不是对原始的矩阵操作，而是会把修改后的矩阵返回
        circle_center = circle_center.reshape(4, 1)
        end2base, _, _ = self.robot.get_end2base_matrix()
        end2base = np.matrix(end2base)
        circle_center_base = end2base * self.__camera2end * circle_center
        print(circle_center_base)

    def find_circle_center(self, point_cloud):
        labels = np.array(point_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        max_label = labels.max()
        print(f"[INFO] point cloud has {max_label + 1} clusters")
        point_array = np.asarray(point_cloud.points)
        print("[INFO] labels shape:", labels.shape)
        print("[INFO] point_array shape:", point_array.shape)

        for i in range(max_label + 1):
            mask = labels == i
            section_array = point_array[mask]
            cloud_point_section = o3d.geometry.PointCloud()
            cloud_point_section.points = o3d.utility.Vector3dVector(section_array)
            if self.__display_image:
                o3d.visualization.draw_geometries([cloud_point_section])

            # sph = pyrsc.Sphere()
            # center, radius, inliers = sph.fit(section_array, thresh=0.4)
            # print("[INFO] label {} center: {} radius: {} inliers: {}".format(i, center, radius, inliers))

            # 最小二乘法拟合球心
            sphere_center, sphere_radius = self.sphere_fit(section_array)
            print("[INFO] label {} center: {} radius: {}".format(i, sphere_center, sphere_radius))

            if sphere_radius < 0.06:
                # 可视化拟合结果
                mesh_circle = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                mesh_circle.compute_vertex_normals()
                mesh_circle.paint_uniform_color([0.9, 0.1, 0.1])
                mesh_circle = mesh_circle.translate((sphere_center[0], sphere_center[1], sphere_center[2]))
                o3d.visualization.draw_geometries([cloud_point_section, mesh_circle])
                return True, sphere_center

        return False, []
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([point_cloud])

    def spherrors(self, para, points):
        """球面拟合误差"""
        a, b, c, r = para
        x = points[0, :]
        y = points[1, :]
        z = points[2, :]
        return pow((x - a), 2) + pow((y - b), 2) + pow((z - c), 2) - pow(r, 2)

    def sphere_fit(self, point):
        """线性最小二乘拟合"""
        tparas = opt.leastsq(self.spherrors, [1, 1, 1, 0.06], point.T, full_output=1)
        paras = tparas[0]
        sphere_r = abs(paras[3])
        sphere_o = [paras[0], paras[1], paras[2]]
        # 计算球度误差
        es = np.mean(np.abs(tparas[2]['fvec'])) / paras[3]  # 'fvec'即为spherrors的值
        return sphere_o, sphere_r


def main():
    cs = CalibrationSimulation(False)
    cs.run_loop()


if __name__ == "__main__":
    main()
