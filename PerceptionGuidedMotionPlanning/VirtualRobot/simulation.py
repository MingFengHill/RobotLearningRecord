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
import scipy.optimize as opt
from threading import Thread
import ur5


# class CloudPointPubThread(Thread):
#     """ 点云发布子线程 """
#     def __init__(self, robot, camera2end):
#         Thread.__init__(self)
#         self.robot = robot
#         self.camera2end = camera2end
#
#     def run(self):
#         pass


class MotionPlanSimulation:
    def __init__(self, display_image=False):
        pygame.init()
        self.screen = pygame.display.set_mode((300, 300))
        pygame.display.set_caption("UR5 & Kinect")
        self.robot = ur5.UR5Robot()
        self.step = 5
        self.cur_point_number = 0
        # 是否显示中间过程的图像
        self.__display_image = display_image
        # 深度相机手眼标定的结果
        self.__camera2end = np.matrix([[-0.0003, 1.0000, 0.0006, 0.0960],
                                       [-1.0000, -0.0003, 0.0006, -0.0140],
                                       [0.0006, -0.0006, 1.0000, 0.0179],
                                       [0, 0, 0, 1]])
        self.__cur_scene = o3d.geometry.PointCloud()

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
                    elif event.key == pygame.K_k:
                        self.simple_scene_reconstruction()
                    elif event.key == pygame.K_l:
                        self.verify_calibration_result()
                    else:
                        print("Invalid input, no corresponding function for this key!")

    def simple_scene_reconstruction(self):
        point_cloud = self.robot.get_point_cloud()
        # point_array ndim: 2, point_array shape: (307200, 3)
        point_array = np.asarray(point_cloud.points)
        print("point_array raw ndim: {}, point_array shape: {}".format(point_array.ndim, point_array.shape))
        point_array = point_array.T
        # print("point_array transpose ndim: {}, point_array shape: {}".format(point_array.ndim, point_array.shape))
        point_array = np.row_stack((point_array, [1 for _ in range(point_array.shape[1])]))
        # print("point_array row_stack ndim: {}, point_array shape: {}".format(point_array.ndim, point_array.shape))
        point_array = np.matrix(point_array)
        end2base, _, _ = self.robot.get_end2base_matrix()
        end2base = np.matrix(end2base)
        point_array = end2base * self.__camera2end * point_array
        # print("point_array multiplication ndim: {}, point_array shape: {}".format(point_array.ndim, point_array.shape))
        point_array = np.array(point_array)
        point_array = np.delete(point_array, 3, 0)
        # print("point_array delete ndim: {}, point_array shape: {}".format(point_array.ndim, point_array.shape))
        point_array = point_array.T
        # print("point_array transpose ndim: {}, point_array shape: {}".format(point_array.ndim, point_array.shape))
        cur_scene_array = np.asarray(self.__cur_scene.points)
        cur_scene_array = np.row_stack((cur_scene_array, point_array))
        self.__cur_scene.points = o3d.utility.Vector3dVector(cur_scene_array)
        o3d.visualization.draw_geometries([self.__cur_scene])

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
    cs = MotionPlanSimulation(False)
    cs.run_loop()


if __name__ == "__main__":
    main()
