#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pygame
import numpy as np
import open3d as o3d
from threading import Thread
import ur5
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


# https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
def convert_cloud_point_open3d_to_ros(open3d_cloud, frame_id="map"):
    """ open3d点云转ROS格式 """
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    point_array = np.asarray(open3d_cloud.points)

    return pc2.create_cloud(header, fields, point_array)


class CloudPointPubThread(Thread):
    """ 点云发布子线程 """
    def __init__(self, robot, camera2end):
        Thread.__init__(self)
        self.robot = robot
        self.camera2end = camera2end
        self.topic_name = "kinect2/points"
        self.cloud_point_pub = rospy.Publisher(self.topic_name, PointCloud2, queue_size=10)

    def run(self):
        rate = rospy.Rate(1)
        image_id = 0
        while not rospy.is_shutdown():
            open3d_cloud = self.robot.get_point_cloud()
            ros_cloud = convert_cloud_point_open3d_to_ros(open3d_cloud)
            self.cloud_point_pub.publish(ros_cloud)
            print("[INFO] pub cloud point id: {}".format(image_id))
            image_id += 1
            rate.sleep()


class MotionPlanSimulation:
    def __init__(self, display_image=False):
        pygame.init()
        rospy.init_node('MotionPlanSimulation', anonymous=True)
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
        self.__cloud_point_pub_thr = CloudPointPubThread(self.robot, self.__camera2end)
        # self.__cloud_point_pub_thr.start()

    def __del__(self):
        self.__cloud_point_pub_thr.join()

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
                        self.__cloud_point_pub_thr.start()
                    else:
                        print("Invalid input, no corresponding function for this key!")

    def simple_scene_reconstruction(self):
        """ 叠加多次拍摄的点云，实现简单的场景重建 """
        point_cloud = self.robot.get_point_cloud()
        # point_array ndim: 2, point_array shape: (307200, 3)
        point_array = np.asarray(point_cloud.points)
        print("point_array raw ndim: {}, point_array shape: {}".format(point_array.ndim, point_array.shape))
        point_array = point_array.T
        point_array = np.row_stack((point_array, [1 for _ in range(point_array.shape[1])]))
        point_array = np.matrix(point_array)
        end2base, _, _ = self.robot.get_end2base_matrix()
        end2base = np.matrix(end2base)
        # 图像坐标系 --> base坐标系
        point_array = end2base * self.__camera2end * point_array
        point_array = np.array(point_array)
        point_array = np.delete(point_array, 3, 0)
        point_array = point_array.T
        cur_scene_array = np.asarray(self.__cur_scene.points)
        # 将图像叠加到已有的场景中
        cur_scene_array = np.row_stack((cur_scene_array, point_array))
        self.__cur_scene.points = o3d.utility.Vector3dVector(cur_scene_array)
        o3d.visualization.draw_geometries([self.__cur_scene])


def main():
    cs = MotionPlanSimulation(False)
    cs.run_loop()


if __name__ == "__main__":
    main()
