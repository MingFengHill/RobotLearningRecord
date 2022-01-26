#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pygame
import numpy as np
import open3d as o3d
import ur5
import rospy
import sensor_msgs.point_cloud2 as pc2
from threading import Thread
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField


# https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
def cloud_points_open3d_to_ros(open3d_cloud, frame_id="world"):
    """ open3d点云转ROS格式 """
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    points_array = np.asarray(open3d_cloud.points)

    return pc2.create_cloud(header, fields, points_array)


def numpy_array_to_ros_cloud_points(points_array, frame_id="world"):
    """ numpy数组转ROS点云格式 """
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    return pc2.create_cloud(header, fields, points_array)


class CloudPointPubThread(Thread):
    """ 点云发布子线程 """
    def __init__(self, robot, camera2end):
        Thread.__init__(self)
        self.robot = robot
        self.camera2end = camera2end
        self.TOPIC_NAME = "kinect2/points"
        self.cloud_point_pub = rospy.Publisher(self.TOPIC_NAME, PointCloud2, queue_size=10)

    def run(self):
        rate = rospy.Rate(1)
        image_id = 0
        while not rospy.is_shutdown():
            rate.sleep()
            # 获取点云
            self.robot.move_lock.acquire()
            ret, open3d_cloud = self.robot.get_point_cloud()
            end2base, _, _ = self.robot.get_end2base_matrix()
            self.robot.move_lock.release()
            if ret == ur5.ur5_return_error:
                continue

            # 图像坐标系 --> base坐标系
            points_array = np.asarray(open3d_cloud.points)
            print("points_array raw ndim: {}, points_array shape: {}".format(points_array.ndim, points_array.shape))
            points_array = points_array.T
            points_array = np.row_stack((points_array, [1 for _ in range(points_array.shape[1])]))
            points_array = np.matrix(points_array)
            end2base = np.matrix(end2base)
            points_array = end2base * self.camera2end * points_array
            points_array = np.array(points_array)
            points_array = np.delete(points_array, 3, 0)
            points_array = points_array.T

            # numpy数组 --> ROS点云
            ros_cloud = numpy_array_to_ros_cloud_points(points_array)
            self.cloud_point_pub.publish(ros_cloud)
            print("[INFO] pub cloud point id: {}".format(image_id))
            image_id += 1


class MotionPlanSimulation:
    def __init__(self, display_image=False, enable_dynamic=False):
        pygame.init()
        rospy.init_node('MotionPlanSimulation', anonymous=True)
        self.screen = pygame.display.set_mode((300, 300))
        pygame.display.set_caption("UR5 & Kinect")
        self.robot = ur5.UR5Robot()
        self.step = 5
        self.cur_point_number = 0
        # 是否显示中间过程的图像
        self.__display_image = display_image
        self.__enable_dynamic = enable_dynamic
        # 深度相机手眼标定的结果
        self.__camera2end = np.matrix([[-0.0003, 1.0000, 0.0006, 0.0960],
                                       [-1.0000, -0.0003, 0.0006, -0.0140],
                                       [0.0006, -0.0006, 1.0000, 0.0179],
                                       [0, 0, 0, 1]])
        self.__cur_scene = o3d.geometry.PointCloud()
        self.__cloud_point_pub_thr = CloudPointPubThread(self.robot, self.__camera2end)
        self.__cloud_point_pub_thr.start()
        self.CUR_SCENE_TOPIC = "kinect2/cur_scene"
        self.cur_scene_pub = rospy.Publisher(self.CUR_SCENE_TOPIC, PointCloud2, queue_size=10)

    def __del__(self):
        self.__cloud_point_pub_thr.join()

    def rotate_robot_joint(self, joint_id, angle):
        self.robot.move_lock.acquire()
        if self.__enable_dynamic:
            self.robot.rotate_joint_dynamic(joint_id, angle)
        else:
            self.robot.rotate_joint_no_dynamic(joint_id, angle)
        self.robot.move_lock.release()

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
                        self.rotate_robot_joint(0, self.step)
                    elif event.key == pygame.K_w:
                        self.rotate_robot_joint(0, -self.step)
                    elif event.key == pygame.K_a:
                        self.rotate_robot_joint(1, self.step)
                    elif event.key == pygame.K_s:
                        self.rotate_robot_joint(1, -self.step)
                    elif event.key == pygame.K_z:
                        self.rotate_robot_joint(2, self.step)
                    elif event.key == pygame.K_x:
                        self.rotate_robot_joint(2, -self.step)
                    elif event.key == pygame.K_e:
                        self.rotate_robot_joint(3, self.step)
                    elif event.key == pygame.K_r:
                        self.rotate_robot_joint(3, -self.step)
                    elif event.key == pygame.K_d:
                        self.rotate_robot_joint(4, self.step)
                    elif event.key == pygame.K_f:
                        self.rotate_robot_joint(4, -self.step)
                    elif event.key == pygame.K_c:
                        self.rotate_robot_joint(5, self.step)
                    elif event.key == pygame.K_v:
                        self.rotate_robot_joint(5, -self.step)
                    elif event.key == pygame.K_k:
                        self.simple_scene_reconstruction()
                    else:
                        print("Invalid input, no corresponding function for this key!")

    def simple_scene_reconstruction(self):
        """ 叠加多次拍摄的点云，实现简单的场景重建 """
        ret, point_cloud = self.robot.get_point_cloud()
        if ret == ur5.ur5_return_error:
            return
        # o3d.visualization.draw_geometries([point_cloud])
        # points_array ndim: 2, points_array shape: (307200, 3)
        points_array = np.asarray(point_cloud.points)
        print("points_array raw ndim: {}, points_array shape: {}".format(points_array.ndim, points_array.shape))
        points_array = points_array.T
        points_array = np.row_stack((points_array, [1 for _ in range(points_array.shape[1])]))
        points_array = np.matrix(points_array)
        end2base, _, _ = self.robot.get_end2base_matrix()
        end2base = np.matrix(end2base)
        # 图像坐标系 --> base坐标系
        points_array = end2base * self.__camera2end * points_array
        points_array = np.array(points_array)
        points_array = np.delete(points_array, 3, 0)
        points_array = points_array.T
        cur_scene_array = np.asarray(self.__cur_scene.points)
        # 将图像叠加到已有的场景中
        cur_scene_array = np.row_stack((cur_scene_array, points_array))
        self.__cur_scene.points = o3d.utility.Vector3dVector(cur_scene_array)
        # o3d.visualization.draw_geometries([self.__cur_scene])
        ros_cloud = cloud_points_open3d_to_ros(self.__cur_scene)
        self.cur_scene_pub.publish(ros_cloud)


def main():
    cs = MotionPlanSimulation(False)
    cs.run_loop()


if __name__ == "__main__":
    main()
