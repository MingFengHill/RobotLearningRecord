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
import matplotlib.pyplot as plt
import scipy.optimize as opt


# 在机器人lua脚本中添加：
# simRemoteApi.start(20001, 1300, false, false)
class UR5Robot:
    def __init__(self):
        self.JOINT_NUM = 6
        self.JOINT_NAME = "UR5_joint"
        self.FORCE_SENSOR_NAME = "Force_sensor"
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

        res, self.__force_sensor_handle = vrep.simxGetObjectHandle(self.__client_id, self.FORCE_SENSOR_NAME,
                                                                   vrep.simx_opmode_blocking)
        if res != vrep.simx_return_ok:
            print("[ERRO] get force sensor handle error, return code: ", res)

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

        res, state, force, torque = vrep.simxReadForceSensor(self.__client_id, self.__force_sensor_handle,
                                                             vrep.simx_opmode_streaming)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxReadForceSensor failed, return code: {}".format(res))

    def get_force_sensor_value(self):
        res, state, force, torque = vrep.simxReadForceSensor(self.__client_id, self.__force_sensor_handle,
                                                             vrep.simx_opmode_buffer)
        if res != vrep.simx_return_ok:
            print("[ERRO] simxReadForceSensor failed, return code: {}".format(res))
        return res, state, force, torque

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

    def get_joint_angles(self):
        joint_states = []
        for i in range(self.JOINT_NUM):
            _, joint_state = vrep.simxGetJointPosition(self.__client_id, self.__joint_handles[i],
                                                       vrep.simx_opmode_blocking)
            joint_states.append(joint_state)
        return joint_states

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

# 存放标定采集的一个点数据
class CalibrationData:
    def __init__(self):
        self.force = []
        self.torque = []
        self.rotation_matrix = []


# 标定的结果
class CalibrationResult:
    def __init__(self):
        self.centroidX = 0
        self.centroidY = 0
        self.centroidZ = 0
        self.momentX0 = 0
        self.momentY0 = 0
        self.momentZ0 = 0
        self.Gcos = 0
        self.Gsin = 0
        self.G = 0
        self.forceX0 = 0
        self.forceY0 = 0
        self.forceZ0 = 0


class CalibrationSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((300, 300))
        pygame.display.set_caption("UR5 & Kinect")
        self.robot = UR5Robot()
        self.step = 5
        self.__datas = []
        self.result = CalibrationResult()
        self.calibration_down = False
        self.FORCE_X_DEVIATION = 10
        self.FORCE_Y_DEVIATION = 20
        self.FORCE_Z_DEVIATION = 30
        self.TORQUE_X_DEVIATION = 40
        self.TORQUE_Y_DEVIATION = 50
        self.TORQUE_Z_DEVIATION = 60

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
                        self.clear_positions()
                    elif event.key == pygame.K_l:
                        self.verify_calibration_result()
                    elif event.key == pygame.K_j:
                        self.do_calibration()
                    else:
                        print("Invalid input, no corresponding function for this key!")

    def add_position(self):
        res, state, force, torque = self.robot.get_force_sensor_value()
        if res != vrep.simx_return_ok:
            return
        if state != 1:
            print("[ERRO] force sensor state error: {}".format(state))
            return
        _, rotation_matrix, _ = self.robot.get_base2end_matrix()
        print("[INFO] force: {}".format(force))
        print("[INFO] torque: {}".format(torque))
        print("[INFO] rotation_matrix: {}".format(rotation_matrix))
        data = CalibrationData()
        data.force = force
        data.torque = torque
        data.force[0] += self.FORCE_X_DEVIATION
        data.force[1] += self.FORCE_Y_DEVIATION
        data.force[2] += self.FORCE_Z_DEVIATION
        data.torque[0] += self.TORQUE_X_DEVIATION
        data.torque[1] += self.TORQUE_Y_DEVIATION
        data.torque[2] += self.TORQUE_Z_DEVIATION
        data.rotation_matrix = rotation_matrix
        self.__datas.append(data)
        print("[INFO] add position success, current data number: {}".format(len(self.__datas)))

    def do_calibration(self):
        if len(self.__datas) < 6:
            print("[ERRO] Insufficient data collected, at least 6 points are required.")

        # Step1: 手爪安装倾角 & 零点辨识
        self.caculate_angle_and_zero_point()

        # Step2: 手爪质心辨识
        self.caculate_centroid()

        with open("calibration_result.txt", "a+", encoding="utf-8") as f:
            f.write("forceX0: {}\n".format(self.result.forceX0))
            f.write("forceY0: {}\n".format(self.result.forceY0))
            f.write("forceZ0: {}\n".format(self.result.forceZ0))
            f.write("momentX0: {}\n".format(self.result.momentX0))
            f.write("momentY0: {}\n".format(self.result.momentY0))
            f.write("momentZ0: {}\n".format(self.result.momentZ0))
            f.write("centroidX: {}\n".format(self.result.centroidX))
            f.write("centroidY: {}\n".format(self.result.centroidY))
            f.write("centroidZ: {}\n".format(self.result.centroidZ))
            f.write("G: {}\n".format(self.result.G))
            f.write("Gsin: {}\n".format(self.result.Gsin))
            f.write("Gcos: {}\n".format(self.result.Gcos))

        self.__datas.clear()
        self.calibration_down = True

    def caculate_angle_and_zero_point(self):
        A = []
        y = []

        for i in range(len(self.__datas)):
            A.append([0, self.__datas[i].force[2], -self.__datas[i].force[1], 1, 0, 0])
            A.append([-self.__datas[i].force[2], 0, self.__datas[i].force[0], 0, 1, 0])
            A.append([self.__datas[i].force[1], self.__datas[i].force[0], 0, 0, 0, 1])
            y.append([self.__datas[i].torque[0]])
            y.append([self.__datas[i].torque[1]])
            y.append([self.__datas[i].torque[2]])
        A = np.mat(A)
        y = np.mat(y)
        # 最小二乘法进行线性回归
        rlt = self.least_square_method(A, y)

        # 获得质心位置
        self.result.centroidX = rlt[0][0]
        self.result.centroidY = rlt[1][0]
        self.result.centroidZ = rlt[2][0]

        # 计算力矩的零点
        self.result.momentX0 = rlt[3][0]
        self.result.momentY0 = rlt[4][0]
        self.result.momentZ0 = rlt[5][0]

    def caculate_centroid(self):
        A = []
        y = []

        for i in range(len(self.__datas)):
            A.append([-self.__datas[i].rotation_matrix[0][2], -self.__datas[i].rotation_matrix[1][2], 0, 1, 0, 0])
            A.append([-self.__datas[i].rotation_matrix[1][2], self.__datas[i].rotation_matrix[0][2], 0, 0, 1, 0])
            A.append([0, 0, -self.__datas[i].rotation_matrix[2][2], 0, 0, 1])
            y.append([self.__datas[i].force[0]])
            y.append([self.__datas[i].force[1]])
            y.append([self.__datas[i].force[2]])
        A = np.mat(A)
        y = np.mat(y)
        # 最小二乘法进行线性回归
        rlt = self.least_square_method(A, y)

        self.result.Gcos = rlt[0][0]
        self.result.Gsin = rlt[1][0]
        self.result.G = rlt[2][0]
        self.result.forceX0 = rlt[3][0]
        self.result.forceY0 = rlt[4][0]
        self.result.forceZ0 = rlt[5][0]

        if self.result.G != 0:
            angle = math.asin(self.result.Gsin / self.result.G)
            print("[INFO] offset Angle of sensor installation: {}".format(angle))
            print("[INFO] sin^2 + cos^2: {}".format(
                (pow(self.result.Gcos, 2) + pow(self.result.Gsin, 2)) / pow(self.result.G, 2)))
        else:
            print("[ERRO] G is zero")

    # A * rlt = y
    def least_square_method(self, A, y):
        return (A.T * A).I * (A.T * y)

    def clear_positions(self):
        self.__datas.clear()
        print("[INFO] clear positions, cur position number: {}".format(len(self.__datas)))

    def verify_calibration_result(self):
        if self.calibration_down == False:
            print("[WARR] please complete calibration first")
            return
        res, state, force, torque = self.robot.get_force_sensor_value()
        if res != vrep.simx_return_ok:
            return
        if state != 1:
            print("[ERRO] force sensor state error: {}".format(state))
            return
        _, rotation_matrix, _ = self.robot.get_base2end_matrix()
        print("[INFO] force before calibration: {}".format(force))
        print("[INFO] torque before calibration: {}".format(torque))
        print('-' * 50)

        Gx = -self.result.Gcos * rotation_matrix[0][2] - self.result.Gsin * rotation_matrix[1][2]
        Gy = self.result.Gsin * rotation_matrix[0][2] - self.result.Gcos * rotation_matrix[1][2]
        Gz = -self.result.G * rotation_matrix[2][2]

        Fex = force[0] - self.result.forceX0 - Gx
        Fey = force[1] - self.result.forceY0 - Gy
        Fez = force[2] - self.result.forceZ0 - Gz
        force_after_calibration = [Fex, Fey, Fez]

        Mgx = Gx * self.result.centroidY - Gy * self.result.centroidZ
        Mgy = Gx * self.result.centroidZ - Gz * self.result.centroidX
        Mgz = Gy * self.result.centroidX - Gx * self.result.centroidY

        Mex = torque[0] - Mgx - self.result.momentX0
        Mey = torque[1] - Mgy - self.result.momentY0
        Mez = torque[2] - Mgz - self.result.momentZ0
        torque_after_calibration = [Mex, Mey, Mez]

        print("[INFO] force after calibration: {}".format(force_after_calibration))
        print("[INFO] torque after calibration: {}".format(torque_after_calibration))
        print('-' * 50)


def main():
    cs = CalibrationSimulation()
    cs.run_loop()


if __name__ == "__main__":
    main()
