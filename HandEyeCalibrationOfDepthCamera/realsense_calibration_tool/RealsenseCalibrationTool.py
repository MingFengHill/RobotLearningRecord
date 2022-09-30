import os
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import rospy
import threading
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse
from sensor_msgs.msg import JointState
import cv2
import scipy.optimize as opt


def quaternion_to_rotation_matrix(q):
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


class CalibrationManager:
    def __init__(self):
        self.__align = None
        self.__depth_scale = None
        self.__depth_sensor = None
        self.__profile = None
        self.__device = None
        self.__pipeline_profile = None
        self.__pipeline_wrapper = None
        self.__config = None
        self.__pipeline = None
        self.__joint_state_sub = None
        self.__compute_fk_sp = None
        self._cur_joint_state = None
        self.__joint_state_mtx = None
        self.__image_num = 1
        self.RESOLUTION_X = 640
        self.RESOLUTION_Y = 480
        self.JOINT_STATE_TOPIC = "/joint_states"
        # self.realsense_init()
        # self.ros_init()

    def __del__(self):
        # self.__pipeline.stop()
        # self.__compute_fk_sp.close()
        pass

    def realsense_init(self):
        # Create a pipeline
        self.__pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.__config = rs.config()

        # Get device product line for setting a supporting resolution
        self.__pipeline_wrapper = rs.pipeline_wrapper(self.__pipeline)
        self.__pipeline_profile = self.__config.resolve(self.__pipeline_wrapper)
        self.__device = self.__pipeline_profile.get_device()
        found_rgb = False
        for s in self.__device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("[ERRO] The demo requires Depth camera with Color sensor")
            exit(0)
        self.__config.enable_stream(rs.stream.depth, self.RESOLUTION_X, self.RESOLUTION_Y, rs.format.z16, 30)
        self.__config.enable_stream(rs.stream.color, self.RESOLUTION_X, self.RESOLUTION_Y, rs.format.rgb8, 30)

        # Start streaming
        self.__profile = self.__pipeline.start(self.__config)

        # https://github.com/sriramn1/realsense-open3d/blob/master/d435_open3d.py
        self.__depth_sensor = self.__profile.get_device().first_depth_sensor()
        # Using preset HighAccuracy for recording #3 for High Accuracy
        self.__depth_sensor.set_option(rs.option.visual_preset, 3)
        # depth sensor's depth scale
        self.__depth_scale = self.__depth_sensor.get_depth_scale()
        print("[INFO] depth_scale: {}".format(self.__depth_scale))
        align_to = rs.stream.color
        self.__align = rs.align(align_to)

    def get_joint_states_callback(self, msg):
        self.__joint_state_mtx.acquire()
        self._cur_joint_state = msg
        self.__joint_state_mtx.release()

    def ros_init(self):
        rospy.init_node('RealSenseCalibrationTool', anonymous=True)
        self.__joint_state_mtx = threading.Lock()
        self.__joint_state_sub = rospy.Subscriber(self.JOINT_STATE_TOPIC, JointState, self.get_joint_states_callback)
        self.__compute_fk_sp = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        self.__compute_fk_sp.wait_for_service()

    def create_points_cloud_csdn(self, depth_image, intrinsics):
        """
            covert realsense depth image to od3 points cloud [CSDN]
            https://blog.csdn.net/hongliyu_lvliyu/article/details/121816515
        """
        o3d_depth = o3d.geometry.Image(depth_image)
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height,
            intrinsics.fx, intrinsics.fy,
            intrinsics.ppx, intrinsics.ppy
        )
        points_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth,
            pinhole_camera_intrinsic,
            depth_scale=1.0 / self.__depth_scale
        )
        return points_cloud

    def create_color_points_cloud_csdn(self, depth_image, color_image, intrinsics):
        """
            covert realsense depth image to od3 points cloud [CSDN]
            https://blog.csdn.net/hongliyu_lvliyu/article/details/121816515
        """
        o3d_depth = o3d.geometry.Image(depth_image.copy())
        o3d_color = o3d.geometry.Image(color_image.copy())
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height,
            intrinsics.fx, intrinsics.fy,
            intrinsics.ppx, intrinsics.ppy
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0 / self.__depth_scale,
            # depth_trunc=2,
            convert_rgb_to_intensity=False)
        points_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
        return points_cloud

    def create_points_cloud(self, depth_image, intrinsics):
        # TODO: realsense采集到图像像素x和y和设置的是相反的
        resolution_x = self.RESOLUTION_Y
        resolution_y = self.RESOLUTION_X

        point_array = np.zeros((resolution_x * resolution_y, 3))
        focal_x = intrinsics.fx
        for i in range(resolution_y):
            for j in range(resolution_x):
                point_array[i * resolution_x + j][2] = depth_image[j][i] / 1000
                # TODO: 确定为什么要对x轴取反
                point_array[i * resolution_x + j][0] = -((j - resolution_x / 2) / focal_x) * \
                                                       point_array[i * resolution_x + j][2]
                point_array[i * resolution_x + j][1] = ((i - resolution_y / 2) / focal_x) * \
                                                       point_array[i * resolution_x + j][2]
        points_cloud = o3d.geometry.PointCloud()
        points_cloud.points = o3d.utility.Vector3dVector(point_array)
        return points_cloud

    def add_color_to_point_cloud(self, cloud_point, color_image):
        # TODO: 该函数无法正常使用
        rgb_image = np.array(color_image, dtype=np.uint8)
        rgb_image.resize([self.RESOLUTION_X * self.RESOLUTION_Y, 3])
        color_array = np.zeros((self.RESOLUTION_X * self.RESOLUTION_Y, 3), dtype=np.float64)
        for i in range(self.RESOLUTION_X * self.RESOLUTION_Y):
            color_array[i][0] = rgb_image[i][0] / 256.0
            color_array[i][1] = rgb_image[i][1] / 256.0
            color_array[i][2] = rgb_image[i][2] / 256.0
        cloud_point.colors = o3d.utility.Vector3dVector(color_array)

    def get_points_cloud_from_realsense(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.__pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.__align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("[ERRO] depth frame or color frame is null")
            return

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap_dim = depth_image.shape
        color_colormap_dim = color_image.shape
        profile = frames.get_profile()
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        print("[INFO] depth image from realsense shape: {}".format(depth_colormap_dim))
        print("[INFO] color image from realsense shape: {}".format(color_colormap_dim))
        print("[INFO] intrinsics.width: {}".format(intrinsics.width))
        print("[INFO] intrinsics.height: {}".format(intrinsics.height))
        print("[INFO] intrinsics.fx: {}".format(intrinsics.fx))
        print("[INFO] intrinsics.fy: {}".format(intrinsics.fy))
        print("[INFO] intrinsics.ppx: {}".format(intrinsics.ppx))
        print("[INFO] intrinsics.ppy: {}".format(intrinsics.ppy))

        points_cloud = self.create_color_points_cloud_csdn(depth_image, color_image, intrinsics)
        o3d.visualization.draw_geometries([points_cloud])
        return points_cloud

    def get_cur_robot_pose(self):
        self.__joint_state_mtx.acquire()
        joint_state = self._cur_joint_state
        self.__joint_state_mtx.release()
        if joint_state is None:
            print("[ERROR] joint state is None")
            return
        # print(str(joint_state))
        request = GetPositionFKRequest()
        # request.header.frame_id = 'base_link'
        request.fk_link_names = ["ee_link"]
        request.robot_state.joint_state = joint_state
        try:
            response = self.__compute_fk_sp.call(request)
        except rospy.ServiceException as e:
            print("[ERROR] call compute fk failed")
            return
        # print(str(response))
        print("[INFO] position: {} {} {}, orientation: {} {} {} {}".format(
              response.pose_stamped[0].pose.position.x, response.pose_stamped[0].pose.position.y, response.pose_stamped[0].pose.position.z,
              response.pose_stamped[0].pose.orientation.x, response.pose_stamped[0].pose.orientation.y,
              response.pose_stamped[0].pose.orientation.z, response.pose_stamped[0].pose.orientation.w))
        q = [response.pose_stamped[0].pose.orientation.x, response.pose_stamped[0].pose.orientation.y,
             response.pose_stamped[0].pose.orientation.z, response.pose_stamped[0].pose.orientation.w]
        rotation_matrix = quaternion_to_rotation_matrix(q)
        base2end = ([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], response.pose_stamped[0].pose.position.x],
                     [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], response.pose_stamped[0].pose.position.y],
                     [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], response.pose_stamped[0].pose.position.z]])
        with open("base2end_matrix.txt", "a+", encoding="utf-8") as f:
            for i in range(3):
                for j in range(4):
                    f.write(str(base2end[i][j]) + " ")
            f.write("\n")

    def data_acquisition(self):
        point_cloud = self.get_points_cloud_from_realsense()
        ret, sphere_center = self.find_sphere_center(point_cloud)
        if ret is False:
            print("[ERROR] can not find circle center")
            return
        option = input("请输入选项中的数字：\n1.保存；\n2.不保存。\n输入:")
        if option == 1:
            with open("circle_center.txt", "a+", encoding="utf-8") as f:
                for i in range(len(sphere_center)):
                    f.write("{:.14f}".format(sphere_center[i]) + " ")
                f.write("\n")
            isExist = os.path.exists("./image/")
            if isExist is False:
                os.mkdir(r"./image/")
            image_name = "./image/image_" + str(self.__image_num) + ".pcd"
            o3d.io.write_point_cloud(image_name, point_cloud)
            self.get_cur_robot_pose()
            self.__image_num += 1

    def read_iamge(self, path):
        pcd = o3d.io.read_point_cloud(path)
        o3d.visualization.draw_geometries([pcd])

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
        return sphere_o, sphere_r, es

    def find_sphere_center(self, point_cloud):
        labels = np.array(point_cloud.cluster_dbscan(eps=0.08, min_points=100, print_progress=True))
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

            # 最小二乘法拟合球心
            sphere_center, sphere_radius, es = self.sphere_fit(section_array)
            print("[INFO] sphere fit error: {}".format(es))
            if np.abs(es) > 0.01:
                continue

            print("[INFO] label {} center: {} radius: {}".format(i, sphere_center, sphere_radius))
            if 0.08 > sphere_radius > 0.05:
                # 可视化拟合结果
                mesh_circle = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                mesh_circle.compute_vertex_normals()
                mesh_circle.paint_uniform_color([0.9, 0.1, 0.1])
                mesh_circle = mesh_circle.translate((sphere_center[0], sphere_center[1], sphere_center[2]))
                o3d.visualization.draw_geometries([point_cloud, mesh_circle])
                return True, sphere_center

        return False, []

    def find_sphere_center_from_file(self, path):
        point_cloud = o3d.io.read_point_cloud(path)
        self.find_sphere_center(point_cloud)

    def run_loop(self):
        while True:
            option = input("Have fun~ :)\n请输入选项中的数字：\n1.采集数据并存储；\n"
                           "2.查看点云文件；\n3.提取指定点云中的标定球；\n8.退出程序。\n输入:")
            if option == '1':
                self.data_acquisition()
            elif option == '2':
                path = input("请输入文件路径:")
                self.read_iamge(path)
            elif option == '3':
                path = input("请输入文件路径:")
                self.find_sphere_center_from_file(path)
            elif option == '8':
                print("bye bye~")
                break
            else:
                print("[WARR] 不支持输入： {}".format(option))


def main():
    manager = CalibrationManager()
    manager.run_loop()


if __name__ == '__main__':
    main()
