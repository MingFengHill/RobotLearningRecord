import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2


class CalibrationManager:
    def __init__(self):
        self.__device = None
        self.__pipeline_profile = None
        self.__pipeline_wrapper = None
        self.__config = None
        self.__pipeline = None
        self.RESOLUTION_X = 640
        self.RESOLUTION_Y = 480
        self.realsense_init()

    def __del__(self):
        self.__pipeline.stop()

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
        self.__config.enable_stream(rs.stream.color, self.RESOLUTION_X, self.RESOLUTION_Y, rs.format.bgr8, 30)

        # Start streaming
        self.__pipeline.start(self.__config)

    def depth_image_2_points_cloud_csdn(self, depth_image, intrinsics):
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
            pinhole_camera_intrinsic
        )
        return points_cloud

    def depth_image_2_points_cloud(self, depth_image, intrinsics):
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

    def get_points_cloud_from_realsense(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.__pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
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

        points_cloud = self.depth_image_2_points_cloud(depth_image, intrinsics)
        o3d.visualization.draw_geometries([points_cloud])

    def get_calibration_ball_center(self):
        pass

    def get_joint_state(self):
        pass

    def data_acquisition(self):
        self.get_points_cloud_from_realsense()

    def run_loop(self):
        while True:
            option = input("请输入选项中的数字：1.采集数据并存储；2.退出程序。")
            if option == '1':
                self.data_acquisition()
            elif option == '2':
                print("bye bye~")
                break
            else:
                print("[WARR] 不支持输入： {}".format(option))


def main():
    manager = CalibrationManager()
    manager.run_loop()


if __name__ == '__main__':
    main()
