import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2


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
