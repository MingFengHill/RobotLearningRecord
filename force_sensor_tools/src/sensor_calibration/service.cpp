#include "sensor_calibration.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sensor_calibration");
    RETURN_CODE ret;

    SensorCalibration sc{};

    ret = sc.init();
    if (ret != SUCCESS) {
        ROS_ERROR("sensor_calibration init fail, return code: %d.", ret);
        return 1;
    }
    
    ROS_INFO("---- sensor calibration tool init success, start service :) ----");
    return sc.start();
}