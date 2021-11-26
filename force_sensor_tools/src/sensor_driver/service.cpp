#include "sensor_driver.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sensor_driver");
    RETURN_CODE ret;

    SensorDriver sd{};

    ret = sd.init();
    if (ret != SUCCESS) {
        ROS_ERROR("SensorDriver init fail");
        return 1;
    }
    
    ROS_INFO("---- driver init success, start service :) ----");
    return sd.start();
}