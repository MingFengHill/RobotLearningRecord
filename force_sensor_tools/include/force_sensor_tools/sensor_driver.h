#ifndef SENSOR_DIRVER_H_
#define SENSOR_DIRVER_H_

#include <ros/ros.h>
#include <force_sensor_tools/GetFloatValue.h>
#include <geometry_msgs/Wrench.h>
#include <rw/rw.hpp>
#include <sensor_msgs/JointState.h>
#include <mutex>

#include "error_code.h"
#include "serial_port_manager.h"
#include "common.h"

class SensorDriver 
{
public:
    SensorDriver() = default;

    ~SensorDriver() = default;

    RETURN_CODE init();

    RETURN_CODE start();

private:
    RETURN_CODE doCalibration(SensorValue& sv);

    RETURN_CODE parseConfiguration(void);
    // ROS相关
    bool getFloatValueCallback(force_sensor_tools::GetFloatValue::Request &req,
                               force_sensor_tools::GetFloatValue::Response &res);

    void getJointStateCallback(const sensor_msgs::JointState& msg);

    ros::NodeHandle nh_;

    ros::ServiceServer get_float_value_srv_;

    ros::Publisher force_sensor_pub_;

    ros::Publisher force_sensor_pub_calibration_;

    ros::Subscriber joint_state_sub_; /* 获取UR机器人位姿 */
    // 配置参数
    bool uesCalibration_;

    CalibrationResult result_;
    // RobWork相关
    rw::models::WorkCell::Ptr workCell_;

    rw::models::Device::Ptr deviceRobot_;

    rw::kinematics::State state_;

    rw::math::Q robotState_;
    // 关节角度
    std::vector<float> jointState_;

    std::mutex jointStateMtx_;
};
#endif // SENSOR_DIRVER_H_