#ifndef SENSOR_CALIBRATION_H_
#define SENSOR_CALIBRATION_H_

#include <ros/ros.h>
#include <force_sensor_tools/GetFloatValue.h>
#include <std_srvs/SetBool.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/Wrench.h>
#include <mutex>
#include <vector>

#include "error_code.h"
#include "algorithm_engine.h"
#include "common.h"

class SensorCalibration
{
public:
    SensorCalibration() = default;

    ~SensorCalibration() = default;

    RETURN_CODE init();

    RETURN_CODE start();

private:
    ros::NodeHandle nh_;

    ros::ServiceServer add_position_srv_;

    ros::ServiceServer clear_position_srv_;

    ros::ServiceServer calibration_srv_;
    // 获取当前力传感器数值
    ros::ServiceClient get_float_value_cli_;
    // 获取UR机器人位姿
    ros::Subscriber joint_state_sub_;

    bool addPositionCallback(std_srvs::SetBool::Request &request,
                             std_srvs::SetBool::Response &response);

    bool clearPositionCallback(std_srvs::SetBool::Request &request,
                               std_srvs::SetBool::Response &response);

    bool calibrationCallback(std_srvs::SetBool::Request &request,
                             std_srvs::SetBool::Response &response);

    void getJointStateCallback(const sensor_msgs::JointState& msg);

    RETURN_CODE generateXMLFile(CalibrationResult& result);
    // 标定算法引擎
    AlgorithmEngine engine_;

    std::vector<float> jointState_;

    std::mutex jointStateMtx_;
};

#endif // SENSOR_CALIBRATION_H_
