#ifndef COMMON_H_
#define COMMON_H_

#include <vector>

#define JOINT_NUM 6

// 标定参数配置文件环境变量
const char ForceSensorConfig[] = "FORCE_SENSOR_CONFIG";

// RobWork的worldcell文件地址
const char WorldCellPath[] = "/home/pc/RHCS/ws_ros1/src/scene/demo_robwork_workcell/arvp.visionservo.pbvs.wc.xml";

// 标定中间结果输出文件
const char CalibrationInfo[] = "./calibration_info.txt";

// 传感器读取到的数据，三个方向的受力加三个方向的力矩
struct SensorValue {
    float forceX;
    float forceY;
    float forceZ;
    float momentX;
    float momentY;
    float momentZ;
};

// 每个位置的传感器数据和关节转动数据，用于标定的计算
struct CalibrationData
{
    CalibrationData() 
    {
        jointState.resize(JOINT_NUM);
    }

    SensorValue sensorValue;
    // 关节数据，代表位姿
    std::vector<float> jointState;
};

// 标定的输出结果
struct CalibrationResult
{
    // 手爪的质心
    float centroidX;
    float centroidY;
    float centroidZ;
    // 安装倾角
    float Gcos;      
    float Gsin;
    float G;
    // 传感器零点
    float forceX0;
    float forceY0;
    float forceZ0;
    float momentX0;
    float momentY0;
    float momentZ0;
};


#endif // COMMON_H_
