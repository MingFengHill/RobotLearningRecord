#include <stdlib.h>
#include <string.h>

#include "sensor_driver.h"
#include "tinyxml2.h"

using namespace tinyxml2;
using rw::loaders::WorkCellLoader;
using rw::models::WorkCell;
using namespace rw::common;
using namespace rw::models;
using namespace rw::math;
using namespace rw::kinematics;

RETURN_CODE SensorDriver::init()
{
    setlocale(LC_ALL,"");//防止发布中文出现乱码
    parseConfiguration();

    // 对串口进行初始化
    SerialPortManager& spmng = SerialPortManager::getInstance();
    if (spmng.init() != SUCCESS) {
        ROS_ERROR("serial port init error");
        return ERROR;
    }

    // 如果驱动打开标定，开始接收JointState
    if (uesCalibration_) {
        joint_state_sub_ = nh_.subscribe("ur_driver/joint_states", 1000, &SensorDriver::getJointStateCallback, this);
        force_sensor_pub_ = nh_.advertise<geometry_msgs::Wrench>("force_sensor_value", 1000);
        force_sensor_pub_calibration_ = nh_.advertise<geometry_msgs::Wrench>("force_sensor_value_calibration", 1000);
        jointState_.resize(JOINT_NUM);
    }
    get_float_value_srv_ = nh_.advertiseService("get_float_value", &SensorDriver::getFloatValueCallback, this);

    // TODO: 当前为硬编码，修改为配置文件加载
    // RobWork初始化
    std::string file(WorldCellPath);
    workCell_ = WorkCellLoader::Factory::load(file);
    if (workCell_.isNull()) {
        ROS_ERROR("workcell init fail");
        return ERROR;
    }
    state_ = workCell_->getDefaultState();
    deviceRobot_ = workCell_->findDevice ("UR");
    if (deviceRobot_ == nullptr) {
        ROS_ERROR("RobWork found UR robot fail");
        return ERROR;
    }
    robotState_ = deviceRobot_->getQ(state_);
    ROS_INFO("SensorDriver init down.");
    return SUCCESS;
}

void SensorDriver::getJointStateCallback(const sensor_msgs::JointState& msg)
{
    std::unique_lock<std::mutex> lockGuard(jointStateMtx_);
    if(msg.position.size() == JOINT_NUM) {
        for (int i = 0; i < JOINT_NUM; ++i) {
            jointState_[i] = msg.position[i];
        }
    } else {
        ROS_ERROR("Failed to get JointState");
    }
}

RETURN_CODE SensorDriver::parseConfiguration(void)
{
    char* configPath = getenv(ForceSensorConfig);
    if (configPath == NULL) {
        ROS_WARN("Cannot get environment variable FORCE_SENSOR_CONFIG.");
        uesCalibration_ = false;
        return SUCCESS;
    }
    XMLDocument Doc;
	if (Doc.LoadFile(configPath) != XML_SUCCESS) {
        ROS_WARN("load file %s failed.", configPath);
        uesCalibration_ = false;
        return SUCCESS;
    }

    do {
        XMLElement* elementSwitch = Doc.FirstChildElement("UseCalibration");
        const char* text1 = elementSwitch->GetText();
        ROS_INFO("---- UseCalibration in XML: %s ----", text1);
        if (strlen(text1) != 4 && strlen(text1) != 5) {
            break;
        }
        if (strcmp(text1, "true") == 0) {
            uesCalibration_ = true;
            ROS_INFO("---- uesCalibration_ set true ----");
        } else if (strcmp(text1, "false") == 0) {
            uesCalibration_ = false;
            ROS_INFO("---- uesCalibration_ set false ----");
        } else {
            break;
        }
        // centroidX
        XMLElement* elementPara = Doc.FirstChildElement("Parameter");
        XMLElement* elementCentroidX = elementPara->FirstChildElement("centroidX");
        const char* text2 = elementCentroidX->GetText();
        ROS_INFO("---- centroidX in XML: %s ----", text2);
        result_.centroidX = atof(text2);
        ROS_INFO("---- centroidX set %f ----", result_.centroidX);
        // centroidY
        XMLElement* elementCentroidY = elementPara->FirstChildElement("centroidY");
        const char* text3 = elementCentroidY->GetText();
        ROS_INFO("---- centroidY in XML: %s ----", text3);
        result_.centroidY = atof(text3);
        ROS_INFO("---- centroidY set %f ----", result_.centroidY);
        // centroidZ
        XMLElement* elementCentroidZ = elementPara->FirstChildElement("centroidZ");
        const char* text4 = elementCentroidZ->GetText();
        ROS_INFO("---- centroidZ in XML: %s ----", text4);
        result_.centroidZ = atof(text4);
        ROS_INFO("---- centroidZ set %f ----", result_.centroidZ);
        // Gcos
        XMLElement* elementGcos = elementPara->FirstChildElement("Gcos");
        const char* text5 = elementGcos->GetText();
        ROS_INFO("---- Gcos in XML: %s ----", text5);
        result_.Gcos = atof(text5);
        ROS_INFO("---- Gcos set %f ----", result_.Gcos);
        // Gsin
        XMLElement* elementGsin = elementPara->FirstChildElement("Gsin");
        const char* text6 = elementGsin->GetText();
        ROS_INFO("---- Gsin in XML: %s ----", text6);
        result_.Gsin = atof(text6);
        ROS_INFO("---- Gsin set %f ----", result_.Gsin);
        // G
        XMLElement* elementG = elementPara->FirstChildElement("G");
        const char* text7 = elementG->GetText();
        ROS_INFO("---- Gcos in XML: %s ----", text7);
        result_.G = atof(text7);
        ROS_INFO("---- Gcos set %f ----", result_.G);
        // ForceX0
        XMLElement* elementForceX0 = elementPara->FirstChildElement("forceX0");
        const char* text8 = elementForceX0->GetText();
        ROS_INFO("---- forceX0 in XML: %s ----", text8);
        result_.forceX0 = atof(text8);
        ROS_INFO("---- forceX0 set %f ----", result_.forceX0);
        // ForceY0
        XMLElement* elementForceY0 = elementPara->FirstChildElement("forceY0");
        const char* text9 = elementForceY0->GetText();
        ROS_INFO("---- forceY0 in XML: %s ----", text9);
        result_.forceY0 = atof(text9);
        ROS_INFO("---- forceY0 set %f ----", result_.forceY0);
        // ForceZ0
        XMLElement* elementForceZ0 = elementPara->FirstChildElement("forceZ0");
        const char* text10 = elementForceZ0->GetText();
        ROS_INFO("---- forceZ0 in XML: %s ----", text10);
        result_.forceZ0 = atof(text10);
        ROS_INFO("---- forceZ0 set %f ----", result_.forceZ0);
        // momentX0
        XMLElement* elementMomentX0 = elementPara->FirstChildElement("momentX0");
        const char* text11 = elementMomentX0->GetText();
        ROS_INFO("---- momentX0 in XML: %s ----", text11);
        result_.momentX0 = atof(text11);
        ROS_INFO("---- momentX0 set %f ----", result_.momentX0);
        // momentY0
        XMLElement* elementMomentY0 = elementPara->FirstChildElement("momentY0");
        const char* text12 = elementMomentY0->GetText();
        ROS_INFO("---- momentY0 in XML: %s ----", text12);
        result_.momentY0 = atof(text12);
        ROS_INFO("---- momentY0 set %f ----", result_.momentY0);
        // momentZ0
        XMLElement* elementMomentZ0 = elementPara->FirstChildElement("momentZ0");
        const char* text13 = elementMomentZ0->GetText();
        ROS_INFO("---- momentZ0 in XML: %s ----", text13);
        result_.momentZ0 = atof(text13);
        ROS_INFO("---- momentZ0 set %f ----", result_.momentZ0);

        ROS_INFO("Parse file %s success.", configPath);
        return SUCCESS;
    } while (0);

    ROS_WARN("Parse file %s failed.", configPath);
    uesCalibration_ = false;
    return SUCCESS;
}

RETURN_CODE SensorDriver::start()
{
    // TODO: 是否需要阻塞线程？
    ros::Rate loop_rate(1);

    if (uesCalibration_) {
        while (ros::ok()) {
            // 获取当前传感器参数
            RETURN_CODE ret;
            SensorValue sv;
            SerialPortManager& spmng = SerialPortManager::getInstance();
            ret = spmng.getCurValue(sv);
            if (ret != SUCCESS) {
                ROS_ERROR("serial port get value error, error code: %d.", ret);
                ros::spinOnce();
                loop_rate.sleep();
                continue;
            }

            // 发布传感器参数
            geometry_msgs::Wrench wrench;
            wrench.force.x = sv.forceX;
            wrench.force.y = sv.forceY;
            wrench.force.z = sv.forceZ;
            wrench.torque.x = sv.momentX;
            wrench.torque.y = sv.momentY;
            wrench.torque.z = sv.momentZ;
            force_sensor_pub_.publish(wrench);

            ROS_INFO("before calibration:\nforce_x: %f, force_y: %f, force_z: %f,\nmoment_x: %f, moment_y: %f, moment_z: %f",
                        sv.forceX, sv.forceY, sv.forceZ, sv.momentX, sv.momentY, sv.momentZ);
            doCalibration(sv);
            ROS_INFO("after calibration:\nforce_x: %f, force_y: %f, force_z: %f,\nmoment_x: %f, moment_y: %f, moment_z: %f",
                        sv.forceX, sv.forceY, sv.forceZ, sv.momentX, sv.momentY, sv.momentZ);
            geometry_msgs::Wrench wrenchAfter;
            wrenchAfter.force.x = sv.forceX;
            wrenchAfter.force.y = sv.forceY;
            wrenchAfter.force.z = sv.forceZ;
            wrenchAfter.torque.x = sv.momentX;
            wrenchAfter.torque.y = sv.momentY;
            wrenchAfter.torque.z = sv.momentZ;
            force_sensor_pub_calibration_.publish(wrenchAfter);

            ros::spinOnce();
            loop_rate.sleep();
        }
    }
    ros::spin();

    return SUCCESS;
}

bool SensorDriver::getFloatValueCallback(force_sensor_tools::GetFloatValue::Request &req,
                                         force_sensor_tools::GetFloatValue::Response &res)
{
    RETURN_CODE ret;
    SensorValue sv;

    SerialPortManager& spmng = SerialPortManager::getInstance();
    ret = spmng.getCurValue(sv);
    if (ret != SUCCESS) {
        ROS_ERROR("serial port get value error, error code: %d.", ret);
        return false;
    }

    if (uesCalibration_) {
        ROS_INFO("+++++++++++++++++++++++++++++++++++++++");
        ROS_INFO("before calibration:\nforce_x: %f, force_y: %f, force_z: %f,\nmoment_x: %f, moment_y: %f, moment_z: %f",
                 sv.forceX, sv.forceY, sv.forceZ, sv.momentX, sv.momentY, sv.momentZ);
        doCalibration(sv);
        ROS_INFO("after calibration:\nforce_x: %f, force_y: %f, force_z: %f,\nmoment_x: %f, moment_y: %f, moment_z: %f",
                 sv.forceX, sv.forceY, sv.forceZ, sv.momentX, sv.momentY, sv.momentZ);
        ROS_INFO("+++++++++++++++++++++++++++++++++++++++");
    }

    res.force_x = sv.forceX;
    res.force_y = sv.forceY;
    res.force_z = sv.forceZ;
    res.moment_x = sv.momentX;
    res.moment_y = sv.momentY;
    res.moment_z = sv.momentZ;
    ROS_INFO("call getFloatValueCallback success.");
    
    return true;
}

RETURN_CODE SensorDriver::doCalibration(SensorValue& sv)
{
    std::unique_lock<std::mutex> lockGuard(jointStateMtx_);

    // ROS_INFO("get joint state: %f | %f | %f | %f| %f |%f",
    //         jointState_[0], jointState_[1], jointState_[2], jointState_[3], jointState_[4], jointState_[5]);

    // RobWork API 获取旋转矩阵
    rw::math::Q q (6, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < 6; i++) {
        q[i] = jointState_[i];
    }
    ROS_INFO("get joint state: %f | %f | %f | %f| %f |%f",
             jointState_[0], jointState_[1], jointState_[2], jointState_[3], jointState_[4], jointState_[5]);

    robotState_ = q;
    deviceRobot_->setQ(robotState_, state_);
    Transform3D<double> base2end = deviceRobot_->baseTend(state_);

    float Gx = -result_.Gcos * (float)base2end(0, 2) - result_.Gsin * (float)base2end(1, 2);
    float Gy = result_.Gsin * (float)base2end(0, 2) - result_.Gcos * (float)base2end(1, 2);
    float Gz = -result_.G * (float)base2end(2, 2);

    float Fex = sv.forceX - result_.forceX0 - Gx;
    float Fey = sv.forceY - result_.forceY0 - Gy;
    float Fez = sv.forceZ - result_.forceZ0 - Gz;

    float Mgx = Gz * result_.centroidY - Gy * result_.centroidZ;
    float Mgy = Gx * result_.centroidZ - Gz * result_.centroidX;
    float Mgz = Gy * result_.centroidX - Gx * result_.centroidY;

    float Mex = sv.momentX - Mgx - result_.momentX0;
    float Mey = sv.momentY - Mgy - result_.momentY0;
    float Mez = sv.momentZ - Mgz - result_.momentZ0;

    sv.forceX = Fex;
    sv.forceY = Fey;
    sv.forceZ = Fez;
    sv.momentX = Mex;
    sv.momentY = Mey;
    sv.momentZ = Mez;

    return SUCCESS;
}
