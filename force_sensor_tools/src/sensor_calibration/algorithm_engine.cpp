#include "algorithm_engine.h"
#include <ros/ros.h>
#include <Eigen/Dense>

using rw::loaders::WorkCellLoader;
using rw::models::WorkCell;
using namespace rw::common;
using namespace rw::models;
using namespace rw::math;
using namespace rw::kinematics;

void AlgorithmEngine::addPosition(CalibrationData& cd)
{
    std::unique_lock<std::mutex> lockGuard(datas_mtx_);
    datas_.push_back(cd);
    ROS_INFO("Add point success, point num is: %ld", datas_.size());
}

void AlgorithmEngine::clearPosition(void)
{
    std::unique_lock<std::mutex> lockGuard(datas_mtx_);
    datas_.clear();
    ROS_INFO("clear down, The number of existing points is: %ld", datas_.size());
}

RETURN_CODE AlgorithmEngine::init()
{
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

    return SUCCESS;
}

RETURN_CODE AlgorithmEngine::caculate(CalibrationResult& cr)
{
    // TODO: 锁的粒度可以减小
    std::unique_lock<std::mutex> lockGuard(datas_mtx_);

    int pointNum = datas_.size();
    ROS_INFO("Point num is: %d", pointNum);

    if (pointNum < 6) {
        ROS_ERROR("Insufficient data collected, at least 6 points are required.");
        return ERROR;
    }

    // Step1: 手爪安装倾角 & 零点辨识
    if (caculateAngleAndZeroPoint(cr) != SUCCESS) {
        ROS_ERROR("caculateAngleAndZeroPoint error");
        return ERROR;
    }

    // Step2: 手爪质心辨识
    if (caculateCentroid(cr) != SUCCESS) {
        ROS_ERROR("caculateCentroid error");
        return ERROR;
    }

    // // mock
    // cr.centroidX = 1.0f;
    // cr.centroidY = 2.0f;
    // cr.centroidZ = 3.0f;
    // cr.Gsin = 4.0f;
    // cr.Gcos = 5.0f;
    // cr.G = 6.0f;
    // cr.forceX0 = 7.0f;
    // cr.forceY0 = 8.0f;
    // cr.forceZ0 = 9.0f;
    // cr.momentX0 = 10.0f;
    // cr.momentY0 = 11.0f;
    // cr.momentZ0 = 12.0f;
    return SUCCESS;
}

RETURN_CODE AlgorithmEngine::caculateCentroid(CalibrationResult& cr)
{
    // 先计算矩阵
    int pointNum = datas_.size();
    int an = 3 * pointNum;
    MatrixXf A(an, 6);
    MatrixXf y(an, 1);
    for (int i = 0; i < pointNum; i++) {
        A(i * 3, 0) = 0;
        A(i * 3, 1) = datas_[i].sensorValue.forceZ;
        A(i * 3, 2) = -datas_[i].sensorValue.forceY;
        A(i * 3, 3) = 1;
        A(i * 3, 4) = 0;
        A(i * 3, 5) = 0;

        A(i * 3 + 1, 0) = -datas_[i].sensorValue.forceZ;
        A(i * 3 + 1, 1) = 0;
        A(i * 3 + 1, 2) = datas_[i].sensorValue.forceX;
        A(i * 3 + 1, 3) = 0;
        A(i * 3 + 1, 4) = 1;
        A(i * 3 + 1, 5) = 0;

        A(i * 3 + 2, 0) = datas_[i].sensorValue.forceY;
        A(i * 3 + 2, 1) = -datas_[i].sensorValue.forceX;
        A(i * 3 + 2, 2) = 0;
        A(i * 3 + 2, 3) = 0;
        A(i * 3 + 2, 4) = 0;
        A(i * 3 + 2, 5) = 1;

        y(i * 3, 0) = datas_[i].sensorValue.momentX;
        y(i * 3 + 1, 0) = datas_[i].sensorValue.momentY;
        y(i * 3 + 2, 0) = datas_[i].sensorValue.momentZ;
    }

    // 最小二乘法进行线性回归
    MatrixXf rlt(6, 1);
    if (leastSquareMethod(A, y, rlt) != SUCCESS) {
        return ERROR;
    }

    // 获得质心位置
    cr.centroidX = rlt(0, 0);
    cr.centroidY = rlt(1, 0);
    cr.centroidZ = rlt(2, 0);

    // 计算力矩的零点
    cr.momentX0 = rlt(3, 0) - cr.forceY0 * cr.centroidZ + cr.forceZ0 * cr.centroidY;
    cr.momentY0 = rlt(4, 0) - cr.forceZ0 * cr.centroidX + cr.forceX0 * cr.centroidZ;
    cr.momentZ0 = rlt(5, 0) - cr.forceX0 * cr.centroidY + cr.forceY0 * cr.centroidX;

    return SUCCESS;
}

RETURN_CODE AlgorithmEngine::caculateAngleAndZeroPoint(CalibrationResult& cr)
{
    // 先计算矩阵
    int pointNum = datas_.size();
    int an = 3 * pointNum;
    MatrixXf A(an, 6);
    MatrixXf y(an, 1);
    for (int i = 0; i < pointNum; i++) {
        // RobWork API 获取旋转矩阵
        rw::math::Q q (6, 0, 0, 0, 0, 0, 0);
        for (int j = 0; j < 6; j++) {
            q[j] = datas_[i].jointState[j];
        }
        robotState_ = q;
        deviceRobot_->setQ(robotState_, state_);
        Transform3D<double> base2end = deviceRobot_->baseTend(state_);

        A(i * 3, 0) = -(float)base2end(0, 2); // -r13
        A(i * 3, 1) = -(float)base2end(1, 2); // -r23
        A(i * 3, 2) = 0;
        A(i * 3, 3) = 1;
        A(i * 3, 4) = 0;
        A(i * 3, 5) = 0;

        A(i * 3 + 1, 0) = -(float)base2end(1, 2); // -r23
        A(i * 3 + 1, 1) = (float)base2end(0, 2); // r13
        A(i * 3 + 1, 2) = 0;
        A(i * 3 + 1, 3) = 0;
        A(i * 3 + 1, 4) = 1;
        A(i * 3 + 1, 5) = 0;

        A(i * 3 + 2, 0) = 0;
        A(i * 3 + 2, 1) = 0;
        A(i * 3 + 2, 2) = -(float)base2end(2, 2); // r13
        A(i * 3 + 2, 3) = 0;
        A(i * 3 + 2, 4) = 0;
        A(i * 3 + 2, 5) = 1;

        y(i * 3, 0) = datas_[i].sensorValue.forceX;
        y(i * 3 + 1, 0) = datas_[i].sensorValue.forceY;
        y(i * 3 + 2, 0) = datas_[i].sensorValue.forceZ;
    }

    // 最小二乘法进行线性回归
    MatrixXf rlt(6, 1);
    if (leastSquareMethod(A, y, rlt) != SUCCESS) {
        return ERROR;
    }

    cr.Gcos = rlt(0, 0);
    cr.Gsin = rlt(1, 0);
    cr.G = rlt(2, 0);
    cr.forceX0 = rlt(3, 0);
    cr.forceY0 = rlt(4, 0);
    cr.forceZ0 = rlt(5, 0);

    return SUCCESS;
}

// A * rlt = y
RETURN_CODE AlgorithmEngine::leastSquareMethod(MatrixXf& A, MatrixXf& y, MatrixXf& rlt)
{
    // TODO: 合法性检查
    rlt = (A.transpose() * A).ldlt().solve(A.transpose() * y);
    return SUCCESS;
}

