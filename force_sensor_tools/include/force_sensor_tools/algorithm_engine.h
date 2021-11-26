#ifndef ALGORITHM_ENGINE_H_
#define ALGORITHM_ENGINE_H_

#include <vector>
#include <string>
#include <mutex>
#include <rw/rw.hpp>
#include <Eigen/Dense>

#include "common.h"
#include "error_code.h"

using Eigen::MatrixXf;
using Eigen::MatrixXd;

class AlgorithmEngine
{
public:
    AlgorithmEngine() = default;

    ~AlgorithmEngine() = default;

    RETURN_CODE init();

    void addPosition(CalibrationData& cd);

    void clearPosition(void);

    RETURN_CODE caculate(CalibrationResult& cr);

private:
    RETURN_CODE caculateCentroid(CalibrationResult& cr);

    RETURN_CODE caculateAngleAndZeroPoint(CalibrationResult& cr);
    // Eigen实现最小二乘法
    RETURN_CODE leastSquareMethod(MatrixXf& A, MatrixXf& y, MatrixXf& rlt);

    std::vector<CalibrationData> datas_;

    std::mutex datas_mtx_;
    // RobWork
    rw::models::WorkCell::Ptr workCell_;

    rw::models::Device::Ptr deviceRobot_;

    rw::kinematics::State state_;

    rw::math::Q robotState_;
    // 用于标定输出计算信息
    int calibration_info_fd_;
};

#endif // ALGORITHM_ENGINE_H_
