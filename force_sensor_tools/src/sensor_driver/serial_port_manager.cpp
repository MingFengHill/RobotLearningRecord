#include "serial_port_manager.h"
#include <string>
#include <ros/ros.h>

RETURN_CODE SerialPortManager::init()
{
    try
    {
        //设置串口属性，并打开串口
        serialInst_.setPort("/dev/ttyUSB0");
        serialInst_.setBaudrate(115200);
        serial::Timeout to = serial::Timeout::simpleTimeout(1000); //超时等待
        serialInst_.setTimeout(to);
        serialInst_.open(); //串口开启
    }
    catch (serial::IOException& e)
    {
        ROS_ERROR("Unable to open port ");
        return ERROR;
    }

    //检测串口是否已经打开，并给出提示信息
    if (serialInst_.isOpen()) {
        ROS_INFO_STREAM("Serial Port initialized");
    } else {
        return ERROR;
    }

    return SUCCESS;
}

RETURN_CODE SerialPortManager::uninit()
{
    return SUCCESS;
}

RETURN_CODE SerialPortManager::getCurValue(SensorValue& sv)
{
    std::unique_lock<std::mutex> lockGuard(mtx_);
    querySensor();
    float FX,FY,FZ,MX,MY,MZ;
    if (serialInst_.available()) {
        std::string result = serialInst_.read(serialInst_.available());//从串口接收原始数据

        //转换为工程单位，每个工程单位数据占四个字节
        char chrFX[4]={result[6],result[7],result[8],result[9]};
        char chrFY[4]={result[10],result[11],result[12],result[13]};
        char chrFZ[4]={result[14],result[15],result[16],result[17]};
        char chrMX[4]={result[18],result[19],result[20],result[21]};
        char chrMY[4]={result[22],result[23],result[24],result[25]};
        char chrMZ[4]={result[26],result[27],result[28],result[29]};

        memcpy(&FX ,chrFX ,4);
        memcpy(&FY ,chrFY ,4);
        memcpy(&FZ ,chrFZ ,4);
        memcpy(&MX ,chrMX ,4);
        memcpy(&MY ,chrMY ,4);
        memcpy(&MZ ,chrMZ ,4);

        //赋值给发布的消息
        sv.forceX = FX;
        sv.forceY = FY;
        sv.forceZ = FZ;
        sv.momentX = MX;
        sv.momentY = MY;
        sv.momentZ = MZ;

        // ROS_INFO("get force sensor value:\nforce_x: %f, force_y: %f, force_z: %f,\nmoment_x: %f, moment_y: %f, moment_z: %f", FX, FY, FZ, MX, MY, MZ);
    } else {
        ROS_WARN("serialport inavailble");
    }
    return SUCCESS;
}

RETURN_CODE SerialPortManager::querySensor()
{
    char data[8] = {'A','T','+','G','O','D','\r','\n'};
    std::string input = data;
    serialInst_.write(input);
    return SUCCESS;
}