#ifndef SERIAL_PORT_MANAGER_
#define SERIAL_PORT_MANAGER_

#include <mutex>
#include <serial/serial.h>

#include "error_code.h"
#include "common.h"

// 单例模式，对串口的访问内部加锁，保证线程安全
class SerialPortManager
{
public:
    ~SerialPortManager() = default;

    SerialPortManager(const SerialPortManager& serialPortManager) = delete;

    SerialPortManager& operator=(const SerialPortManager& serialPortManager) = delete;

    // Meyers' Singleton
    // https://stackoverflow.com/questions/449436/singleton-instance-declared-as-static-variable-of-getinstance-method-is-it-thre/449823#449823
    static SerialPortManager& getInstance()
    {
        static SerialPortManager instance;
        return instance;
    }

    RETURN_CODE init();

    RETURN_CODE getCurValue(SensorValue& sv);

    RETURN_CODE uninit();

private:
    SerialPortManager() = default;

    RETURN_CODE querySensor();

    std::mutex mtx_;

    serial::Serial serialInst_;
};

#endif // SERIAL_PORT_MANAGER_