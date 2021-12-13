#include "thread_pool.h"
#include <iostream>

ThreadPool::ThreadPool(int threadNum) : threadNum_(threadNum)
{
    if (threadNum_ == 0) {
        unsigned const coreNum = std::thread::hardware_concurrency();
        std::cout << "[INFO] Your computer has " << coreNum << " CPU cores." << std::endl;
    }
    std::unique_lock<std::mutex> lockGuard(taskMutex_);
    isRunning_ = true;
    for (int i = 0; i < threadNum_; i++) {
        workers_.emplace_back(&ThreadPool::WorkerFunction, this);
    }
}

ThreadPool::~ThreadPool()
{
    std::unique_lock<std::mutex> lockGuard(taskMutex_);
    isRunning_ = false;
    taskConditionVariable_.notify_all();
    lockGuard.unlock();
    for (auto& worker : workers_) {
        worker.join();
    }
    std::cout << "[INFO] Number of outstanding tasks: " << taskQueue_.size() << std::endl;
}

void ThreadPool::WorkerFunction()
{
    std::unique_lock<std::mutex> lockGuard(taskMutex_);
    while (isRunning_) {
        if (taskQueue_.empty()) {
            taskConditionVariable_.wait(lockGuard);
        }
        if (taskQueue_.empty()) {
            continue;
        }
        auto task = taskQueue_.front();
        taskQueue_.pop();
        lockGuard.unlock();
        task();
        lockGuard.lock();
    }
}
