#ifndef THREADPOOL_THREAD_POOL_H
#define THREADPOOL_THREAD_POOL_H

#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>
#include <functional>
#include <thread>
#include <atomic>

class ThreadPool
{
public:
    ThreadPool(int threadNum = 0);

    ~ThreadPool();

    template<typename FunctionType>
    void AddTask(FunctionType& task) {
        std::unique_lock<std::mutex> lockGuard(taskMutex_);
        taskQueue_.push(std::function<void()>(task));
        taskConditionVariable_.notify_one();
    }

    // This Function must be public.
    void WorkerFunction();

private:
    std::queue<std::function<void()>> taskQueue_;
    std::vector<std::thread> workers_;
    std::condition_variable taskConditionVariable_;
    std::mutex taskMutex_;
    int threadNum_;
    std::atomic<bool> isRunning_;
};

#endif //THREADPOOL_THREAD_POOL_H
