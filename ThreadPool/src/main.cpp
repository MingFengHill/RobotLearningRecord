#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

#include "thread_pool.h"

using namespace std;

atomic<int> g_id;

void SayNum()
{
//    cout << ++g_id << endl;
//    cout << 1 << endl;
}

int main()
{
    g_id = 0;
    ThreadPool threadPool{};
    for (int i = 0; i < 100; i++) {
        threadPool.AddTask(SayNum);
    }

    this_thread::sleep_for(chrono::seconds(1));
    return 0;
}
