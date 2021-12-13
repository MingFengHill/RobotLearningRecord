set(UT_THREAD_POOL_SRC ut_thread_pool.cpp)
add_executable(ThreadPoolTest ${UT_THREAD_POOL_SRC})
target_link_libraries(ThreadPoolTest ThreadPool gtest)

install(TARGETS ThreadPoolTest DESTINATION ${PROJECT_SOURCE_DIR}/output/ut)