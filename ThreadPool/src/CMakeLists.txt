include_directories(../include)
#aux_source_directory(. DIR_SRC)

set(THREAD_POOL_SRC thread_pool.cpp)
add_library(ThreadPool SHARED ${THREAD_POOL_SRC})
target_link_libraries(ThreadPool pthread)

set(RUN_SRC main.cpp)
add_executable(run ${RUN_SRC})
target_link_libraries(run ThreadPool)

# TODO: use install
install(TARGETS run DESTINATION ${PROJECT_SOURCE_DIR}/output)
install(TARGETS ThreadPool DESTINATION ${PROJECT_SOURCE_DIR}/output/lib)