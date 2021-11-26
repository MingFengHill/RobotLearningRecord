set(SENSOR_CALIBRATION_SRC algorithm_engine.cpp sensor_calibration.cpp service.cpp tinyxml2.cpp)
# 包内的组件保证命名的简洁优美
add_executable(calibration ${SENSOR_CALIBRATION_SRC})
target_link_libraries(calibration ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${ROBWORK_LIBRARIES})
