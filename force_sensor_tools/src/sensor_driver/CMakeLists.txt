set(SENSOR_DRIVER_SRC sensor_driver.cpp serial_port_manager.cpp service.cpp tinyxml2.cpp)
# 包内的组件保证命名的简洁优美
add_executable(driver ${SENSOR_DRIVER_SRC})
target_link_libraries(driver ${catkin_LIBRARIES} ${ROBWORK_LIBRARIES})
