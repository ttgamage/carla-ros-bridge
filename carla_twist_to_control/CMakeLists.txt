cmake_minimum_required(VERSION 3.5)
project(carla_twist_to_control)

find_package(ros_environment REQUIRED)
set(ROS_VERSION $ENV{ROS_VERSION})

if(${ROS_VERSION} EQUAL 1)

  find_package(catkin REQUIRED COMPONENTS rospy roslaunch)

  catkin_python_setup()

  if(CATKIN_ENABLE_TESTING)
    roslaunch_add_file_check(launch)
  endif()

  catkin_package(CATKIN_DEPENDS rospy)

  catkin_install_python(
    PROGRAMS src/carla_twist_to_control/carla_twist_to_control.py DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION})

  install(DIRECTORY launch/
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

elseif(${ROS_VERSION} EQUAL 2)
  
  find_package(ament_cmake REQUIRED)
  find_package(rclpy REQUIRED)
  ament_export_dependencies(rclpy)
          
  install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME})
                
  ament_package()
                
endif()
