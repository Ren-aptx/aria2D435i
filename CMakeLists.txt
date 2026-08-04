cmake_minimum_required(VERSION 3.10)
project(aria2d435i LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORB_SLAM3_ROOT "/home/tenda/ORB_SLAM3" CACHE PATH
    "Path to a built ORB-SLAM3 source tree")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 3.1 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2 REQUIRED)

set(ORB_SLAM3_LIBRARY "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so")
set(DBOW2_LIBRARY "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so")
set(G2O_LIBRARY "${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so")

foreach(required_file IN ITEMS
        "${ORB_SLAM3_LIBRARY}" "${DBOW2_LIBRARY}" "${G2O_LIBRARY}")
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR
      "Missing ${required_file}. Build ORB-SLAM3 first, or configure with "
      "-DORB_SLAM3_ROOT=/path/to/a/built/ORB_SLAM3")
  endif()
endforeach()

add_executable(RealSenseStereoInertial cpp/RealSenseStereoInertial.cc)

function(configure_orb_target target)
  target_include_directories(${target} PRIVATE
    "${ORB_SLAM3_ROOT}"
    "${ORB_SLAM3_ROOT}/include"
    "${ORB_SLAM3_ROOT}/include/CameraModels"
    "${ORB_SLAM3_ROOT}/Thirdparty/Sophus"
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
  )
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  # ORB_SLAM3.so depends on DBoW2/g2o by soname. Use an old-style RPATH and
  # retain the direct dependencies so no LD_LIBRARY_PATH is required.
  target_link_options(${target} PRIVATE
    -Wl,--disable-new-dtags
    -Wl,--no-as-needed
  )
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${ORB_SLAM3_ROOT}/lib;${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib;${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib"
    INSTALL_RPATH "${ORB_SLAM3_ROOT}/lib;${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib;${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib"
  )
  target_link_libraries(${target} PRIVATE
    ${ORB_SLAM3_LIBRARY}
    ${DBOW2_LIBRARY}
    ${G2O_LIBRARY}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    realsense2::realsense2
    Eigen3::Eigen
    pthread
  )
endfunction()

configure_orb_target(RealSenseStereoInertial)

include(CTest)
if(BUILD_TESTING)
  add_executable(ORBSettingsSmoke tests/orb_settings_smoke.cc)
  configure_orb_target(ORBSettingsSmoke)
  add_test(
    NAME orb_settings_pinhole_smoke
    COMMAND ORBSettingsSmoke "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/orbslam3_pinhole.yaml"
  )
endif()
