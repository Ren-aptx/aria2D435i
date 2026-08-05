cmake_minimum_required(VERSION 3.10)
project(aria2d435i LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORB_SLAM3_ROOT "/home/tenda/ORB_SLAM3" CACHE PATH
    "Path to a built ORB-SLAM3 source tree")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
find_package(Eigen3 3.1 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2 REQUIRED)

set(ORB_SLAM3_LIBRARY "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so")
set(DBOW2_LIBRARY "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so")
set(G2O_LIBRARY "${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so")
set(ORB_SLAM3_VOCABULARY "${ORB_SLAM3_ROOT}/Vocabulary/ORBvoc.txt")

foreach(required_file IN ITEMS
        "${ORB_SLAM3_LIBRARY}" "${DBOW2_LIBRARY}" "${G2O_LIBRARY}")
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR
      "Missing ${required_file}. Build ORB-SLAM3 first, or configure with "
      "-DORB_SLAM3_ROOT=/path/to/a/built/ORB_SLAM3")
  endif()
endforeach()

# ORB-SLAM3's Rectified settings path does not create a second camera object,
# but the upstream Settings constructor leaves all camera pointers
# uninitialised.  Printing the settings can therefore dereference a random
# originalCalib2_ value.  Generate a minimally patched translation unit in the
# build tree so this project remains self-contained and the external checkout
# is not modified.
set(ORB_SETTINGS_SOURCE "${ORB_SLAM3_ROOT}/src/Settings.cc")
if(NOT EXISTS "${ORB_SETTINGS_SOURCE}")
  message(FATAL_ERROR "Missing ${ORB_SETTINGS_SOURCE}")
endif()
file(READ "${ORB_SETTINGS_SOURCE}" ORB_SETTINGS_COMPAT_SOURCE)
if(NOT ORB_SETTINGS_COMPAT_SOURCE MATCHES "calibration1_\\(nullptr\\)")
  set(ORB_SETTINGS_CTOR_OLD
"Settings::Settings(const std::string &configFile, const int& sensor) :
    bNeedToUndistort_(false)")
  set(ORB_SETTINGS_CTOR_NEW
"Settings::Settings(const std::string &configFile, const int& sensor) :
    sensor_(sensor),
    calibration1_(nullptr), calibration2_(nullptr),
    originalCalib1_(nullptr), originalCalib2_(nullptr),
    bNeedToUndistort_(false)")
  string(FIND "${ORB_SETTINGS_COMPAT_SOURCE}"
    "${ORB_SETTINGS_CTOR_OLD}" ORB_SETTINGS_CTOR_POSITION)
  if(ORB_SETTINGS_CTOR_POSITION EQUAL -1)
    message(FATAL_ERROR
      "Could not apply the ORB Settings pointer-initialisation compatibility patch; "
      "the external Settings.cc layout has changed")
  endif()
  string(REPLACE "${ORB_SETTINGS_CTOR_OLD}" "${ORB_SETTINGS_CTOR_NEW}"
    ORB_SETTINGS_COMPAT_SOURCE "${ORB_SETTINGS_COMPAT_SOURCE}")
endif()
if(NOT ORB_SETTINGS_COMPAT_SOURCE MATCHES
    "if\\(settings.originalCalib2_ != nullptr\\)")
  set(ORB_SETTINGS_PRINT_OLD
"            for(size_t i = 0; i < settings.originalCalib2_->size(); i++){
                output << \" \" << settings.originalCalib2_->getParameter(i);
            }")
  set(ORB_SETTINGS_PRINT_NEW
"            if(settings.originalCalib2_ != nullptr){
                for(size_t i = 0; i < settings.originalCalib2_->size(); i++){
                    output << \" \" << settings.originalCalib2_->getParameter(i);
                }
            }")
  string(FIND "${ORB_SETTINGS_COMPAT_SOURCE}"
    "${ORB_SETTINGS_PRINT_OLD}" ORB_SETTINGS_PRINT_POSITION)
  if(ORB_SETTINGS_PRINT_POSITION EQUAL -1)
    message(FATAL_ERROR
      "Could not apply the ORB Settings camera-2 null-check compatibility patch; "
      "the external Settings.cc layout has changed")
  endif()
  string(REPLACE "${ORB_SETTINGS_PRINT_OLD}" "${ORB_SETTINGS_PRINT_NEW}"
    ORB_SETTINGS_COMPAT_SOURCE "${ORB_SETTINGS_COMPAT_SOURCE}")
endif()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ORB_SETTINGS_COMPAT_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/generated/SettingsCompat.cc")
set(ORB_SETTINGS_COMPAT_CURRENT "")
if(EXISTS "${ORB_SETTINGS_COMPAT_FILE}")
  file(READ "${ORB_SETTINGS_COMPAT_FILE}" ORB_SETTINGS_COMPAT_CURRENT)
endif()
if(NOT ORB_SETTINGS_COMPAT_CURRENT STREQUAL ORB_SETTINGS_COMPAT_SOURCE)
  file(WRITE "${ORB_SETTINGS_COMPAT_FILE}" "${ORB_SETTINGS_COMPAT_SOURCE}")
endif()

add_library(ORBSettingsCompat OBJECT "${ORB_SETTINGS_COMPAT_FILE}")
target_include_directories(ORBSettingsCompat PRIVATE
  "${ORB_SLAM3_ROOT}"
  "${ORB_SLAM3_ROOT}/include"
  "${ORB_SLAM3_ROOT}/include/CameraModels"
  "${ORB_SLAM3_ROOT}/Thirdparty/Sophus"
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
)
target_compile_options(ORBSettingsCompat PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(ORBSettingsCompat PRIVATE
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  Eigen3::Eigen
)

add_executable(RealSenseStereoInertial cpp/RealSenseStereoInertial.cc)

function(configure_orb_target target)
  target_sources(${target} PRIVATE $<TARGET_OBJECTS:ORBSettingsCompat>)
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
    # Export the patched Settings symbols so calls originating inside
    # libORB_SLAM3.so resolve to this executable's safe implementation.
    -Wl,--export-dynamic
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
  if(EXISTS "${ORB_SLAM3_VOCABULARY}")
    add_test(
      NAME orb_system_d435i_rectified_settings_smoke
      COMMAND ORBSettingsSmoke
        "${ORB_SLAM3_VOCABULARY}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/orbslam3_d435i_rectified.yaml"
    )
    set_tests_properties(orb_system_d435i_rectified_settings_smoke PROPERTIES TIMEOUT 120)
  else()
    message(WARNING
      "Skipping ORB System settings smoke test: missing ${ORB_SLAM3_VOCABULARY}")
  endif()
endif()
