# CMake generated Testfile for 
# Source directory: /home/tenda/aria2D435i
# Build directory: /home/tenda/aria2D435i/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(orb_system_d435i_rectified_settings_smoke "/home/tenda/aria2D435i/build/ORBSettingsSmoke" "/home/tenda/ORB_SLAM3/Vocabulary/ORBvoc.txt" "/home/tenda/aria2D435i/tests/fixtures/orbslam3_d435i_rectified.yaml")
set_tests_properties(orb_system_d435i_rectified_settings_smoke PROPERTIES  TIMEOUT "120" _BACKTRACE_TRIPLES "/home/tenda/aria2D435i/CMakeLists.txt;155;add_test;/home/tenda/aria2D435i/CMakeLists.txt;0;")
