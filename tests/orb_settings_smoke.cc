#include <System.h>

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: ORBSettingsSmoke vocabulary.txt settings.yaml\n";
        return 2;
    }
    ORB_SLAM3::System slam(
        argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);
    slam.Shutdown();
    return 0;
}
