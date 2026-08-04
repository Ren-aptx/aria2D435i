#include <Settings.h>
#include <System.h>

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ORBSettingsSmoke settings.yaml\n";
        return 2;
    }
    ORB_SLAM3::Settings settings(argv[1], ORB_SLAM3::System::IMU_STEREO);
    std::cout << settings;
    return 0;
}

