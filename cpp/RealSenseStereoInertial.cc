// Native RealSense D435i recorder and ORB-SLAM3 stereo-inertial frontend.
// Intentionally has no ROS/ROS2 dependency.

#include <System.h>

#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace {

std::atomic<bool> g_running(true);

void handle_signal(int) { g_running.store(false); }

struct Options {
    std::string vocabulary;
    std::string output;
    std::string serial = "261622079447";
    int max_frames = 0;
    bool record_stream = true;
    bool viewer = false;
};

void usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " --vocabulary ORBvoc.txt --output SESSION [options]\n"
        << "Options:\n"
        << "  --serial SERIAL       RealSense serial (default: 261622079447)\n"
        << "  --max-frames N        Stop after N frames; 0 means until Ctrl+C\n"
        << "  --no-bag              Do not also save raw/sample.db3 (legacy flag)\n"
        << "  --no-recording        Alias for --no-bag\n"
        << "  --viewer              Enable the ORB-SLAM3 Pangolin viewer\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--vocabulary") {
            options.vocabulary = value(arg);
        } else if (arg == "--output") {
            options.output = value(arg);
        } else if (arg == "--serial") {
            options.serial = value(arg);
        } else if (arg == "--max-frames") {
            options.max_frames = std::stoi(value(arg));
        } else if (arg == "--no-bag" || arg == "--no-recording") {
            options.record_stream = false;
        } else if (arg == "--viewer") {
            options.viewer = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.vocabulary.empty() || options.output.empty()) {
        throw std::invalid_argument("--vocabulary and --output are required");
    }
    if (options.max_frames < 0) {
        throw std::invalid_argument("--max-frames must be non-negative");
    }
    return options;
}

bool is_directory(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

void make_directories(const std::string& path) {
    if (path.empty() || is_directory(path)) {
        return;
    }
    std::string current;
    if (path.front() == '/') {
        current = "/";
    }
    std::stringstream parts(path);
    std::string part;
    while (std::getline(parts, part, '/')) {
        if (part.empty()) {
            continue;
        }
        if (!current.empty() && current.back() != '/') {
            current += '/';
        }
        current += part;
        if (!is_directory(current) && mkdir(current.c_str(), 0755) != 0) {
            throw std::runtime_error("could not create directory: " + current);
        }
    }
}

std::string join(const std::string& left, const std::string& right) {
    return left.empty() || left.back() == '/' ? left + right : left + "/" + right;
}

std::string frame_name(std::size_t index) {
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << index << ".png";
    return name.str();
}

std::int64_t timestamp_ns(const rs2::frame& frame) {
    return static_cast<std::int64_t>(std::llround(frame.get_timestamp() * 1e6));
}

Eigen::Matrix4f extrinsics_matrix(const rs2_extrinsics& extrinsics) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    // librealsense stores the 3x3 rotation in column-major order.
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            transform(row, col) = extrinsics.rotation[col * 3 + row];
        }
        transform(row, 3) = extrinsics.translation[row];
    }
    return transform;
}

void write_matrix_json(std::ostream& out, const Eigen::Matrix4f& transform) {
    out << "[";
    for (int row = 0; row < 4; ++row) {
        if (row) out << ",";
        out << "[";
        for (int col = 0; col < 4; ++col) {
            if (col) out << ",";
            out << transform(row, col);
        }
        out << "]";
    }
    out << "]";
}

void write_intrinsics_json(std::ostream& out, const rs2_intrinsics& intrinsics) {
    out << "{\"width\":" << intrinsics.width
        << ",\"height\":" << intrinsics.height
        << ",\"fx\":" << intrinsics.fx
        << ",\"fy\":" << intrinsics.fy
        << ",\"cx\":" << intrinsics.ppx
        << ",\"cy\":" << intrinsics.ppy
        << ",\"model\":" << static_cast<int>(intrinsics.model)
        << ",\"coeffs\":[";
    for (int i = 0; i < 5; ++i) {
        if (i) out << ",";
        out << intrinsics.coeffs[i];
    }
    out << "]}";
}

void write_calibration(
    const std::string& path,
    const rs2::video_stream_profile& color,
    const rs2::video_stream_profile& depth,
    const rs2::video_stream_profile& left,
    const rs2::video_stream_profile& right,
    const rs2::motion_stream_profile& gyro,
    float depth_scale) {
    const Eigen::Matrix4f color_to_left =
        extrinsics_matrix(color.get_extrinsics_to(left));
    const Eigen::Matrix4f depth_to_color =
        extrinsics_matrix(depth.get_extrinsics_to(color));
    const Eigen::Matrix4f right_to_left =
        extrinsics_matrix(right.get_extrinsics_to(left));
    const Eigen::Matrix4f left_to_imu =
        extrinsics_matrix(left.get_extrinsics_to(gyro));

    std::ofstream out(path);
    if (!out) throw std::runtime_error("could not write " + path);
    out << std::setprecision(10) << "{\n"
        << "  \"timestamp_unit\": \"nanoseconds\",\n"
        << "  \"depth_scale_m\": " << depth_scale << ",\n"
        << "  \"color\": ";
    write_intrinsics_json(out, color.get_intrinsics());
    out << ",\n  \"depth\": ";
    write_intrinsics_json(out, depth.get_intrinsics());
    out << ",\n  \"ir_left\": ";
    write_intrinsics_json(out, left.get_intrinsics());
    out << ",\n  \"ir_right\": ";
    write_intrinsics_json(out, right.get_intrinsics());
    out << ",\n  \"T_color_to_left_ir\": ";
    write_matrix_json(out, color_to_left);
    out << ",\n  \"T_depth_to_color\": ";
    write_matrix_json(out, depth_to_color);
    out << ",\n  \"T_right_ir_to_left_ir\": ";
    write_matrix_json(out, right_to_left);
    out << ",\n  \"T_left_ir_to_imu\": ";
    write_matrix_json(out, left_to_imu);
    out << "\n}\n";
}

void write_orb_settings(
    const std::string& path,
    const rs2::video_stream_profile& left,
    const rs2::video_stream_profile& right,
    const rs2::motion_stream_profile& gyro) {
    const rs2_intrinsics left_intrinsics = left.get_intrinsics();
    const Eigen::Matrix4f right_to_left =
        extrinsics_matrix(right.get_extrinsics_to(left));
    const float baseline = right_to_left.block<3, 1>(0, 3).norm();
    const Eigen::Matrix4f left_to_imu =
        extrinsics_matrix(left.get_extrinsics_to(gyro));

    std::ofstream out(path);
    if (!out) throw std::runtime_error("could not write " + path);
    out << std::setprecision(10)
        << "%YAML:1.0\n---\n"
        << "File.version: \"1.0\"\n"
        // D435i infrared stereo frames are rectified by the device/SDK. Asking
        // ORB-SLAM3 to rectify them again degrades the epipolar geometry.
        << "Camera.type: \"Rectified\"\n"
        << "Camera1.fx: " << left_intrinsics.fx << "\n"
        << "Camera1.fy: " << left_intrinsics.fy << "\n"
        << "Camera1.cx: " << left_intrinsics.ppx << "\n"
        << "Camera1.cy: " << left_intrinsics.ppy << "\n"
        << "Stereo.b: " << baseline << "\n"
        << "Camera.width: " << left_intrinsics.width << "\n"
        << "Camera.height: " << left_intrinsics.height << "\n"
        << "Camera.fps: 30\n"
        << "Camera.RGB: 1\n"
        << "Stereo.ThDepth: 40.0\n"
        << "IMU.T_b_c1: !!opencv-matrix\n"
        << "   rows: 4\n   cols: 4\n   dt: f\n   data: [";
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (row || col) out << ", ";
            out << left_to_imu(row, col);
        }
    }
    out << "]\n"
        << "IMU.InsertKFsWhenLost: 0\n"
        << "IMU.NoiseGyro: 1.0e-3\n"
        << "IMU.NoiseAcc: 1.0e-2\n"
        << "IMU.GyroWalk: 1.0e-6\n"
        << "IMU.AccWalk: 1.0e-4\n"
        << "IMU.Frequency: " << gyro.fps() << ".0\n"
        << "ORBextractor.nFeatures: 1500\n"
        << "ORBextractor.scaleFactor: 1.2\n"
        << "ORBextractor.nLevels: 8\n"
        << "ORBextractor.iniThFAST: 20\n"
        << "ORBextractor.minThFAST: 7\n"
        << "Viewer.KeyFrameSize: 0.05\n"
        << "Viewer.KeyFrameLineWidth: 1.0\n"
        << "Viewer.GraphLineWidth: 0.9\n"
        << "Viewer.PointSize: 2.0\n"
        << "Viewer.CameraSize: 0.08\n"
        << "Viewer.CameraLineWidth: 3.0\n"
        << "Viewer.ViewpointX: 0.0\n"
        << "Viewer.ViewpointY: -0.7\n"
        << "Viewer.ViewpointZ: -3.5\n"
        << "Viewer.ViewpointF: 500.0\n";
}

struct MotionSample {
    double timestamp_s = 0.0;
    rs2_vector value {};
};

struct CaptureBuffers {
    std::mutex mutex;
    std::deque<MotionSample> gyroscope;
    std::deque<MotionSample> accelerometer;
};

rs2_vector interpolate_acceleration(
    const std::deque<MotionSample>& samples, double timestamp_s) {
    if (samples.empty()) return rs2_vector{0.0f, 0.0f, 0.0f};
    const auto after = std::lower_bound(
        samples.begin(), samples.end(), timestamp_s,
        [](const MotionSample& sample, double time) { return sample.timestamp_s < time; });
    if (after == samples.begin()) return after->value;
    if (after == samples.end()) return samples.back().value;
    const MotionSample& right = *after;
    const MotionSample& left = *(after - 1);
    const double span = right.timestamp_s - left.timestamp_s;
    if (span <= 1e-9) return right.value;
    const float alpha = static_cast<float>((timestamp_s - left.timestamp_s) / span);
    return rs2_vector{
        left.value.x + alpha * (right.value.x - left.value.x),
        left.value.y + alpha * (right.value.y - left.value.y),
        left.value.z + alpha * (right.value.z - left.value.z)};
}

std::vector<ORB_SLAM3::IMU::Point> take_imu_until(
    CaptureBuffers& buffers, double previous_image_s, double image_s) {
    std::vector<ORB_SLAM3::IMU::Point> result;
    std::lock_guard<std::mutex> lock(buffers.mutex);
    for (const MotionSample& gyro : buffers.gyroscope) {
        if (gyro.timestamp_s <= previous_image_s || gyro.timestamp_s > image_s) continue;
        const rs2_vector accel = interpolate_acceleration(buffers.accelerometer, gyro.timestamp_s);
        result.emplace_back(
            accel.x, accel.y, accel.z,
            gyro.value.x, gyro.value.y, gyro.value.z,
            gyro.timestamp_s);
    }
    while (!buffers.gyroscope.empty()
           && buffers.gyroscope.front().timestamp_s <= image_s) {
        buffers.gyroscope.pop_front();
    }
    while (buffers.accelerometer.size() > 1
           && buffers.accelerometer[1].timestamp_s <= image_s) {
        buffers.accelerometer.pop_front();
    }
    return result;
}

void set_supported_option(
    const rs2::sensor& sensor, rs2_option option, float value) {
    if (sensor.supports(option) && !sensor.is_option_read_only(option)) {
        try {
            sensor.set_option(option, value);
        } catch (const rs2::error& error) {
            std::cerr << "Warning: could not set " << rs2_option_to_string(option)
                      << ": " << error.what() << '\n';
        }
    }
}

void configure_device(const rs2::device& device) {
    for (const rs2::sensor& sensor : device.query_sensors()) {
        const std::string name = sensor.supports(RS2_CAMERA_INFO_NAME)
            ? sensor.get_info(RS2_CAMERA_INFO_NAME) : "";
        if (name.find("Stereo") != std::string::npos) {
            set_supported_option(sensor, RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.0f);
            // Projected dots damage visual features in the IR pair.
            set_supported_option(sensor, RS2_OPTION_EMITTER_ENABLED, 0.0f);
        } else if (name.find("RGB") != std::string::npos) {
            set_supported_option(sensor, RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.0f);
        } else if (name.find("Motion") != std::string::npos) {
            // ORB-SLAM3 expects raw measurements, not SDK motion correction.
            set_supported_option(sensor, RS2_OPTION_ENABLE_MOTION_CORRECTION, 0.0f);
        }
    }
}

int select_motion_fps(
    const rs2::device& device, rs2_stream stream, int preferred_fps) {
    std::vector<int> supported;
    for (const rs2::sensor& sensor : device.query_sensors()) {
        for (const rs2::stream_profile& profile : sensor.get_stream_profiles()) {
            if (profile.stream_type() == stream
                && profile.format() == RS2_FORMAT_MOTION_XYZ32F) {
                supported.push_back(profile.fps());
            }
        }
    }
    std::sort(supported.begin(), supported.end());
    supported.erase(std::unique(supported.begin(), supported.end()), supported.end());
    if (supported.empty()) {
        throw std::runtime_error(
            std::string("device exposes no MOTION_XYZ32F profile for ")
            + rs2_stream_to_string(stream));
    }
    if (std::find(supported.begin(), supported.end(), preferred_fps) != supported.end()) {
        return preferred_fps;
    }
    const int selected = supported.back();
    std::cerr << "Warning: " << rs2_stream_to_string(stream) << " does not support "
              << preferred_fps << " Hz; using " << selected << " Hz. Supported:";
    for (const int fps : supported) std::cerr << ' ' << fps;
    std::cerr << '\n';
    return selected;
}

cv::Mat frame_mat(const rs2::video_frame& frame, int type) {
    return cv::Mat(
        cv::Size(frame.get_width(), frame.get_height()), type,
        const_cast<void*>(frame.get_data()), cv::Mat::AUTO_STEP);
}

struct SelectedStreams {
    rs2::sensor stereo_sensor;
    rs2::sensor color_sensor;
    rs2::sensor motion_sensor;

    rs2::stream_profile depth;
    rs2::stream_profile left;
    rs2::stream_profile right;
    rs2::stream_profile color;
    rs2::stream_profile accel;
    rs2::stream_profile gyro;

    bool has_stereo = false;
    bool has_color = false;
    bool has_motion = false;
};

bool matches_video_profile(
    const rs2::stream_profile& profile,
    rs2_stream stream,
    int index,
    int width,
    int height,
    rs2_format format,
    int fps) {
    if (profile.stream_type() != stream || profile.stream_index() != index
        || profile.format() != format || profile.fps() != fps) {
        return false;
    }
    const rs2::video_stream_profile video =
        profile.as<rs2::video_stream_profile>();
    return video && video.width() == width && video.height() == height;
}

bool matches_motion_profile(
    const rs2::stream_profile& profile,
    rs2_stream stream,
    int fps) {
    return profile.stream_type() == stream
        && profile.format() == RS2_FORMAT_MOTION_XYZ32F
        && profile.fps() == fps;
}

SelectedStreams select_streams(
    const rs2::device& device, int accel_fps, int gyro_fps) {
    SelectedStreams result;

    for (const rs2::sensor& sensor : device.query_sensors()) {
        rs2::stream_profile local_depth;
        rs2::stream_profile local_left;
        rs2::stream_profile local_right;
        rs2::stream_profile local_color;
        rs2::stream_profile local_accel;
        rs2::stream_profile local_gyro;

        for (const rs2::stream_profile& profile : sensor.get_stream_profiles()) {
            if (matches_video_profile(
                    profile, RS2_STREAM_DEPTH, 0,
                    848, 480, RS2_FORMAT_Z16, 30)) {
                local_depth = profile;
            } else if (matches_video_profile(
                           profile, RS2_STREAM_INFRARED, 1,
                           848, 480, RS2_FORMAT_Y8, 30)) {
                local_left = profile;
            } else if (matches_video_profile(
                           profile, RS2_STREAM_INFRARED, 2,
                           848, 480, RS2_FORMAT_Y8, 30)) {
                local_right = profile;
            } else if (matches_video_profile(
                           profile, RS2_STREAM_COLOR, 0,
                           848, 480, RS2_FORMAT_BGR8, 30)) {
                local_color = profile;
            } else if (matches_motion_profile(
                           profile, RS2_STREAM_ACCEL, accel_fps)) {
                local_accel = profile;
            } else if (matches_motion_profile(
                           profile, RS2_STREAM_GYRO, gyro_fps)) {
                local_gyro = profile;
            }
        }

        if (local_depth && local_left && local_right) {
            result.stereo_sensor = sensor;
            result.depth = local_depth;
            result.left = local_left;
            result.right = local_right;
            result.has_stereo = true;
        }
        if (local_color) {
            result.color_sensor = sensor;
            result.color = local_color;
            result.has_color = true;
        }
        if (local_accel && local_gyro) {
            result.motion_sensor = sensor;
            result.accel = local_accel;
            result.gyro = local_gyro;
            result.has_motion = true;
        }
    }

    if (!result.has_stereo) {
        throw std::runtime_error(
            "could not find Depth + IR1 + IR2 profiles at 848x480@30");
    }
    if (!result.has_color) {
        throw std::runtime_error(
            "could not find Color BGR8 profile at 848x480@30");
    }
    if (!result.has_motion) {
        throw std::runtime_error(
            "could not find the selected Accel/Gyro MOTION_XYZ32F profiles");
    }
    return result;
}

void print_active_profile(const rs2::stream_profile& profile) {
    std::cout << "ACTIVE STREAM: "
              << rs2_stream_to_string(profile.stream_type());
    if (profile.stream_type() == RS2_STREAM_INFRARED) {
        std::cout << " " << profile.stream_index();
    }
    if (const rs2::video_stream_profile video =
            profile.as<rs2::video_stream_profile>()) {
        std::cout << " " << video.width() << "x" << video.height();
    } else {
        std::cout << " " << profile.fps() << " Hz";
    }
    std::cout << " " << rs2_format_to_string(profile.format()) << '\n';
}

void stop_sensor_noexcept(rs2::sensor& sensor, bool& started) {
    if (!started) return;
    try {
        sensor.stop();
    } catch (const std::exception& error) {
        std::cerr << "Warning: sensor.stop() failed: " << error.what() << '\n';
    }
    started = false;
}

void close_sensor_noexcept(rs2::sensor& sensor, bool& opened) {
    if (!opened) return;
    try {
        sensor.close();
    } catch (const std::exception& error) {
        std::cerr << "Warning: sensor.close() failed: " << error.what() << '\n';
    }
    opened = false;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        const std::string raw = join(options.output, "raw");
        const std::vector<std::string> image_directories = {
            "rgb", "depth", "aligned_depth", "ir_left", "ir_right"};
        make_directories(raw);
        for (const std::string& directory : image_directories) {
            make_directories(join(raw, directory));
        }

        rs2::context context;
        rs2::device selected;
        for (const rs2::device& device : context.query_devices()) {
            if (device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)
                && options.serial == device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
                selected = device;
                break;
            }
        }
        if (!selected) {
            throw std::runtime_error("RealSense serial " + options.serial + " was not found");
        }

        configure_device(selected);
        const int accel_fps = select_motion_fps(selected, RS2_STREAM_ACCEL, 200);
        const int gyro_fps = select_motion_fps(selected, RS2_STREAM_GYRO, 200);
        std::cout << "Selected IMU rates: accel=" << accel_fps
                  << " Hz, gyro=" << gyro_fps << " Hz\n";

        // Query the exact profiles directly from the selected device. Video is
        // later owned by a video-only pipeline; the motion module is opened
        // independently and delivers motion_frame objects into motion_queue.
        SelectedStreams streams = select_streams(selected, accel_fps, gyro_fps);
        const auto color_profile = streams.color.as<rs2::video_stream_profile>();
        const auto depth_profile = streams.depth.as<rs2::video_stream_profile>();
        const auto left_profile = streams.left.as<rs2::video_stream_profile>();
        const auto right_profile = streams.right.as<rs2::video_stream_profile>();
        const auto gyro_profile = streams.gyro.as<rs2::motion_stream_profile>();
        const rs2::depth_sensor depth_sensor =
            streams.stereo_sensor.as<rs2::depth_sensor>();
        if (!depth_sensor) {
            throw std::runtime_error("selected stereo sensor is not a depth sensor");
        }

        print_active_profile(streams.depth);
        print_active_profile(streams.left);
        print_active_profile(streams.right);
        print_active_profile(streams.color);
        print_active_profile(streams.gyro);
        print_active_profile(streams.accel);

        write_calibration(
            join(raw, "calibration.json"), color_profile, depth_profile,
            left_profile, right_profile, gyro_profile, depth_sensor.get_depth_scale());
        const std::string orb_settings = join(raw, "orbslam3_runtime.yaml");
        write_orb_settings(orb_settings, left_profile, right_profile, gyro_profile);

        const Eigen::Matrix4f color_to_left = extrinsics_matrix(
            color_profile.get_extrinsics_to(left_profile));

        // Load ORB vocabulary before starting the camera. This prevents several
        // seconds of stale video/IMU data from accumulating during initialization.
        ORB_SLAM3::System slam(
            options.vocabulary, orb_settings, ORB_SLAM3::System::IMU_STEREO,
            options.viewer);

        std::ofstream imu_log(join(raw, "imu.csv"));
        std::ofstream timestamps(join(raw, "timestamps.csv"));
        std::ofstream trajectory_rgb(join(raw, "trajectory_rgb.txt"));
        std::ofstream trajectory_left(join(raw, "trajectory_left_ir.txt"));
        if (!imu_log || !timestamps || !trajectory_rgb || !trajectory_left) {
            throw std::runtime_error("could not create one or more output log files in " + raw);
        }

        imu_log << "timestamp_ns,type,x,y,z\n" << std::setprecision(10);
        timestamps
            << "idx,rgb_timestamp_ns,depth_timestamp_ns,aligned_depth_timestamp_ns,"
               "ir_left_timestamp_ns,ir_right_timestamp_ns,rgb_file,depth_file,"
               "aligned_depth_file,ir_left_file,ir_right_file,tracking_state,pose_valid\n";
        trajectory_rgb << std::setprecision(15);
        trajectory_left << std::setprecision(15);

        CaptureBuffers buffers;

        // Critical separation:
        //   * the pipeline contains VIDEO STREAMS ONLY, so wait_for_frames()
        //     returns a synchronized Depth + IR1 + IR2 + Color frameset;
        //   * the motion sensor writes individual Accel/Gyro motion_frame objects
        //     into a dedicated frame_queue and never participates in video matching.
        rs2::pipeline video_pipeline(context);
        rs2::config video_config;
        video_config.enable_device(options.serial);
        video_config.enable_stream(
            RS2_STREAM_DEPTH, 0, 848, 480, RS2_FORMAT_Z16, 30);
        video_config.enable_stream(
            RS2_STREAM_INFRARED, 1, 848, 480, RS2_FORMAT_Y8, 30);
        video_config.enable_stream(
            RS2_STREAM_INFRARED, 2, 848, 480, RS2_FORMAT_Y8, 30);
        video_config.enable_stream(
            RS2_STREAM_COLOR, 0, 848, 480, RS2_FORMAT_BGR8, 30);
        if (options.record_stream) {
            // The synchronized image streams are saved in this optional bag.
            // IMU is always retained losslessly in raw/imu.csv.
            video_config.enable_record_to_file(join(raw, "sample.db3"));
        }

        rs2::frame_queue motion_queue(1024, false);
        std::mutex worker_error_mutex;
        std::exception_ptr worker_error;
        const auto report_worker_error = [&]() {
            std::lock_guard<std::mutex> lock(worker_error_mutex);
            if (!worker_error) worker_error = std::current_exception();
            g_running.store(false);
        };

        bool video_started = false;
        bool motion_opened = false;
        bool motion_started = false;
        std::thread motion_worker;

        const auto stop_video_noexcept = [&]() {
            if (!video_started) return;
            try {
                video_pipeline.stop();
            } catch (const std::exception& error) {
                std::cerr << "Warning: video_pipeline.stop() failed: "
                          << error.what() << '\n';
            }
            video_started = false;
        };

        const auto cleanup_streaming = [&]() {
            // Stop producers first, then let the motion consumer leave its wait.
            stop_sensor_noexcept(streams.motion_sensor, motion_started);
            stop_video_noexcept();
            g_running.store(false);

            if (motion_worker.joinable()) motion_worker.join();
            close_sensor_noexcept(streams.motion_sensor, motion_opened);
        };

        std::size_t index = 0;
        std::size_t dropped_capture_frames = 0;

        try {
            // Start the video-only pipeline first. Unlike the previous manual
            // rs2::syncer path, pipeline.wait_for_frames() uses the device's
            // matcher topology and returns one coherent set for all four video
            // streams requested above.
            const rs2::pipeline_profile active_video =
                video_pipeline.start(video_config);
            video_started = true;

            std::cout << "VIDEO PIPELINE STREAMS:\n";
            for (const rs2::stream_profile& profile : active_video.get_streams()) {
                print_active_profile(profile);
            }

            streams.motion_sensor.open(
                std::vector<rs2::stream_profile>{streams.accel, streams.gyro});
            motion_opened = true;

            motion_worker = std::thread([&]() {
                try {
                    while (g_running.load()) {
                        rs2::frame frame;
                        if (!motion_queue.try_wait_for_frame(&frame, 100)) continue;
                        const rs2::motion_frame motion = frame.as<rs2::motion_frame>();
                        if (!motion) continue;

                        const rs2_stream stream = motion.get_profile().stream_type();
                        if (stream != RS2_STREAM_GYRO
                            && stream != RS2_STREAM_ACCEL) {
                            continue;
                        }

                        const MotionSample sample{
                            motion.get_timestamp() * 1e-3,
                            motion.get_motion_data()};
                        {
                            std::lock_guard<std::mutex> lock(buffers.mutex);
                            if (stream == RS2_STREAM_GYRO) {
                                buffers.gyroscope.push_back(sample);
                            } else {
                                buffers.accelerometer.push_back(sample);
                            }
                        }

                        imu_log << timestamp_ns(motion) << ','
                                << (stream == RS2_STREAM_GYRO ? "gyro" : "accel")
                                << ',' << sample.value.x << ',' << sample.value.y
                                << ',' << sample.value.z << '\n';
                    }
                } catch (...) {
                    report_worker_error();
                }
            });

            streams.motion_sensor.start(motion_queue);
            motion_started = true;

            rs2::align align_to_color(RS2_STREAM_COLOR);
            double previous_image_s = -std::numeric_limits<double>::infinity();
            bool have_left_frame_number = false;
            unsigned long long previous_left_frame_number = 0;
            std::size_t video_timeouts = 0;
            std::size_t incomplete_sets = 0;
            bool printed_first_frameset = false;

            while (g_running.load()
                   && (options.max_frames == 0
                       || index < static_cast<std::size_t>(options.max_frames))) {
                rs2::frameset frames;
                if (!video_pipeline.try_wait_for_frames(&frames, 500)) {
                    ++video_timeouts;
                    {
                        std::lock_guard<std::mutex> lock(worker_error_mutex);
                        if (worker_error) break;
                    }
                    if (video_timeouts % 10 == 0) {
                        std::cerr << "Waiting for synchronized video frames; "
                                  << "timeouts=" << video_timeouts << '\n';
                    }
                    continue;
                }

                const rs2::video_frame color = frames.get_color_frame();
                const rs2::depth_frame depth = frames.get_depth_frame();
                const rs2::video_frame left = frames.get_infrared_frame(1);
                const rs2::video_frame right = frames.get_infrared_frame(2);
                if (!color || !depth || !left || !right) {
                    ++incomplete_sets;
                    if (incomplete_sets <= 5 || incomplete_sets % 30 == 0) {
                        std::cerr << "Incomplete video frameset: depth="
                                  << static_cast<bool>(depth)
                                  << " ir1=" << static_cast<bool>(left)
                                  << " ir2=" << static_cast<bool>(right)
                                  << " color=" << static_cast<bool>(color)
                                  << " count=" << incomplete_sets << '\n';
                    }
                    continue;
                }

                if (!printed_first_frameset) {
                    std::cout << "First synchronized video frameset received: "
                              << "depth+ir1+ir2+color\n";
                    printed_first_frameset = true;
                }

                const double image_s = left.get_timestamp() * 1e-3;
                if (image_s <= previous_image_s) continue;

                const unsigned long long left_frame_number = left.get_frame_number();
                if (have_left_frame_number
                    && left_frame_number > previous_left_frame_number + 1) {
                    dropped_capture_frames += static_cast<std::size_t>(
                        left_frame_number - previous_left_frame_number - 1);
                }
                previous_left_frame_number = left_frame_number;
                have_left_frame_number = true;

                const std::vector<ORB_SLAM3::IMU::Point> imu =
                    take_imu_until(buffers, previous_image_s, image_s);
                const cv::Mat left_image = frame_mat(left, CV_8UC1);
                const cv::Mat right_image = frame_mat(right, CV_8UC1);
                const Sophus::SE3f world_to_left =
                    slam.TrackStereo(left_image, right_image, image_s, imu);
                const int tracking_state = slam.GetTrackingState();
                const bool pose_valid =
                    tracking_state == 2 && world_to_left.matrix().allFinite();

                const rs2::frameset aligned = align_to_color.process(frames);
                const rs2::depth_frame aligned_depth = aligned.get_depth_frame();
                if (!aligned_depth) continue;

                const std::string name = frame_name(index);
                const bool wrote_rgb = cv::imwrite(
                    join(join(raw, "rgb"), name), frame_mat(color, CV_8UC3));
                const bool wrote_depth = cv::imwrite(
                    join(join(raw, "depth"), name), frame_mat(depth, CV_16UC1));
                const bool wrote_aligned = cv::imwrite(
                    join(join(raw, "aligned_depth"), name),
                    frame_mat(aligned_depth, CV_16UC1));
                const bool wrote_left = cv::imwrite(
                    join(join(raw, "ir_left"), name), left_image);
                const bool wrote_right = cv::imwrite(
                    join(join(raw, "ir_right"), name), right_image);
                if (!wrote_rgb || !wrote_depth || !wrote_aligned
                    || !wrote_left || !wrote_right) {
                    throw std::runtime_error("failed to write image " + name);
                }

                timestamps << index << ',' << timestamp_ns(color) << ','
                           << timestamp_ns(depth) << ',' << timestamp_ns(aligned_depth)
                           << ',' << timestamp_ns(left) << ',' << timestamp_ns(right)
                           << ',' << "rgb/" << name << ",depth/" << name
                           << ",aligned_depth/" << name
                           << ",ir_left/" << name << ",ir_right/" << name << ','
                           << tracking_state << ',' << (pose_valid ? 1 : 0) << '\n';

                if (pose_valid) {
                    const Eigen::Matrix4f left_to_world =
                        world_to_left.inverse().matrix();
                    const Eigen::Matrix4f color_to_world =
                        left_to_world * color_to_left;
                    const auto write_pose = [&image_s](
                        std::ofstream& out, const Eigen::Matrix4f& pose) {
                        const Eigen::Quaternionf quaternion(
                            pose.block<3, 3>(0, 0));
                        out << image_s << ' ' << pose(0, 3) << ' ' << pose(1, 3)
                            << ' ' << pose(2, 3) << ' ' << quaternion.x() << ' '
                            << quaternion.y() << ' ' << quaternion.z() << ' '
                            << quaternion.w() << '\n';
                    };
                    write_pose(trajectory_left, left_to_world);
                    write_pose(trajectory_rgb, color_to_world);
                }

                previous_image_s = image_s;
                ++index;
                if (index % 30 == 0) {
                    std::cout << "Captured " << index << " frames, tracking="
                              << tracking_state << ", imu=" << imu.size()
                              << "\r" << std::flush;
                }
            }

            cleanup_streaming();
        } catch (...) {
            cleanup_streaming();
            slam.Shutdown();
            throw;
        }

        slam.Shutdown();
        timestamps.flush();
        trajectory_rgb.flush();
        trajectory_left.flush();
        imu_log.flush();

        {
            std::lock_guard<std::mutex> lock(worker_error_mutex);
            if (worker_error) std::rethrow_exception(worker_error);
        }

        std::ofstream summary(join(raw, "capture_summary.json"));
        if (!summary) {
            throw std::runtime_error("could not write capture_summary.json");
        }
        summary << "{\n  \"serial\": \"" << options.serial << "\",\n"
                << "  \"frames\": " << index << ",\n"
                << "  \"dropped_capture_frames\": "
                << dropped_capture_frames << "\n}\n";

        std::cout << "\nCapture complete: " << index << " frames in " << raw << '\n';
        return index == 0 ? 2 : 0;
    } catch (const rs2::error& error) {
        std::cerr << "RealSense error in " << error.get_failed_function() << "("
                  << error.get_failed_args() << "): " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
    }
    return 1;
}