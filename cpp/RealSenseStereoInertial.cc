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
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
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
    bool record_bag = true;
    bool viewer = false;
};

void usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " --vocabulary ORBvoc.txt --output SESSION [options]\n"
        << "Options:\n"
        << "  --serial SERIAL       RealSense serial (default: 261622079447)\n"
        << "  --max-frames N        Stop after N frames; 0 means until Ctrl+C\n"
        << "  --no-bag              Do not also save raw/sample.bag\n"
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
        } else if (arg == "--no-bag") {
            options.record_bag = false;
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
    const rs2_intrinsics right_intrinsics = right.get_intrinsics();
    const Eigen::Matrix4f right_to_left =
        extrinsics_matrix(right.get_extrinsics_to(left));
    const Eigen::Matrix4f left_to_imu =
        extrinsics_matrix(left.get_extrinsics_to(gyro));

    std::ofstream out(path);
    if (!out) throw std::runtime_error("could not write " + path);
    out << std::setprecision(10)
        << "%YAML:1.0\n---\n"
        << "File.version: \"1.0\"\n"
        // ORB-SLAM3's File.version=1.0 Rectified branch leaves originalCalib2_
        // null and then dereferences it in Settings::operator<<. Supplying the
        // complete factory stereo model avoids that upstream crash and lets
        // ORB-SLAM3 compute consistent rectification maps itself.
        << "Camera.type: \"PinHole\"\n"
        << "Camera1.fx: " << left_intrinsics.fx << "\n"
        << "Camera1.fy: " << left_intrinsics.fy << "\n"
        << "Camera1.cx: " << left_intrinsics.ppx << "\n"
        << "Camera1.cy: " << left_intrinsics.ppy << "\n"
        << "Camera2.fx: " << right_intrinsics.fx << "\n"
        << "Camera2.fy: " << right_intrinsics.fy << "\n"
        << "Camera2.cx: " << right_intrinsics.ppx << "\n"
        << "Camera2.cy: " << right_intrinsics.ppy << "\n"
        << "Stereo.T_c1_c2: !!opencv-matrix\n"
        << "   rows: 4\n   cols: 4\n   dt: f\n   data: [";
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (row || col) out << ", ";
            out << right_to_left(row, col);
        }
    }
    out << "]\n"
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
    std::condition_variable ready;
    std::deque<rs2::frameset> frames;
    std::deque<MotionSample> gyroscope;
    std::deque<MotionSample> accelerometer;
    std::size_t dropped_frames = 0;
    double last_enqueued_timestamp_s = -1.0;
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

        rs2::config config;
        config.enable_device(options.serial);
        config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
        config.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);
        config.enable_stream(RS2_STREAM_INFRARED, 1, 848, 480, RS2_FORMAT_Y8, 30);
        config.enable_stream(RS2_STREAM_INFRARED, 2, 848, 480, RS2_FORMAT_Y8, 30);
        config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, accel_fps);
        config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, gyro_fps);
        if (options.record_bag) {
            config.enable_record_to_file(join(raw, "sample.bag"));
        }

        CaptureBuffers buffers;
        std::mutex imu_log_mutex;
        std::ofstream imu_log(join(raw, "imu.csv"));
        imu_log << "timestamp_ns,type,x,y,z\n" << std::setprecision(10);

        rs2::pipeline pipeline;
        const std::size_t max_queued_frames = 8;
        const rs2::pipeline_profile profile = pipeline.start(
            config, [&](const rs2::frame& frame) {
                if (const rs2::motion_frame motion = frame.as<rs2::motion_frame>()) {
                    const MotionSample sample{
                        motion.get_timestamp() * 1e-3, motion.get_motion_data()};
                    const rs2_stream stream = motion.get_profile().stream_type();
                    {
                        std::lock_guard<std::mutex> lock(buffers.mutex);
                        if (stream == RS2_STREAM_GYRO) buffers.gyroscope.push_back(sample);
                        if (stream == RS2_STREAM_ACCEL) buffers.accelerometer.push_back(sample);
                    }
                    {
                        std::lock_guard<std::mutex> lock(imu_log_mutex);
                        imu_log << timestamp_ns(motion) << ','
                                << (stream == RS2_STREAM_GYRO ? "gyro" : "accel") << ','
                                << sample.value.x << ',' << sample.value.y << ','
                                << sample.value.z << '\n';
                    }
                    return;
                }

                const rs2::frameset frames = frame.as<rs2::frameset>();
                if (!frames || !frames.get_infrared_frame(1)
                    || !frames.get_infrared_frame(2) || !frames.get_color_frame()
                    || !frames.get_depth_frame()) {
                    return;
                }
                const double time_s = frames.get_infrared_frame(1).get_timestamp() * 1e-3;
                {
                    std::lock_guard<std::mutex> lock(buffers.mutex);
                    if (std::abs(time_s - buffers.last_enqueued_timestamp_s) < 1e-7) return;
                    buffers.last_enqueued_timestamp_s = time_s;
                    if (buffers.frames.size() >= max_queued_frames) {
                        buffers.frames.pop_front();
                        ++buffers.dropped_frames;
                    }
                    buffers.frames.push_back(frames);
                }
                buffers.ready.notify_one();
            });

        const auto color_profile = profile.get_stream(RS2_STREAM_COLOR)
            .as<rs2::video_stream_profile>();
        const auto depth_profile = profile.get_stream(RS2_STREAM_DEPTH)
            .as<rs2::video_stream_profile>();
        const auto left_profile = profile.get_stream(RS2_STREAM_INFRARED, 1)
            .as<rs2::video_stream_profile>();
        const auto right_profile = profile.get_stream(RS2_STREAM_INFRARED, 2)
            .as<rs2::video_stream_profile>();
        const auto gyro_profile = profile.get_stream(RS2_STREAM_GYRO)
            .as<rs2::motion_stream_profile>();
        const rs2::depth_sensor depth_sensor = profile.get_device().first<rs2::depth_sensor>();

        write_calibration(
            join(raw, "calibration.json"), color_profile, depth_profile,
            left_profile, right_profile, gyro_profile, depth_sensor.get_depth_scale());
        const std::string orb_settings = join(raw, "orbslam3_runtime.yaml");
        write_orb_settings(orb_settings, left_profile, right_profile, gyro_profile);

        const Eigen::Matrix4f color_to_left = extrinsics_matrix(
            color_profile.get_extrinsics_to(left_profile));
        ORB_SLAM3::System slam(
            options.vocabulary, orb_settings, ORB_SLAM3::System::IMU_STEREO,
            options.viewer);

        std::ofstream timestamps(join(raw, "timestamps.csv"));
        timestamps
            << "idx,rgb_timestamp_ns,depth_timestamp_ns,aligned_depth_timestamp_ns,"
               "ir_left_timestamp_ns,ir_right_timestamp_ns,rgb_file,depth_file,"
               "aligned_depth_file,ir_left_file,ir_right_file,tracking_state,pose_valid\n";
        std::ofstream trajectory_rgb(join(raw, "trajectory_rgb.txt"));
        std::ofstream trajectory_left(join(raw, "trajectory_left_ir.txt"));
        trajectory_rgb << std::setprecision(15);
        trajectory_left << std::setprecision(15);

        rs2::align align_to_color(RS2_STREAM_COLOR);
        std::size_t index = 0;
        double previous_image_s = -std::numeric_limits<double>::infinity();
        while (g_running.load()
               && (options.max_frames == 0 || index < static_cast<std::size_t>(options.max_frames))) {
            rs2::frameset frames;
            {
                std::unique_lock<std::mutex> lock(buffers.mutex);
                buffers.ready.wait_for(lock, std::chrono::milliseconds(250), [&] {
                    return !buffers.frames.empty() || !g_running.load();
                });
                if (buffers.frames.empty()) continue;
                frames = buffers.frames.front();
                buffers.frames.pop_front();
            }

            const rs2::video_frame color = frames.get_color_frame();
            const rs2::depth_frame depth = frames.get_depth_frame();
            const rs2::video_frame left = frames.get_infrared_frame(1);
            const rs2::video_frame right = frames.get_infrared_frame(2);
            const double image_s = left.get_timestamp() * 1e-3;
            if (image_s <= previous_image_s) continue;

            const std::vector<ORB_SLAM3::IMU::Point> imu =
                take_imu_until(buffers, previous_image_s, image_s);
            const cv::Mat left_image = frame_mat(left, CV_8UC1);
            const cv::Mat right_image = frame_mat(right, CV_8UC1);
            const Sophus::SE3f world_to_left =
                slam.TrackStereo(left_image, right_image, image_s, imu);
            const int tracking_state = slam.GetTrackingState();
            const bool pose_valid = tracking_state == 2 && world_to_left.matrix().allFinite();

            const rs2::frameset aligned = align_to_color.process(frames);
            const rs2::depth_frame aligned_depth = aligned.get_depth_frame();
            const std::string name = frame_name(index);
            cv::imwrite(join(join(raw, "rgb"), name), frame_mat(color, CV_8UC3));
            cv::imwrite(join(join(raw, "depth"), name), frame_mat(depth, CV_16UC1));
            cv::imwrite(
                join(join(raw, "aligned_depth"), name),
                frame_mat(aligned_depth, CV_16UC1));
            cv::imwrite(join(join(raw, "ir_left"), name), left_image);
            cv::imwrite(join(join(raw, "ir_right"), name), right_image);

            timestamps << index << ',' << timestamp_ns(color) << ',' << timestamp_ns(depth)
                       << ',' << timestamp_ns(aligned_depth) << ',' << timestamp_ns(left)
                       << ',' << timestamp_ns(right) << ','
                       << "rgb/" << name << ",depth/" << name << ",aligned_depth/" << name
                       << ",ir_left/" << name << ",ir_right/" << name << ','
                       << tracking_state << ',' << (pose_valid ? 1 : 0) << '\n';

            if (pose_valid) {
                const Eigen::Matrix4f left_to_world = world_to_left.inverse().matrix();
                const Eigen::Matrix4f color_to_world = left_to_world * color_to_left;
                const auto write_pose = [&](std::ofstream& out, const Eigen::Matrix4f& pose) {
                    const Eigen::Quaternionf quaternion(pose.block<3, 3>(0, 0));
                    out << image_s << ' ' << pose(0, 3) << ' ' << pose(1, 3) << ' '
                        << pose(2, 3) << ' ' << quaternion.x() << ' ' << quaternion.y()
                        << ' ' << quaternion.z() << ' ' << quaternion.w() << '\n';
                };
                // Trajectory time is the left-IR time used by TrackStereo. The converter
                // interpolates this RGB optical-frame pose onto the actual RGB timestamps.
                write_pose(trajectory_left, left_to_world);
                write_pose(trajectory_rgb, color_to_world);
            }

            previous_image_s = image_s;
            ++index;
            if (index % 30 == 0) {
                std::cout << "Captured " << index << " frames, tracking="
                          << tracking_state << "\r" << std::flush;
            }
        }

        g_running.store(false);
        pipeline.stop();
        slam.Shutdown();
        timestamps.flush();
        trajectory_rgb.flush();
        trajectory_left.flush();
        imu_log.flush();

        std::ofstream summary(join(raw, "capture_summary.json"));
        summary << "{\n  \"serial\": \"" << options.serial << "\",\n"
                << "  \"frames\": " << index << ",\n"
                << "  \"dropped_capture_frames\": " << buffers.dropped_frames << "\n}\n";
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
