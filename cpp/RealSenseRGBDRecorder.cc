// Fixed-camera RealSense D435i RGB-D recorder.
// Captures only color, raw depth, and depth aligned to color. No ROS, IR, IMU,
// ORB-SLAM3, Pangolin, or trajectory generation is involved.

#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace {

std::atomic<bool> g_running(true);

void handle_signal(int) { g_running.store(false); }

struct Options {
    std::string output;
    std::string serial = "261622079447";
    int max_frames = 0;
    int warmup_frames = 60;
    bool record_stream = false;
};

void usage(const char* program) {
    std::cerr
        << "Usage: " << program << " --output SESSION [options]\n"
        << "Options:\n"
        << "  --serial SERIAL       RealSense serial (default: 261622079447)\n"
        << "  --max-frames N        Stop after N frames; 0 means until Ctrl+C\n"
        << "  --warmup-frames N     Discard N startup frames (default: 60)\n"
        << "  --record              Also save raw/sample.db3\n";
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
        if (arg == "--output") {
            options.output = value(arg);
        } else if (arg == "--serial") {
            options.serial = value(arg);
        } else if (arg == "--max-frames") {
            options.max_frames = std::stoi(value(arg));
        } else if (arg == "--warmup-frames") {
            options.warmup_frames = std::stoi(value(arg));
        } else if (arg == "--record") {
            options.record_stream = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.output.empty()) {
        throw std::invalid_argument("--output is required");
    }
    if (options.max_frames < 0 || options.warmup_frames < 0) {
        throw std::invalid_argument("frame counts must be non-negative");
    }
    return options;
}

bool is_directory(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool is_file(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

void make_directories(const std::string& path) {
    if (path.empty() || is_directory(path)) return;
    std::string current = path.front() == '/' ? "/" : "";
    std::stringstream parts(path);
    std::string part;
    while (std::getline(parts, part, '/')) {
        if (part.empty()) continue;
        if (!current.empty() && current.back() != '/') current += '/';
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

cv::Mat frame_mat(const rs2::video_frame& frame, int type) {
    return cv::Mat(
        cv::Size(frame.get_width(), frame.get_height()), type,
        const_cast<void*>(frame.get_data()), frame.get_stride_in_bytes());
}

void write_intrinsics(std::ostream& out, const rs2_intrinsics& value) {
    out << "{\"width\":" << value.width
        << ",\"height\":" << value.height
        << ",\"fx\":" << value.fx
        << ",\"fy\":" << value.fy
        << ",\"cx\":" << value.ppx
        << ",\"cy\":" << value.ppy
        << ",\"model\":" << static_cast<int>(value.model)
        << ",\"coeffs\":[";
    for (int index = 0; index < 5; ++index) {
        if (index) out << ',';
        out << value.coeffs[index];
    }
    out << "]}";
}

void write_extrinsics(std::ostream& out, const rs2_extrinsics& value) {
    out << '[';
    for (int row = 0; row < 4; ++row) {
        if (row) out << ',';
        out << '[';
        for (int column = 0; column < 4; ++column) {
            if (column) out << ',';
            if (row == 3 || column == 3) {
                out << (row == column ? 1.0f :
                        column == 3 ? value.translation[row] : 0.0f);
            } else {
                // librealsense stores rotation matrices in column-major order.
                out << value.rotation[column * 3 + row];
            }
        }
        out << ']';
    }
    out << ']';
}

void write_identity(std::ostream& out) {
    out << "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]";
}

void write_calibration(
    const std::string& path,
    const rs2::video_stream_profile& color,
    const rs2::video_stream_profile& depth,
    float depth_scale) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("could not write " + path);
    out << std::setprecision(10)
        << "{\n  \"timestamp_unit\": \"nanoseconds\",\n"
        << "  \"fixed_camera\": true,\n"
        << "  \"depth_scale_m\": " << depth_scale << ",\n"
        << "  \"color\": ";
    write_intrinsics(out, color.get_intrinsics());
    out << ",\n  \"depth\": ";
    write_intrinsics(out, depth.get_intrinsics());
    out << ",\n  \"T_depth_to_color\": ";
    write_extrinsics(out, depth.get_extrinsics_to(color));
    out << ",\n  \"T_color_to_device\": ";
    write_identity(out);
    out << "\n}\n";
}

float depth_scale(const rs2::device& device) {
    for (const rs2::sensor& sensor : device.query_sensors()) {
        const rs2::depth_sensor depth = sensor.as<rs2::depth_sensor>();
        if (depth) return depth.get_depth_scale();
    }
    throw std::runtime_error("active device has no depth sensor");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        const std::string raw = join(options.output, "raw");
        const std::string timestamps_path = join(raw, "timestamps.csv");
        if (is_file(timestamps_path)) {
            throw std::runtime_error(
                timestamps_path + " already exists; use a new session directory");
        }
        make_directories(join(raw, "rgb"));
        make_directories(join(raw, "depth"));
        make_directories(join(raw, "aligned_depth"));

        rs2::pipeline pipeline;
        rs2::config config;
        config.enable_device(options.serial);
        config.enable_stream(RS2_STREAM_COLOR, 0, 848, 480, RS2_FORMAT_BGR8, 30);
        config.enable_stream(RS2_STREAM_DEPTH, 0, 848, 480, RS2_FORMAT_Z16, 30);
        if (options.record_stream) {
            config.enable_record_to_file(join(raw, "sample.db3"));
        }

        const rs2::pipeline_profile active = pipeline.start(config);
        const auto color_profile = active.get_stream(RS2_STREAM_COLOR)
            .as<rs2::video_stream_profile>();
        const auto depth_profile = active.get_stream(RS2_STREAM_DEPTH)
            .as<rs2::video_stream_profile>();
        write_calibration(
            join(raw, "calibration.json"), color_profile, depth_profile,
            depth_scale(active.get_device()));

        std::cout << "Fixed-camera streams: color " << color_profile.width() << 'x'
                  << color_profile.height() << " @ " << color_profile.fps()
                  << ", depth " << depth_profile.width() << 'x'
                  << depth_profile.height() << " @ " << depth_profile.fps() << '\n';
        std::cout << "Warming up, discarding " << options.warmup_frames << " frames...\n";
        for (int index = 0; index < options.warmup_frames && g_running.load(); ++index) {
            pipeline.wait_for_frames(3000);
        }

        std::ofstream timestamps(timestamps_path);
        if (!timestamps) {
            throw std::runtime_error("could not create " + timestamps_path);
        }
        timestamps
            << "idx,rgb_timestamp_ns,depth_timestamp_ns,aligned_depth_timestamp_ns,"
               "rgb_file,depth_file,aligned_depth_file\n";

        rs2::align align_to_color(RS2_STREAM_COLOR);
        std::size_t index = 0;
        std::size_t dropped_capture_frames = 0;
        bool have_frame_number = false;
        unsigned long long previous_frame_number = 0;

        while (g_running.load()
               && (options.max_frames == 0
                   || index < static_cast<std::size_t>(options.max_frames))) {
            const rs2::frameset frames = pipeline.wait_for_frames(3000);
            const rs2::video_frame color = frames.get_color_frame();
            const rs2::depth_frame depth = frames.get_depth_frame();
            if (!color || !depth) continue;
            const rs2::frameset aligned = align_to_color.process(frames);
            const rs2::depth_frame aligned_depth = aligned.get_depth_frame();
            if (!aligned_depth) continue;

            const unsigned long long frame_number = color.get_frame_number();
            if (have_frame_number && frame_number > previous_frame_number + 1) {
                dropped_capture_frames += static_cast<std::size_t>(
                    frame_number - previous_frame_number - 1);
            }
            previous_frame_number = frame_number;
            have_frame_number = true;

            const std::string name = frame_name(index);
            const bool wrote_rgb = cv::imwrite(
                join(join(raw, "rgb"), name), frame_mat(color, CV_8UC3));
            const bool wrote_depth = cv::imwrite(
                join(join(raw, "depth"), name), frame_mat(depth, CV_16UC1));
            const bool wrote_aligned = cv::imwrite(
                join(join(raw, "aligned_depth"), name),
                frame_mat(aligned_depth, CV_16UC1));
            if (!wrote_rgb || !wrote_depth || !wrote_aligned) {
                throw std::runtime_error("failed to write image " + name);
            }

            timestamps << index << ',' << timestamp_ns(color) << ','
                       << timestamp_ns(depth) << ',' << timestamp_ns(aligned_depth)
                       << ",rgb/" << name << ",depth/" << name
                       << ",aligned_depth/" << name << '\n';
            ++index;
            if (index % 30 == 0) {
                std::cout << "Captured " << index << " RGB-D frames\r" << std::flush;
            }
        }

        pipeline.stop();
        timestamps.flush();
        std::ofstream summary(join(raw, "capture_summary.json"));
        if (!summary) throw std::runtime_error("could not write capture_summary.json");
        summary << "{\n  \"serial\": \"" << options.serial << "\",\n"
                << "  \"camera_mode\": \"fixed_rgbd\",\n"
                << "  \"frames\": " << index << ",\n"
                << "  \"dropped_capture_frames\": " << dropped_capture_frames
                << "\n}\n";

        std::cout << "\nCapture complete: " << index << " frames in " << raw << '\n';
        return index == 0 ? 2 : 0;
    } catch (const rs2::error& error) {
        std::cerr << "RealSense error in " << error.get_failed_function() << '(' 
                  << error.get_failed_args() << "): " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
