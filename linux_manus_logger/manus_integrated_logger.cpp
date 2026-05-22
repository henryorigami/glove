#include <ManusSDK.h>
#include <ManusSDKTypeInitializers.h>
#include <ManusSDKTypes.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static std::atomic<bool> g_stop{false};

static int64_t wall_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string utc_stamp() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

static uint64_t manus_time(const ManusTimestamp& timestamp) {
    return timestamp.time;
}

static bool sane_pose(float px, float py, float pz, float qw, float qx, float qy, float qz) {
    const float vals[] = {px, py, pz, qw, qx, qy, qz};
    for (float v : vals) {
        if (!std::isfinite(v) || std::fabs(v) > 100.0f) return false;
    }
    return true;
}

static void signal_handler(int) {
    g_stop.store(true);
}

struct NodeMeta {
    int32_t chain_type = -1;
    int32_t side = -1;
    int32_t finger_joint_type = -1;
    uint32_t parent_id = 0;
};

struct NodeRow {
    uint32_t node_id = 0;
    NodeMeta meta;
    float px = 0, py = 0, pz = 0;
    float qw = 1, qx = 0, qy = 0, qz = 0;
};

struct SkeletonSample {
    int64_t t_mono_ns = 0;
    int64_t t_wall_ns = 0;
    uint64_t t_manus = 0;
    uint32_t glove_id = 0;
    uint32_t frame_seq = 0;
    std::vector<NodeRow> nodes;
};

struct RawDeviceSample {
    int64_t t_mono_ns = 0;
    int64_t t_wall_ns = 0;
    uint64_t t_manus = 0;
    uint32_t device_id = 0;
    uint32_t frame_seq = 0;
    float qw = 1, qx = 0, qy = 0, qz = 0;
    std::vector<NodeRow> sensors;
};

struct ErgoSample {
    int64_t t_mono_ns = 0;
    int64_t t_wall_ns = 0;
    uint64_t t_manus = 0;
    uint32_t glove_id = 0;
    bool is_user_id = false;
    uint32_t frame_seq = 0;
    std::vector<float> values;
};

class IntegratedLogger {
public:
    IntegratedLogger(fs::path session_dir, std::string prefix, int duration_sec)
        : dir_(std::move(session_dir)), prefix_(std::move(prefix)), duration_sec_(duration_sec) {}

    bool start() {
        if (instance_) return false;
        instance_ = this;
        fs::create_directories(dir_);
        t0_ = Clock::now();

        skel_.open(dir_ / (prefix_ + "raw_skeleton.csv"));
        ergo_.open(dir_ / (prefix_ + "ergonomics.csv"));
        raw_dev_.open(dir_ / (prefix_ + "raw_devices.csv"));

        skel_ << "t_mono_ns,t_manus,t_wall_ns,glove_id,frame_seq,node_id,chain_type,side,finger_joint_type,parent_id,px,py,pz,qw,qx,qy,qz\n";
        ergo_ << "t_mono_ns,t_manus,t_wall_ns,glove_id,is_user_id,frame_seq,metric_idx,value\n";
        raw_dev_ << "t_mono_ns,t_manus,t_wall_ns,device_id,frame_seq,sensor_idx,px,py,pz,qw,qx,qy,qz\n";

        SDKReturnCode rc = CoreSdk_InitializeIntegrated();
        if (rc != SDKReturnCode_Success) {
            std::fprintf(stderr, "CoreSdk_InitializeIntegrated failed: %d\n", (int)rc);
            return false;
        }

        if (!register_callbacks()) return false;
        if (!init_coordinate_system()) return false;

        ManusHost empty;
        ManusHost_Init(&empty);
        rc = CoreSdk_ConnectToHost(empty);
        if (rc != SDKReturnCode_Success) {
            std::fprintf(stderr, "CoreSdk_ConnectToHost(integrated) failed: %d\n", (int)rc);
            return false;
        }

        rc = CoreSdk_SetRawSkeletonHandMotion(HandMotion_Auto);
        if (rc != SDKReturnCode_Success) {
            std::fprintf(stderr, "CoreSdk_SetRawSkeletonHandMotion warning: %d\n", (int)rc);
        }

        running_.store(true);
        writer_ = std::thread([this] { writer_loop(); });
        std::printf("Linux integrated MANUS logger started: %s\n", dir_.string().c_str());
        return true;
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (writer_.joinable()) writer_.join();
        CoreSdk_ShutDown();
        skel_.close();
        ergo_.close();
        raw_dev_.close();
        instance_ = nullptr;
    }

    int64_t mono_ns() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0_).count();
    }

private:
    static IntegratedLogger* instance_;

    static void on_log(LogSeverity sev, const char* const msg, uint32_t length) {
        std::string text(msg, msg + length);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        std::fprintf(stderr, "[manus-linux sev=%d] %s\n", (int)sev, text.c_str());
    }

    static void on_connect(const ManusHost* const host) {
        if (!host) return;
        std::printf("Connected integrated host: %s @ %s\n", host->hostName, host->ipAddress);
    }

    static void on_disconnect(const ManusHost* const) {
        std::fprintf(stderr, "MANUS integrated host disconnected\n");
    }

    static void on_raw_skeleton(const SkeletonStreamInfo* const info) {
        if (!instance_ || !info) return;
        for (uint32_t i = 0; i < info->skeletonsCount; ++i) {
            RawSkeletonInfo si;
            if (CoreSdk_GetRawSkeletonInfo(i, &si) != SDKReturnCode_Success) continue;
            std::vector<SkeletonNode> nodes(si.nodesCount);
            if (CoreSdk_GetRawSkeletonData(i, nodes.data(), si.nodesCount) != SDKReturnCode_Success) continue;

            const auto* topo = instance_->topology(si.gloveId, si.nodesCount);
            SkeletonSample sample;
            sample.t_mono_ns = instance_->mono_ns();
            sample.t_wall_ns = wall_ns();
            sample.t_manus = manus_time(info->publishTime);
            sample.glove_id = si.gloveId;
            sample.frame_seq = instance_->skel_seq_++;
            sample.nodes.reserve(nodes.size());

            for (const auto& n : nodes) {
                NodeRow row;
                row.node_id = n.id;
                row.px = n.transform.position.x;
                row.py = n.transform.position.y;
                row.pz = n.transform.position.z;
                row.qw = n.transform.rotation.w;
                row.qx = n.transform.rotation.x;
                row.qy = n.transform.rotation.y;
                row.qz = n.transform.rotation.z;
                if (topo) {
                    auto it = topo->find(n.id);
                    if (it != topo->end()) row.meta = it->second;
                }
                if (!sane_pose(row.px, row.py, row.pz, row.qw, row.qx, row.qy, row.qz)) {
                    sample.nodes.clear();
                    break;
                }
                sample.nodes.push_back(row);
            }
            if (sample.nodes.empty()) {
                instance_->skel_bad_frames_.fetch_add(1);
                continue;
            }

            std::lock_guard<std::mutex> lock(instance_->skel_mtx_);
            instance_->skel_q_.push_back(std::move(sample));
        }
    }

    static void on_raw_device(const RawDeviceDataInfo* const info) {
        if (!instance_ || !info) return;
        for (uint32_t i = 0; i < info->rawDeviceDataCount; ++i) {
            RawDeviceData dev;
            if (CoreSdk_GetRawDeviceData(i, &dev) != SDKReturnCode_Success) continue;

            RawDeviceSample sample;
            sample.t_mono_ns = instance_->mono_ns();
            sample.t_wall_ns = wall_ns();
            sample.t_manus = manus_time(info->publishTime);
            sample.device_id = dev.id;
            sample.frame_seq = instance_->raw_dev_seq_++;
            sample.qw = dev.rotation.w;
            sample.qx = dev.rotation.x;
            sample.qy = dev.rotation.y;
            sample.qz = dev.rotation.z;
            sample.sensors.reserve(dev.sensorCount);

            for (uint32_t j = 0; j < dev.sensorCount; ++j) {
                const auto& s = dev.sensorData[j];
                NodeRow row;
                row.node_id = j;
                row.px = s.position.x;
                row.py = s.position.y;
                row.pz = s.position.z;
                row.qw = s.rotation.w;
                row.qx = s.rotation.x;
                row.qy = s.rotation.y;
                row.qz = s.rotation.z;
                if (sane_pose(row.px, row.py, row.pz, row.qw, row.qx, row.qy, row.qz)) {
                    sample.sensors.push_back(row);
                }
            }

            std::lock_guard<std::mutex> lock(instance_->raw_dev_mtx_);
            instance_->raw_dev_q_.push_back(std::move(sample));
        }
    }

    static void on_ergo(const ErgonomicsStream* const stream) {
        if (!instance_ || !stream) return;
        for (uint32_t i = 0; i < stream->dataCount; ++i) {
            const ErgonomicsData& e = stream->data[i];
            ErgoSample sample;
            sample.t_mono_ns = instance_->mono_ns();
            sample.t_wall_ns = wall_ns();
            sample.t_manus = manus_time(stream->publishTime);
            sample.glove_id = e.id;
            sample.is_user_id = e.isUserID;
            sample.frame_seq = instance_->ergo_seq_++;
            sample.values.assign(e.data, e.data + ErgonomicsDataType_MAX_SIZE);
            std::lock_guard<std::mutex> lock(instance_->ergo_mtx_);
            instance_->ergo_q_.push_back(std::move(sample));
        }
    }

    bool register_callbacks() {
        bool ok = true;
        ok &= CoreSdk_RegisterCallbackForOnConnect(on_connect) == SDKReturnCode_Success;
        ok &= CoreSdk_RegisterCallbackForOnDisconnect(on_disconnect) == SDKReturnCode_Success;
        ok &= CoreSdk_RegisterCallbackForOnLog(on_log) == SDKReturnCode_Success;
        ok &= CoreSdk_RegisterCallbackForRawSkeletonStream(on_raw_skeleton) == SDKReturnCode_Success;
        ok &= CoreSdk_RegisterCallbackForRawDeviceDataStream(on_raw_device) == SDKReturnCode_Success;
        ok &= CoreSdk_RegisterCallbackForErgonomicsStream(on_ergo) == SDKReturnCode_Success;
        if (!ok) std::fprintf(stderr, "Failed to register one or more MANUS callbacks\n");
        return ok;
    }

    bool init_coordinate_system() {
        CoordinateSystemVUH vuh;
        CoordinateSystemVUH_Init(&vuh);
        vuh.handedness = Side_Right;
        vuh.up = AxisPolarity_PositiveZ;
        vuh.view = AxisView_XFromViewer;
        vuh.unitScale = 1.0f;
        SDKReturnCode rc = CoreSdk_InitializeCoordinateSystemWithVUH(vuh, true);
        if (rc != SDKReturnCode_Success) {
            std::fprintf(stderr, "CoreSdk_InitializeCoordinateSystemWithVUH failed: %d\n", (int)rc);
            return false;
        }
        return true;
    }

    const std::unordered_map<uint32_t, NodeMeta>* topology(uint32_t glove_id, uint32_t expected_count) {
        std::lock_guard<std::mutex> lock(topo_mtx_);
        auto it = topo_.find(glove_id);
        if (it != topo_.end()) return &it->second;

        uint32_t count = 0;
        if (CoreSdk_GetRawSkeletonNodeCount(glove_id, count) != SDKReturnCode_Success || count == 0) return nullptr;
        std::vector<NodeInfo> infos(count);
        if (CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, infos.data(), count) != SDKReturnCode_Success) return nullptr;
        auto& map = topo_[glove_id];
        for (const auto& info : infos) {
            NodeMeta meta;
            meta.chain_type = (int32_t)info.chainType;
            meta.side = (int32_t)info.side;
            meta.finger_joint_type = (int32_t)info.fingerJointType;
            meta.parent_id = info.parentId;
            map[info.nodeId] = meta;
        }
        std::printf("Topology glove=%u nodes=%u expected=%u\n", glove_id, count, expected_count);
        return &map;
    }

    void writer_loop() {
        auto last_stats = Clock::now();
        while (running_.load() || !queues_empty()) {
            drain_skeleton();
            drain_ergo();
            drain_raw_device();
            skel_.flush();
            ergo_.flush();
            raw_dev_.flush();
            auto now = Clock::now();
            if (now - last_stats >= std::chrono::seconds(2)) {
                std::printf("stats skeleton_frames=%u ergonomics_frames=%u raw_device_frames=%u skipped_bad_skeleton=%u\n",
                            skel_frames_.load(), ergo_frames_.load(),
                            raw_dev_frames_.load(), skel_bad_frames_.load());
                std::fflush(stdout);
                last_stats = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::printf("stats skeleton_frames=%u ergonomics_frames=%u raw_device_frames=%u skipped_bad_skeleton=%u\n",
                    skel_frames_.load(), ergo_frames_.load(),
                    raw_dev_frames_.load(), skel_bad_frames_.load());
        std::fflush(stdout);
    }

    bool queues_empty() {
        std::lock_guard<std::mutex> a(skel_mtx_);
        std::lock_guard<std::mutex> b(ergo_mtx_);
        std::lock_guard<std::mutex> c(raw_dev_mtx_);
        return skel_q_.empty() && ergo_q_.empty() && raw_dev_q_.empty();
    }

    void drain_skeleton() {
        std::deque<SkeletonSample> q;
        {
            std::lock_guard<std::mutex> lock(skel_mtx_);
            q.swap(skel_q_);
        }
        for (const auto& s : q) {
            skel_frames_.fetch_add(1);
            for (const auto& n : s.nodes) {
                skel_ << s.t_mono_ns << ',' << s.t_manus << ',' << s.t_wall_ns << ','
                      << s.glove_id << ',' << s.frame_seq << ',' << n.node_id << ','
                      << n.meta.chain_type << ',' << n.meta.side << ',' << n.meta.finger_joint_type << ','
                      << n.meta.parent_id << ','
                      << n.px << ',' << n.py << ',' << n.pz << ','
                      << n.qw << ',' << n.qx << ',' << n.qy << ',' << n.qz << '\n';
            }
        }
    }

    void drain_ergo() {
        std::deque<ErgoSample> q;
        {
            std::lock_guard<std::mutex> lock(ergo_mtx_);
            q.swap(ergo_q_);
        }
        for (const auto& s : q) {
            ergo_frames_.fetch_add(1);
            for (size_t i = 0; i < s.values.size(); ++i) {
                ergo_ << s.t_mono_ns << ',' << s.t_manus << ',' << s.t_wall_ns << ','
                      << s.glove_id << ',' << (s.is_user_id ? 1 : 0) << ',' << s.frame_seq << ','
                      << i << ',' << s.values[i] << '\n';
            }
        }
    }

    void drain_raw_device() {
        std::deque<RawDeviceSample> q;
        {
            std::lock_guard<std::mutex> lock(raw_dev_mtx_);
            q.swap(raw_dev_q_);
        }
        for (const auto& s : q) {
            raw_dev_frames_.fetch_add(1);
            if (s.sensors.empty()) {
                raw_dev_ << s.t_mono_ns << ',' << s.t_manus << ',' << s.t_wall_ns << ','
                         << s.device_id << ',' << s.frame_seq << ",-1,,,,"
                         << s.qw << ',' << s.qx << ',' << s.qy << ',' << s.qz << '\n';
                continue;
            }
            for (size_t i = 0; i < s.sensors.size(); ++i) {
                const auto& n = s.sensors[i];
                raw_dev_ << s.t_mono_ns << ',' << s.t_manus << ',' << s.t_wall_ns << ','
                         << s.device_id << ',' << s.frame_seq << ',' << i << ','
                         << n.px << ',' << n.py << ',' << n.pz << ','
                         << n.qw << ',' << n.qx << ',' << n.qy << ',' << n.qz << '\n';
            }
        }
    }

    fs::path dir_;
    std::string prefix_;
    int duration_sec_ = 0;
    Clock::time_point t0_;
    std::atomic<bool> running_{false};

    std::ofstream skel_;
    std::ofstream ergo_;
    std::ofstream raw_dev_;

    std::mutex skel_mtx_;
    std::mutex ergo_mtx_;
    std::mutex raw_dev_mtx_;
    std::deque<SkeletonSample> skel_q_;
    std::deque<ErgoSample> ergo_q_;
    std::deque<RawDeviceSample> raw_dev_q_;

    std::mutex topo_mtx_;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, NodeMeta>> topo_;

    std::atomic<uint32_t> skel_seq_{0};
    std::atomic<uint32_t> ergo_seq_{0};
    std::atomic<uint32_t> raw_dev_seq_{0};
    std::atomic<uint32_t> skel_frames_{0};
    std::atomic<uint32_t> ergo_frames_{0};
    std::atomic<uint32_t> raw_dev_frames_{0};
    std::atomic<uint32_t> skel_bad_frames_{0};
    std::thread writer_;
};

IntegratedLogger* IntegratedLogger::instance_ = nullptr;

static fs::path make_session_dir(const fs::path& base) {
    return base / ("manus_linux_session_" + utc_stamp());
}

int main(int argc, char** argv) {
    fs::path session_dir;
    fs::path base_dir = fs::current_path();
    std::string prefix = "manus_";
    int duration = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            std::printf("Usage: %s [--session-dir PATH] [--duration SEC] [--prefix STR]\n", argv[0]);
            return 0;
        } else if (arg == "--session-dir" && i + 1 < argc) {
            session_dir = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::atoi(argv[++i]);
        } else if (arg == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (arg == "--base-dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", arg.c_str());
            return 2;
        }
    }

    if (session_dir.empty()) session_dir = make_session_dir(base_dir);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    IntegratedLogger logger(session_dir, prefix, duration);
    if (!logger.start()) return 1;

    const auto start = Clock::now();
    while (!g_stop.load()) {
        if (duration > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count();
            if (elapsed >= duration) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    logger.stop();
    std::printf("Stopped.\n");
    return 0;
}
