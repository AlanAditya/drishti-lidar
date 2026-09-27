#pragma once
// Reader for SemanticKITTI sequences (pre-recorded Velodyne HDL-64 road scans + per-point labels).
// Plain C++, no matrix / MLX, so it can be tested on its own. Layout of one sequence:
//
//   sequences/08/velodyne/000000.bin   one scan: float32 x, y, z, remission per point (16 bytes each, no header)
//   sequences/08/labels/000000.label   uint32 per point, same order: lower 16 bits = SemanticKITTI class id
//                                      (40 = road, 10 = car, 252 = moving-car...), upper 16 bits = instance id
//   sequences/08/poses.txt             one line per scan: the 3x4 pose, in the *left camera* frame
//   sequences/08/calib.txt             "Tr:" = LiDAR -> left camera (3x4)
//
// Coordinates: x forward, y left, z up, metres, relative to the LiDAR.

#include <simd/simd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kitti {

struct Scan {
    std::vector<float> xyz;          // [N * 3] x, y, z (metres)
    std::vector<float> remission;    // [N]
    std::vector<uint16_t> labels;    // [N] SemanticKITTI class ids; empty if the scan has no .label file
    size_t size() const { return remission.size(); }
};

class Sequence {
public:
    // `dir` is one sequence folder, e.g. ".../dataset/sequences/08".
    explicit Sequence(const std::string& dir) : dir_(dir) {
        namespace fs = std::filesystem;
        if (!fs::is_directory(dir_ + "/velodyne")) {
            throw std::runtime_error("kitti::Sequence: no velodyne/ folder in " + dir_);
        }
        for (auto& e : fs::directory_iterator(dir_ + "/velodyne")) {
            if (e.path().extension() == ".bin") ++count_;
        }
        load_poses();
    }

    size_t size() const { return count_; }
    bool has_labels() const { return std::filesystem::exists(path("labels", 0, ".label")); }
    bool has_poses() const { return !poses_.empty(); }

    Scan load(size_t index) const {
        Scan scan;
        std::vector<float> raw = read_file<float>(path("velodyne", index, ".bin"));
        if (raw.size() % 4 != 0) throw std::runtime_error("kitti: truncated scan " + path("velodyne", index, ".bin"));
        const size_t n = raw.size() / 4;
        scan.xyz.resize(n * 3);
        scan.remission.resize(n);
        for (size_t i = 0; i < n; ++i) {
            scan.xyz[i * 3 + 0] = raw[i * 4 + 0];
            scan.xyz[i * 3 + 1] = raw[i * 4 + 1];
            scan.xyz[i * 3 + 2] = raw[i * 4 + 2];
            scan.remission[i] = raw[i * 4 + 3];
        }
        std::string label_path = path("labels", index, ".label");
        if (std::filesystem::exists(label_path)) {
            std::vector<uint32_t> packed = read_file<uint32_t>(label_path);
            if (packed.size() != n) throw std::runtime_error("kitti: label count doesn't match " + label_path);
            scan.labels.resize(n);
            for (size_t i = 0; i < n; ++i) scan.labels[i] = (uint16_t)(packed[i] & 0xFFFF);
        }
        return scan;
    }

    // Pose of scan `index` in the LiDAR frame of scan 0 (maps this scan's points into that frame).
    // Identity if there's no poses.txt.
    simd_float4x4 pose(size_t index) const {
        return index < poses_.size() ? poses_[index] : matrix_identity_float4x4;
    }

    // "000042"-style file path.
    std::string path(const char* sub, size_t index, const char* ext) const {
        char name[32];
        std::snprintf(name, sizeof(name), "%06zu", index);
        return dir_ + "/" + sub + "/" + name + ext;
    }

private:
    std::string dir_;
    size_t count_ = 0;
    std::vector<simd_float4x4> poses_;

    template <typename T>
    static std::vector<T> read_file(const std::string& p) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("kitti: can't open " + p);
        std::streamsize bytes = f.tellg();
        f.seekg(0);
        std::vector<T> out((size_t)bytes / sizeof(T));
        f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(out.size() * sizeof(T)));
        return out;
    }

    // Row-major 3x4 (12 numbers) -> column-major simd 4x4 with a (0,0,0,1) bottom row.
    static simd_float4x4 from_3x4(const float* v) {
        simd_float4x4 m = matrix_identity_float4x4;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) m.columns[c][r] = v[r * 4 + c];
        return m;
    }

    // poses.txt is in the left camera's frame; with Tr (LiDAR -> camera) from calib.txt the LiDAR
    // pose is Tr^-1 * P * Tr (the conversion SemanticKITTI's own tools use).
    void load_poses() {
        std::ifstream pf(dir_ + "/poses.txt");
        if (!pf) return;
        simd_float4x4 Tr = matrix_identity_float4x4;
        std::ifstream cf(dir_ + "/calib.txt");
        for (std::string line; cf && std::getline(cf, line);) {
            if (line.rfind("Tr:", 0) != 0) continue;
            std::istringstream ss(line.substr(3));
            float v[12];
            for (float& x : v) ss >> x;
            Tr = from_3x4(v);
        }
        simd_float4x4 TrInv = simd_inverse(Tr);
        for (std::string line; std::getline(pf, line);) {
            std::istringstream ss(line);
            float v[12];
            int k = 0;
            while (k < 12 && ss >> v[k]) ++k;
            if (k == 12) poses_.push_back(simd_mul(TrInv, simd_mul(from_3x4(v), Tr)));
        }
    }
};

// ---- classes -----------------------------------------------------------------------------------

// SemanticKITTI class id (0..259) -> SalsaNext's 20 training classes (data_cfg.yaml "learning_map":
// moving-X merged into X, rare classes into 0 = unlabelled).
inline int to_train_class(uint16_t id) {
    switch (id) {
        case 10: case 252: return 1;                      // car
        case 11: return 2;                                // bicycle
        case 15: case 255: return 3;                      // motorcycle
        case 18: case 258: return 4;                      // truck
        case 13: case 16: case 20: case 256: case 257: case 259: return 5;   // other-vehicle
        case 30: case 254: return 6;                      // person
        case 31: case 253: return 7;                      // bicyclist
        case 32: return 8;                                // motorcyclist
        case 40: case 60: return 9;                       // road (+ lane-marking)
        case 44: return 10;                               // parking
        case 48: return 11;                               // sidewalk
        case 49: return 12;                               // other-ground
        case 50: return 13;                               // building
        case 51: return 14;                               // fence
        case 70: return 15;                               // vegetation
        case 71: return 16;                               // trunk
        case 72: return 17;                               // terrain
        case 80: return 18;                               // pole
        case 81: return 19;                               // traffic-sign
        default: return 0;                                // unlabelled / outlier / other-*
    }
}

// SemanticKITTI ids >= 252 are objects that were actually moving in that scan.
inline bool is_moving(uint16_t id) { return id >= 252 && id <= 259; }

// The project's three groups (PS 26053): what a planner cares about.
enum class Group : int { Unknown = 0, Terrain = 1, Static = 2, Dynamic = 3 };

// SalsaNext's 20 classes -> group. "Dynamic" = can move (vehicles, people, riders); whether it
// actually *is* moving needs ego-motion-compensated tracking across scans (or is_moving() on labels).
inline Group group_of(int train_class) {
    switch (train_class) {
        case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: return Group::Dynamic;
        case 9: case 10: case 11: case 12: case 17: return Group::Terrain;
        case 13: case 14: case 15: case 16: case 18: case 19: return Group::Static;
        default: return Group::Unknown;
    }
}

} // namespace kitti
