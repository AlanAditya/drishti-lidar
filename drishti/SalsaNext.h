#pragma once
// SalsaNext LiDAR semantic segmentation (Cortinhal et al., pretrained on SemanticKITTI), in MLX.
//
// Input : a range image, [1, H, W, 5] float32 NHWC (H = 64, W = 2048 for the HDL-64 it was trained
//         on), channels (range, x, y, z, remission), each already normalised with the model's
//         means/stds: (value - mean) / std. Pixels with no LiDAR return are 0 in every channel.
// Output: per-pixel class probabilities, [1, H, W, classes] (20: 0 = unlabelled, then car ...
//         traffic-sign; `class_names` / `learning_map_inv` give the name / SemanticKITTI label id).
//
// Weights come from tools/salsanext/convert_salsanext.py, which also checked that these exact ops
// reproduce the original PyTorch model (salsanext_golden.safetensors). Layout: NHWC activations,
// conv weights [out, kh, kw, in], every BatchNorm pre-folded into a per-channel scale + shift
// (not into the conv before it: SalsaNext puts the LeakyReLU in between).
//
// macOS only: MLX is linked for macOS / visionOS, not iOS.

#include <TargetConditionals.h>
#if !TARGET_OS_IPHONE

#include <mlx/mlx.h>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai {

namespace mx = mlx::core;

class SalsaNext {
public:
    int height = 64, width = 2048, classes = 20;
    float fov_up = 3.0f, fov_down = -25.0f;       // degrees, of the sensor it was trained on
    std::vector<float> means, stds;                // per input channel: range, x, y, z, remission
    std::vector<int> learning_map_inv;             // output class -> SemanticKITTI label id
    std::vector<std::string> class_names;          // output class -> name
    std::vector<float> class_colors;               // output class -> RGB in [0, 1] (3 per class), SemanticKITTI's colours

    explicit SalsaNext(const std::string& safetensors_path) {
        auto [arrays, meta] = mx::load_safetensors(safetensors_path);
        weights_ = std::move(arrays);
        auto get = [&](const char* key) -> std::string {
            auto it = meta.find(key);
            if (it == meta.end()) throw std::runtime_error(std::string("SalsaNext: missing metadata '") + key + "'");
            return it->second;
        };
        height = std::stoi(get("height"));
        width = std::stoi(get("width"));
        classes = std::stoi(get("classes"));
        fov_up = std::stof(get("fov_up"));
        fov_down = std::stof(get("fov_down"));
        means = split_floats(get("img_means"));
        stds = split_floats(get("img_stds"));
        for (float v : split_floats(get("learning_map_inv"))) learning_map_inv.push_back((int)v);
        class_names = split(get("class_names"));
        for (float v : split_floats(get("class_colors_rgb"))) class_colors.push_back(v / 255.0f);

        // Compiled once: later calls with the same input shape skip rebuilding the graph.
        compiled_ = mx::compile([this](const std::vector<mx::array>& in) {
            return std::vector<mx::array>{ forward(in[0]) };
        });
    }

    // [1, H, W, 5] normalised range image -> [1, H, W, classes] probabilities (lazy: mx::eval it).
    mx::array operator()(const mx::array& x) { return compiled_({x})[0]; }

    // The uncompiled forward pass, for debugging.
    mx::array forward(const mx::array& x) {
        mx::array d = res_context(x, "downCntx");
        d = res_context(d, "downCntx2");
        d = res_context(d, "downCntx3");
        auto [down0c, down0b] = res_block(d, "resBlock1");
        auto [down1c, down1b] = res_block(down0c, "resBlock2");
        auto [down2c, down2b] = res_block(down1c, "resBlock3");
        auto [down3c, down3b] = res_block(down2c, "resBlock4");
        mx::array down5c = res_block_no_pool(down3c, "resBlock5");
        mx::array up = up_block(down5c, down3b, "upBlock1");
        up = up_block(up, down2b, "upBlock2");
        up = up_block(up, down1b, "upBlock3");
        up = up_block(up, down0b, "upBlock4");
        return mx::softmax(conv(up, "logits"), -1);
    }

private:
    std::unordered_map<std::string, mx::array> weights_;
    std::unordered_map<int, mx::array> pool_kernels_;   // by channel count
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled_;

    const mx::array& w(const std::string& name) const {
        auto it = weights_.find(name);
        if (it == weights_.end()) throw std::runtime_error("SalsaNext: missing weight '" + name + "'");
        return it->second;
    }

    mx::array conv(const mx::array& x, const std::string& name, int padding = 0, int dilation = 1) const {
        return mx::add(mx::conv2d(x, w(name + ".weight"), {1, 1}, {padding, padding}, {dilation, dilation}),
                       w(name + ".bias"));
    }
    mx::array bn(const mx::array& x, const std::string& name) const {
        return mx::add(mx::multiply(x, w(name + ".scale")), w(name + ".shift"));
    }
    static mx::array leaky(const mx::array& x) {   // nn.LeakyReLU(), slope 0.01
        return mx::maximum(x, mx::multiply(x, mx::array(0.01f)));
    }

    // nn.AvgPool2d(3, stride=2, padding=1) with count_include_pad (always divides by 9), as a
    // depthwise conv with a constant 1/9 kernel.
    mx::array avg_pool3_s2(const mx::array& x) {
        int c = x.shape(-1);
        auto it = pool_kernels_.find(c);
        if (it == pool_kernels_.end()) {
            it = pool_kernels_.emplace(c, mx::full({c, 3, 3, 1}, 1.0f / 9.0f, mx::float32)).first;
        }
        return mx::conv2d(x, it->second, {2, 2}, {1, 1}, {1, 1}, c);
    }

    // nn.PixelShuffle(2) in NHWC: out[2h+i, 2w+j, c] = in[h, w, 4c + 2i + j].
    static mx::array pixel_shuffle2(const mx::array& x) {
        int n = x.shape(0), h = x.shape(1), wd = x.shape(2), c = x.shape(3) / 4;
        mx::array y = mx::reshape(x, {n, h, wd, c, 2, 2});
        y = mx::transpose(y, {0, 1, 4, 2, 5, 3});
        return mx::reshape(y, {n, h * 2, wd * 2, c});
    }

    mx::array act_bn(const mx::array& x, const std::string& p, const std::string& conv_name,
                     const std::string& bn_name, int padding = 0, int dilation = 1) const {
        return bn(leaky(conv(x, p + conv_name, padding, dilation)), p + bn_name);
    }

    mx::array res_context(const mx::array& x, const std::string& p) const {
        mx::array shortcut = leaky(conv(x, p + ".conv1"));
        mx::array a1 = act_bn(shortcut, p, ".conv2", ".bn1", 1);
        mx::array a2 = act_bn(a1, p, ".conv3", ".bn2", 2, 2);
        return mx::add(shortcut, a2);
    }

    mx::array res_block_no_pool(const mx::array& x, const std::string& p) const {
        mx::array shortcut = leaky(conv(x, p + ".conv1"));
        mx::array a1 = act_bn(x, p, ".conv2", ".bn1", 1);
        mx::array a2 = act_bn(a1, p, ".conv3", ".bn2", 2, 2);
        mx::array a3 = act_bn(a2, p, ".conv4", ".bn3", 1, 2);
        mx::array a = act_bn(mx::concatenate({a1, a2, a3}, -1), p, ".conv5", ".bn4");
        return mx::add(shortcut, a);
    }
    // Returns (pooled, pre-pool): the pre-pool one is the skip connection for the up path.
    std::pair<mx::array, mx::array> res_block(const mx::array& x, const std::string& p) {
        mx::array a = res_block_no_pool(x, p);
        return {avg_pool3_s2(a), a};
    }

    mx::array up_block(const mx::array& x, const mx::array& skip, const std::string& p) const {
        mx::array b = mx::concatenate({pixel_shuffle2(x), skip}, -1);
        mx::array e1 = act_bn(b, p, ".conv1", ".bn1", 1);
        mx::array e2 = act_bn(e1, p, ".conv2", ".bn2", 2, 2);
        mx::array e3 = act_bn(e2, p, ".conv3", ".bn3", 1, 2);
        return act_bn(mx::concatenate({e1, e2, e3}, -1), p, ".conv4", ".bn4");
    }

    static std::vector<std::string> split(const std::string& s) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        for (std::string item; std::getline(ss, item, ',');) out.push_back(item);
        return out;
    }
    static std::vector<float> split_floats(const std::string& s) {
        std::vector<float> out;
        for (auto& item : split(s)) out.push_back(std::stof(item));
        return out;
    }
};

} // namespace ai

#endif // !TARGET_OS_IPHONE
