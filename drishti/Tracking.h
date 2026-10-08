#pragma once
// Multi-object tracking of box centres across scans, for speed (PS 26053: is that car moving?).
//
// Feed it each scan's detections as positions in a *fixed world frame*: the sensor-frame box
// centre moved by the scan's pose (for KITTI, kitti::Sequence::pose(i), the LiDAR pose in scan 0's
// frame). Without that, the vehicle's own motion shows up as every parked car's speed.
//
//   - Association: each track's position is predicted with its current velocity, then pairs
//     (track, detection) closer than `gate` are matched greedily, nearest first. Unmatched
//     detections start new tracks; a track unmatched for more than `max_missed` scans is dropped.
//   - Velocity: least-squares slope of x(t) and y(t) over the last `window` positions (1 s at
//     10 Hz). A box centre jitters by tens of centimetres as the visible part of a car changes;
//     a straight-line fit over a second averages that out far better than frame-to-frame
//     differences. Reported once a track has `min_hits` positions.
//
// Measured on SemanticKITTI 08, scans 0-499, SalsaNext car boxes (bbox::fit_boxes) in the frame
// of kitti::Sequence::pose, ego vehicle at ~9.4 m/s: parked cars read median 0.36 m/s, 90% under
// 1.0 m/s, 2.7% above 1.5 m/s; the three moving cars read a steady 9.7-10.4 m/s. (A 10-scan window
// gave 4.8% of parked readings above 1.5 m/s.) SemanticKITTI's "moving-car" also tags cars that
// are stationary in those scans, so compare against it by track, not by the label alone.
//
// Plain C++, 2D (bird's-eye view: speed along the ground).

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace track {

struct Params {
    float gate = 2.5f;          // metres between prediction and detection to match
    int max_missed = 2;         // scans a track survives without a detection
    int window = 15;            // positions in the velocity fit (1.5 s at 10 Hz)
    int min_hits = 6;           // positions needed before the velocity is reported
};

struct Sample { double t; float x, y; };

struct Track {
    int id = 0;
    std::deque<Sample> history; // oldest first, at most `window`
    int missed = 0;
    float vx = 0, vy = 0;       // m/s, world frame
    bool has_velocity = false;
    float speed() const { return std::hypot(vx, vy); }
};

class Tracker {
public:
    explicit Tracker(const Params& p = {}) : p_(p) {}

    // One scan at time `t` (seconds): `xy` = detections' world positions (x, y pairs).
    // Returns the track index (into tracks()) of each detection.
    std::vector<int> update(const std::vector<float>& xy, double t) {
        const size_t nd = xy.size() / 2;
        // Predicted positions.
        std::vector<float> px(tracks_.size()), py(tracks_.size());
        for (size_t k = 0; k < tracks_.size(); ++k) {
            const Track& tr = tracks_[k];
            const Sample& s = tr.history.back();
            float dt = (float)(t - s.t);
            px[k] = s.x + (tr.has_velocity ? tr.vx * dt : 0.0f);
            py[k] = s.y + (tr.has_velocity ? tr.vy * dt : 0.0f);
        }
        // Greedy nearest-first matching inside the gate.
        struct Pair { float d; int k, j; };
        std::vector<Pair> pairs;
        for (size_t k = 0; k < tracks_.size(); ++k)
            for (size_t j = 0; j < nd; ++j) {
                float d = std::hypot(xy[j * 2] - px[k], xy[j * 2 + 1] - py[k]);
                if (d < p_.gate) pairs.push_back({d, (int)k, (int)j});
            }
        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d < b.d; });
        std::vector<int> det_track(nd, -1);
        std::vector<char> track_used(tracks_.size(), 0);
        for (const Pair& pr : pairs) {
            if (track_used[pr.k] || det_track[pr.j] >= 0) continue;
            track_used[pr.k] = 1;
            det_track[pr.j] = pr.k;
        }
        // Update matched tracks, age unmatched ones.
        for (size_t k = 0; k < tracks_.size(); ++k) {
            if (!track_used[k]) { tracks_[k].missed++; continue; }
        }
        for (size_t j = 0; j < nd; ++j) {
            if (det_track[j] >= 0) {
                Track& tr = tracks_[det_track[j]];
                tr.missed = 0;
                tr.history.push_back({t, xy[j * 2], xy[j * 2 + 1]});
                if ((int)tr.history.size() > p_.window) tr.history.pop_front();
                fit_velocity(tr);
            } else {
                Track tr;
                tr.id = next_id_++;
                tr.history.push_back({t, xy[j * 2], xy[j * 2 + 1]});
                det_track[j] = (int)tracks_.size();
                tracks_.push_back(std::move(tr));
            }
        }
        // Drop stale tracks, remapping the detections' indices.
        std::vector<int> remap(tracks_.size(), -1);
        std::vector<Track> kept;
        for (size_t k = 0; k < tracks_.size(); ++k) {
            if (tracks_[k].missed > p_.max_missed) continue;
            remap[k] = (int)kept.size();
            kept.push_back(std::move(tracks_[k]));
        }
        tracks_ = std::move(kept);
        for (int& k : det_track) k = remap[k];
        return det_track;
    }

    const std::vector<Track>& tracks() const { return tracks_; }
    void reset() { tracks_.clear(); }

private:
    Params p_;
    std::vector<Track> tracks_;
    int next_id_ = 1;

    void fit_velocity(Track& tr) const {
        const size_t n = tr.history.size();
        if ((int)n < p_.min_hits) { tr.has_velocity = false; return; }
        double mt = 0, mx = 0, my = 0;
        for (const Sample& s : tr.history) { mt += s.t; mx += s.x; my += s.y; }
        mt /= n; mx /= n; my /= n;
        double stt = 0, stx = 0, sty = 0;
        for (const Sample& s : tr.history) {
            double dt = s.t - mt;
            stt += dt * dt;
            stx += dt * (s.x - mx);
            sty += dt * (s.y - my);
        }
        if (stt <= 0) { tr.has_velocity = false; return; }
        tr.vx = (float)(stx / stt);
        tr.vy = (float)(sty / stt);
        tr.has_velocity = true;
    }
};

} // namespace track
