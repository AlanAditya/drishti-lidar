// OpenCV-style window for the WorldOf3D renderer, outside Xcode.
//
//   auto r = Renderer::init(1500, 800, "matrix");                          // opens a window, returns immediately
//   auto [g, s, h] = r.subplots<GraphViewer, Viewer, PlotterViewer>(1, 3);  // one tile per viewer type
//   auto wave  = g->plot(x, matrix::sin(x));                               // LineController
//   auto cloud = s->points(pts, colors);                                   // PointCloudController
//   h->update_plot(heat);
//   while (r.is_open()) { wave.update(x, y2); r.poll(); }                  // draw a frame + handle input
//   r.wait();                                                              // until every window is closed
//
// Each tile IS a Viewer (the app's own classes, or a subclass of yours): draw into it with its
// methods, which return the controllers that own the new nodes. Nothing here owns your main():
// the window only draws and reacts to input inside poll() / show() / wait_key() / wait(), like
// OpenCV's highgui. Everything runs on the main thread (macOS only lets it drive windows).
#pragma once
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <map>
#include <unordered_map>
#include "matrix.h"
// Same order as MatrixH.mm: the SOPs expect Camera3D and CPURasteriser to be declared already.
#include "Mods/Camera3D.h"
#include "cpu_rasteriser.hpp"
#include "SOPs/Viewer.cpp"
#include "SOPs/Controllers.cpp"
#include "SOPs/PlotterViewer.h"
#include "SOPs/GraphViewer.cpp"
#include "SOPs/HistogramViewer.cpp"
#include "SOPs/Colors.cpp"

class Renderer {
public:
    // Opens a window (and the app, on the first call). Several windows are fine.
    static Renderer init(int width = 1280, int height = 800, const std::string& title = "matrix");

    // Lays the window out as a rows x cols grid, filled row by row with one new viewer per type,
    // and returns them: auto [a, b, c] = r.subplots<GraphViewer, Viewer, MyViewer>(1, 3);
    // Replaces whatever the window showed before. Each tile gets its own camera.
    template <class... Ts>
    std::tuple<std::shared_ptr<Ts>...> subplots(int rows, int cols) {
        static_assert(sizeof...(Ts) > 0, "subplots<...>: name at least one viewer type");
        static_assert((std::is_base_of_v<Viewer, Ts> && ...), "subplots<...>: every type must derive from Viewer");
        std::tuple<std::shared_ptr<Ts>...> tiles{std::make_shared<Ts>()...};
        std::vector<std::shared_ptr<Viewer>> list;
        std::apply([&](auto&... v) { (list.push_back(v), ...); }, tiles);
        layout(list, rows, cols);
        return tiles;
    }
    template <class... Ts>
    std::tuple<std::shared_ptr<Ts>...> subplots() { return subplots<Ts...>(1, (int)sizeof...(Ts)); }   // one row

    // Adds a viewer you made yourself, over the rectangle `rect` = {x, y, width, height} as fractions
    // of the window (origin top left). Keeps what's already there.
    template <class T>
    std::shared_ptr<T> add(std::shared_ptr<T> viewer, simd_float4 rect = {0, 0, 1, 1}) {
        add_viewer(viewer, rect);
        return viewer;
    }

    void clear();                          // removes every viewer from this window
    void set_grid(bool on);                // infinite ground grid behind viewers with show_grid (default on)
    void set_background(simd_float4 rgba); // default black

    // --- event loop. Keys come back as the character pressed ('a', ' ', 27 = Esc), or -1 for none.
    int poll();                   // handle pending input, draw a frame if one is due (~1/120 s since the last). Never blocks.
    int show();                   // handle pending input and draw a frame now
    int wait_key(int ms = 0);     // cv::waitKey: keep drawing until a key (returned), `ms` elapse, or every window closes; ms <= 0: no timeout
    void wait();                  // keep drawing until every window is closed
    bool save(const std::string& png_path);   // draws a frame and writes it to a PNG
    bool is_open() const;
    void close();

    struct Impl;
    std::shared_ptr<Impl> impl;

private:
    void layout(const std::vector<std::shared_ptr<Viewer>>& viewers, int rows, int cols);
    void add_viewer(std::shared_ptr<Viewer> viewer, simd_float4 rect);
};
