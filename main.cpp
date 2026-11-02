// drishti demo: plays a SemanticKITTI sequence and draws the foveated 2.5D grid (one tile per
// occupied cell, brighter = taller) and/or the points, coloured by SemanticKITTI class or by group
// (green terrain, blue static, red dynamic), and a labelled box around every car and person, with
// its tracked speed.
//
//   ./drishti <sequence dir> <salsanext.safetensors> [--scans N] [--save frame.png]
//
//   Space: pause / play    Esc: quit
//   --scans N        stop after N scans
//   --save out.png   write the last frame drawn to a PNG
//
// What's drawn is set by SHOW_GRID, SHOW_POINTS and BY_CLASS below.
#include "renderer.h"
#include "drishti.cpp"
#include "drishti_draw.cpp"
#include <chrono>
#include <cstdio>
#include <ratio>
#include <string>

// What's drawn.
constexpr bool SHOW_GRID = true;      // the foveated grid: one tile per occupied cell
constexpr bool SHOW_POINTS = false;   // the points
constexpr bool BY_CLASS = true;       // SemanticKITTI class colours; false: group colours (terrain / static / dynamic)

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <sequence dir> <salsanext.safetensors> [--scans N] [--save frame.png]\n", argv[0]);
        return 1;
    }
    size_t scans = SIZE_MAX;
    std::string png;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--scans") scans = std::stoul(argv[i + 1]);
        if (std::string(argv[i]) == "--save") png = argv[i + 1];
    }

    // ---- what lives across scans: the data, the network, the map, the trackers ----
    drishti::KittiSequence seq(argv[1]);                // scan count, poses; scans as matrices
    drishti::SalsaNext net(argv[2]);                    // weights + the sensor it was trained on
    drishti::Sensor sensor = net.sensor();              // 64 x 2048, +3 to -25 degrees (HDL-64)
    matrix groups = drishti::kitti_group_table();       // [20] Int32: class -> unknown / terrain / static / dynamic
    drishti::FoveatedGrid grid;                         // 5 cm cells to 10 m ... 40 cm to 100 m

    const int kCar = 1, kPerson = 6;                    // SalsaNext's classes
    drishti::BoxParams car_params;                      // defaults are tuned on cars
    drishti::BoxParams person_params;                   // people: fewer points, no minimum width
    person_params.min_points = 8;
    person_params.min_width = 0.0f;
    const float car_cell = 0.3f, person_cell = 0.2f;    // clustering cell: people stand close together
    drishti::Tracker cars, people;

    // ---- the window: one 3D view behind and above our car, the grid as the ground ----
    auto r = Renderer::init(1600, 900, "drishti");
    auto [scene] = r.subplots<Viewer>();
    scene->camera->lookAt(simd_make_float3(0, 6.0f, 12.0f), simd_make_float3(0, -1.73f, -8.0f));
    r.set_grid(false);
    auto grid_mesh    = scene->add(TriangleMeshController("foveated grid"));
    auto cloud        = scene->add(PointCloudController("points"));
    auto car_boxes    = scene->add(LabelledBoxController("cars"));
    auto person_boxes = scene->add(LabelledBoxController("people", simd_float4{1.0f, 0.86f, 0.55f, 1.0f},
                                                                   simd_float4{1.0f, 0.55f, 0.15f, 0.28f}));
    // Each label's colour: SemanticKITTI's for the 20 classes, or the 4 group colours.
    const matrix palette = BY_CLASS ? drishti::draw::class_palette(net) : drishti::draw::group_palette();

    bool paused = false;
    for (size_t i = 0; i < std::min(scans, seq.size()) && r.is_open();) {
        const auto t0 = clk::now();

        // 1. The scan.
        matrix points = seq.points(i);                                       // [N, 4] x, y, z, remission
        matrix xyz    = points[R(), R(0, 3)];                                // [N, 3]
        matrix range  = matrix::sqrt((xyz * xyz).sum(-1));                   // [N]

        // 2. Projection: each point's pixel, the closest point per pixel, the range image.
        matrix pixel  = drishti::spherical_projection(xyz, sensor);          // [N] Int32, H*W = no pixel
        matrix owner  = drishti::closest_per_pixel(pixel, range, sensor);    // [H*W] Int32, -1 = empty
        matrix image  = drishti::range_image(points, range, owner, sensor);  // [1, H, W, 5] in metres

        // 3. The network: normalises the image, runs MLX. The first step that evaluates anything.
        matrix probs  = net.inference(image);                                // [1, H, W, 20]
        const double net_ms = ms_since(t0);

        // 4. Back to points: a class and a group per point.
        matrix cls    = drishti::to_points(probs.argmax(-1), pixel);         // [N] Int32
        matrix group  = groups.take(cls, 0);                                 // [N] Int32

        // 5. The map.
        matrix cells  = grid.bin(points, group, cls);                        // [G, 5] zmin, zmax, count, group, class

        // 6. Which points can make an object: its class, and the point its pixel shows (the
        //    points behind only inherit the class: the wall behind a car becomes "car").
        matrix owns       = drishti::owns_pixel(pixel, owner);                // [N] Bool
        matrix car_mask   = matrix::logical_and(matrix::equal(cls, matrix::scalar(kCar, dtype::Int32)), owns);     // [N] Bool
        matrix ped_mask   = matrix::logical_and(matrix::equal(cls, matrix::scalar(kPerson, dtype::Int32)), owns);
        matrix per_pixel  = probs.reshape((int)sensor.pixels(), net.classes()); // [H*W, 20]
        matrix car_prob   = drishti::to_points(per_pixel[R(), R(kCar, kCar + 1)], pixel);         // [N] Float
        matrix ped_prob   = drishti::to_points(per_pixel[R(), R(kPerson, kPerson + 1)], pixel);

        // Everything since the network is still graph: run what the CPU reads next.
        matrix::eval(cells, car_mask, ped_mask, car_prob, ped_prob);

        // 7. Objects: split each class's points into instances (bird's-eye-view cells, touching
        //    cells joined), then one box per instance ([B, 9]: centre, size, yaw, points, score),
        //    dropping instances that are too small, too thin, or the network wasn't sure about.
        matrix car_id     = drishti::cluster(xyz, car_mask, car_cell);        // [N] Int32, -1 = not kept
        matrix ped_id     = drishti::cluster(xyz, ped_mask, person_cell);
        matrix car_box    = drishti::fit_boxes(xyz, car_id, car_prob, car_params);
        matrix ped_box    = drishti::fit_boxes(xyz, ped_id, ped_prob, person_params);

        // 8. Tracks in the world frame (the pose removes our own motion; 10 Hz): + track id, speed.
        matrix car_tracked = cars.update(car_box, seq.pose(i), i * 0.1);     // [B, 11]
        matrix ped_tracked = people.update(ped_box, seq.pose(i), i * 0.1);
        const double total_ms = ms_since(t0);

        // ---- draw ----
        if (SHOW_GRID) {
            const auto by = BY_CLASS ? drishti::draw::TileColour::Class : drishti::draw::TileColour::Group;
            drishti::draw::Mesh tiles = drishti::draw::tiles(grid, cells, palette, by);
            if (tiles.count) grid_mesh.update(tiles.vertices, tiles.colours, tiles.indices);
            else grid_mesh.clear();
        }
        if (SHOW_POINTS) {
            drishti::draw::Cloud c = drishti::draw::points(points, BY_CLASS ? cls : group, palette);
            cloud.update(c.positions, c.colours);
        }
        car_boxes.update(drishti::draw::boxes(car_tracked), drishti::draw::speed_texts(car_tracked, "CAR", 1.5f, "STOPPED"));
        person_boxes.update(drishti::draw::boxes(ped_tracked), drishti::draw::speed_texts(ped_tracked, "PERSON", 0.4f, "STANDING"));
        std::printf("scan %zu: %u points, %u cars, %u people | network %.1f ms, all %.1f ms\n", i,
                    (unsigned)points.shape()[0], (unsigned)car_tracked.shape()[0], (unsigned)ped_tracked.shape()[0], net_ms, total_ms);

        // 10 Hz, the sensor's rate: keep drawing and handling keys until the next scan is due.
        do {
            const int key = r.wait_key(std::max(100 - (int)ms_since(t0), 1));
            if (key == 27) r.close();
            if (key == ' ') paused = !paused;
        } while (r.is_open() && (paused || ms_since(t0) < 100.0));
        ++i;
    }
    if (!png.empty() && r.is_open()) r.save(png);
    return 0;
}
