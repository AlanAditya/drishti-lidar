# drishti

LiDAR perception for self-driving on Apple silicon: semantic segmentation, a variable-resolution 2.5D map, and tracked 3D boxes around cars and people, from raw scans of a spinning LiDAR. Built on [matrix](https://github.com/AlanAditya/matrix) (a lazy, GPU-accelerated tensor library for Metal), with [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext) running in [MLX](https://github.com/ml-explore/mlx) for the segmentation.

*Drishti* (दृष्टि) is Sanskrit for sight.

Every step is a function or object with matrices in and out, and its settings in the open. A program runs the steps itself: in its own order, reading or changing what passes between them, swapping one out, and choosing where evaluation happens.

## The steps

| Step | In | Out |
|---|---|---|
| `KittiSequence::points` | a SemanticKITTI scan | `[N, 4]` x, y, z, remission |
| `spherical_projection` | `xyz [N, 3]`, `Sensor` | `[N]` each point's pixel (`H*W` = none) |
| `closest_per_pixel` | pixel, range | `[H*W]` the point each pixel shows (`-1` = empty) |
| `range_image` | points, range, owner | `[1, H, W, 5]` range, x, y, z, remission in metres |
| `SalsaNext::inference` | range image | `[1, H, W, 20]` class probabilities |
| `to_points` | per-pixel values, pixel | `[N]` each point's pixel's value |
| `owns_pixel` | pixel, owner | `[N]` whether the point is the one its pixel shows |
| `FoveatedGrid::bin` | points, group, class | `[G, 5]` zmin, zmax, count, group, class per cell |
| `cluster` | xyz, mask | `[N]` object id per point (`-1` = not kept) |
| `fit_boxes` | xyz, id, probability | `[B, 9]` oriented boxes (L-shape fit) |
| `Tracker::update` | boxes, pose, time | `[B, 11]` + track id and speed |

The foveated grid is fine near the car and coarse far away: 5 cm cells to 10 m, 10 cm to 20 m, 20 cm to 40 m, 40 cm to 100 m. That covers 200 m × 200 m in 610,000 cells (7.3 MB), where a uniform 5 cm grid would need 16 million.

The graph steps (projection, owner, image, `to_points`, `owns_pixel`, the grid) only build graph. Nothing runs until `matrix::eval`, or a step that reads on the CPU: `inference` (its image goes into MLX), `cluster`, `fit_boxes` and the tracker. [`main.cpp`](main.cpp) runs all of them, one after the other, for every scan.

## Requirements

- A Mac with Apple silicon, Xcode (for the Metal compiler: `xcrun metal`) and CMake 3.20 or later.
- MLX's C++ library with its CMake package. Build it from [source](https://github.com/ml-explore/mlx) and `cmake --install` it, so that `find_package(MLX)` works.

## Data and weights

**Scans.** `tools/fetch_kitti_subset.py` downloads part of one SemanticKITTI sequence (scans, labels, poses) without the 80 GB zip, a couple of MB per scan:

```bash
python3 tools/fetch_kitti_subset.py ~/datasets/SemanticKITTI/dataset 08 0 500
```

The KITTI odometry scans are © KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute); the labels are [SemanticKITTI](http://www.semantic-kitti.org)'s. Check both licences before using them.

**Weights.** Download the pretrained SalsaNext model from [the SalsaNext repository](https://github.com/TiagoCortinhal/SalsaNext) and convert it to MLX (needs PyTorch and MLX for Python):

```bash
python3 tools/convert_salsanext.py <folder with the SalsaNext checkpoint>
```

It writes `salsanext.safetensors` next to the checkpoint, after checking that the MLX network reproduces PyTorch's output.

## Build and run

```bash
git clone --recursive https://github.com/AlanAditya/drishti-lidar.git
cd drishti-lidar
cmake -S . -B build
cmake --build build
./build/drishti ~/datasets/SemanticKITTI/dataset/sequences/08 <path>/salsanext.safetensors
```

Space pauses, Esc quits. `--scans N` stops after N scans, and `--save frame.png` writes the last frame to a PNG.

To use drishti in your own program, link the `drishti` CMake target (it puts `drishti/` on the include path) and `#include "drishti.cpp"` from your one source file, compiled as Objective-C++ (matrix's headers use Metal). `-DDRISHTI_DEMO=OFF` skips the demo and its renderer.

## Layout

| Path | What |
|---|---|
| `drishti/drishti.cpp` | the API |
| `drishti/FoveatedGrid.metal` | the grid's two Metal kernels |
| `drishti/*.h` | the pieces it builds on: the KITTI reader, SalsaNext in MLX, the grid layout, box fitting, tracking |
| `main.cpp`, `drishti_draw.cpp` | the demo, and its helpers that turn drishti's matrices into renderer input |
| `renderer/` | the demo renderer, including its SOPs, camera and Metal shaders |
| `matrix/` | the matrix library (git submodule) |
| `tools/` | the KITTI download and the weights conversion |

drishti is mirrored from a larger project, [WorldOf3D](https://github.com/AlanAditya/WorldOf3D), by a script, so its history only has the commits that touched these files. Make changes there, not here.
