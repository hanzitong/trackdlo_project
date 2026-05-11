# preprocessing

Standalone C++ library — no ROS dependency.

Camera image → segmented 3D point cloud pipeline, and node visibility computation.
Produces the inputs required by `pure_trackdlo`.

## Dependencies

- Eigen3
- OpenCV
- PCL 1.8

## API

```cpp
#include "preprocessing.h"

// Step 1: extract cable pixels from BGR image
cv::Mat mask = color_threshold(bgr_image, lower_hsv, upper_hsv);

// Step 2: unproject masked pixels to 3D using pinhole model + voxel downsampling
Eigen::MatrixXd X = images_to_pointcloud(bgr_image, depth_image, proj_matrix, mask);

// Step 3: determine which nodes are visible in the current frame
std::vector<int> visible_nodes, visible_nodes_extended;
compute_visible_nodes(Y, X, proj_matrix, geodesic_coord,
                      img_rows, img_cols,
                      visibility_threshold, d_vis, dlo_pixel_width,
                      visible_nodes, visible_nodes_extended);
```

## Build

```bash
mkdir build && cd build
cmake .. && make
ctest --output-on-failure   # 5 tests
```
