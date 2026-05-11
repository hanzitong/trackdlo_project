# evaluation

Standalone C++ library — no ROS dependency.

Benchmark tools for measuring TrackDLO tracking accuracy against ground truth.

## Dependencies

- Eigen3
- OpenCV
- PCL 1.8
- pure_trackdlo (geometry helpers)

## API

```cpp
#include "evaluator.h"

evaluator eval(length, trial, pct_occlusion, alg, bag_file,
               save_location, start_record_at, exit_at,
               wait_before_occlusion, bag_rate, num_of_nodes);

// Detect ground-truth nodes from marker colors in RGB image
Eigen::MatrixXd Y_true = eval.get_ground_truth_nodes(rgb_image, cloud);

// Compute bidirectional piecewise Hausdorff error
double error = eval.compute_error(Y_tracked, Y_true);

// Compute and append error to a text file
eval.compute_and_save_error(Y_tracked, Y_true);
```

## Build

```bash
mkdir build && cd build
cmake .. && make
```

> No automated tests — ground truth data (rosbag) is required to run the evaluator.
