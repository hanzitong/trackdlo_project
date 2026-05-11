# pure_trackdlo

Standalone C++ library — no ROS dependency.

TrackDLO algorithm for tracking deformable linear objects (cables) in 3D space.
Takes a segmented point cloud and returns node coordinates frame by frame.

## Dependencies

- Eigen3 (matrix operations)

## API

```cpp
#include "trackdlo.h"

// Initialize
TrackdloParams params;          // tuning parameters (beta, lambda, mu, ...)
TrackdloState  state = make_trackdlo_state(num_nodes);
state.Y = initial_nodes;        // Eigen::MatrixXd (M×3)
state.geodesic_coord = ...;     // arc-length of each node

// Per-frame call
tracking_step(state, X, visible_nodes, visible_nodes_extended, params);
// state.Y is updated in-place
```

## Build

```bash
mkdir build && cd build
cmake .. && make
ctest --output-on-failure   # 17 tests
```
