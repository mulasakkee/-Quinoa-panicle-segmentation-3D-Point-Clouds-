# Input data

Place the point cloud to process at `data/input.las`.

Input requirements:

- XYZ coordinates use metres.
- The positive Z axis points upward.
- The scene contains enough visible ground for local ground fitting.
- RGB attributes are optional.

Raw point clouds and generated intermediate files are intentionally excluded
from Git. Do not reuse intermediate results produced from another point cloud:
use a new empty output directory (or clear the previous pipeline output) before
processing a different `input.las`.
