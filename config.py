"""Central configuration for the ground-normalized panicle pipeline.

All paths are resolved from this file so the scripts do not depend on the
caller's current working directory.  Importing this module has no filesystem
side effects; each pipeline stage creates only the directories it needs.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_LAS = PROJECT_DIR / "data" / "input.las"

OUTPUT_DIR = PROJECT_DIR / "results_ground_normalized"
FIGURE_DIR = OUTPUT_DIR / "figures"
INSTANCE_DIR = OUTPUT_DIR / "instances"

VOXEL_FILE = OUTPUT_DIR / "voxels.npz"
SEED_FILE = OUTPUT_DIR / "maps_and_seeds.npz"
RAW_LABEL_FILE = OUTPUT_DIR / "voxel_labels_raw.npy"


# Shared spatial resolution.
VOXEL_SIZE = 0.008
GRID_SIZE = VOXEL_SIZE


# Ground model and above-ground filtering.
GROUND_GRID_SIZE_M = 0.20
GROUND_CELL_PERCENTILE = 5.0
GROUND_RANSAC_TOLERANCE_M = 0.02
GROUND_RESIDUAL_LIMIT_M = 0.04
GROUND_MEDIAN_FILTER_SIZE = 3
GROUND_CLEARANCE_M = 0.08
GROUND_MAX_SAMPLE_POINTS = 2_000_000
GROUND_MIN_CELL_POINTS = 50
GROUND_RANSAC_ITERATIONS = 1_000
GROUND_LOWER_OUTLIER_PERCENTILE = 0.1
GROUND_RANSAC_SEED = 42


# Stage 02: 2-D top seed detection.
SMOOTH_SIGMA_M = 0.016
PEAK_MIN_DISTANCE_M = 0.040
PEAK_HEIGHT_PERCENTILE = 35
MIN_LOCAL_SUPPORT = 0.10
MIN_BASIN_CELLS = 12


# Stage 03: directed 3-D propagation.
UPWARD_PENALTY = 4.0
WATERSHED_CROSS_PENALTY = 3.0
DOWNWARD_VERTICAL_COST = 0.35
MAX_VERTICAL_GAP_VOXELS = 4
MAX_PATH_COST = 120.0
MAX_FALLBACK_DROP_M = 1.8
MAX_FALLBACK_RISE_M = 0.03


# Stage 04: instance filtering and optional PLY exports.
MIN_INSTANCE_VOXELS = 80
MIN_INSTANCE_POINTS = 500
EXPORT_INSTANCES_COLORED = True
EXPORT_INSTANCES_WITH_CONTEXT = True
EXPORT_INDIVIDUAL_INSTANCES = True

# Compatibility aliases for the initial plan terminology.
EXPORT_CONTEXT_PLY = EXPORT_INSTANCES_WITH_CONTEXT
EXPORT_INSTANCE_PLY = EXPORT_INDIVIDUAL_INSTANCES
