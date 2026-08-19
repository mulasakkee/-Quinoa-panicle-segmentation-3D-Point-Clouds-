import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from runtime_env import configure_runtime_environment

configure_runtime_environment()

import numpy as np

from ground_model import build_ground_model, interpolate_ground_z


class GroundModelTests(unittest.TestCase):
    def test_recovers_sloped_ground_with_canopy_and_low_outliers(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(0.0, 2.0, 40_000)
        y = rng.uniform(0.0, 2.0, 40_000)
        true_ground = 0.30 - 0.004 * x - 0.002 * y
        z = true_ground + rng.normal(0.0, 0.004, len(x))

        canopy = rng.random(len(x)) < 0.55
        z[canopy] += rng.uniform(0.4, 1.3, np.count_nonzero(canopy))
        z[:12] = -4.0
        xyz = np.column_stack((x, y, z))

        origin, grid, diagnostics = build_ground_model(
            xyz,
            grid_size=0.20,
            cell_percentile=5.0,
            ransac_tolerance=0.02,
            residual_limit=0.04,
            median_filter_size=3,
            max_sample_points=40_000,
            min_cell_points=20,
            ransac_iterations=500,
            random_seed=42,
        )

        query = np.column_stack((x[::100], y[::100]))
        estimated = interpolate_ground_z(query, origin, 0.20, grid)
        expected = 0.30 - 0.004 * query[:, 0] - 0.002 * query[:, 1]
        rmse = float(np.sqrt(np.mean((estimated - expected) ** 2)))

        self.assertLess(rmse, 0.03)
        self.assertGreater(diagnostics["candidate_cell_count"], 50)
        self.assertGreater(diagnostics["accepted_cell_count"], 20)
        self.assertTrue(np.all(np.isfinite(grid)))

    def test_interpolation_clips_queries_to_grid_boundary(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        query = np.array([[-10.0, -10.0], [10.0, 10.0], [0.5, 0.5]])
        result = interpolate_ground_z(query, np.array([0.0, 0.0]), 1.0, grid)

        np.testing.assert_allclose(result, [1.0, 4.0, 2.5])


if __name__ == "__main__":
    unittest.main()
