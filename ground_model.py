"""Pure numerical helpers for estimating and querying a local ground DTM."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, generic_filter, median_filter


def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive value, got {value!r}.")


def _fit_plane_least_squares(xy_local: np.ndarray, z: np.ndarray) -> np.ndarray:
    design = np.column_stack((xy_local, np.ones(len(xy_local), dtype=np.float64)))
    coefficients, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    return coefficients


def _ransac_plane(
    xy_local: np.ndarray,
    z: np.ndarray,
    tolerance: float,
    iterations: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``z = ax + by + c`` and return refined coefficients and inliers."""

    candidate_count = len(z)
    if candidate_count < 3:
        raise ValueError(
            "At least three populated ground cells are required to fit a ground plane."
        )

    rng = np.random.default_rng(random_seed)
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_error = np.inf

    for _ in range(iterations):
        sample_indices = rng.choice(candidate_count, size=3, replace=False)
        sample_xy = xy_local[sample_indices]
        sample_z = z[sample_indices]
        design = np.column_stack((sample_xy, np.ones(3, dtype=np.float64)))

        determinant = np.linalg.det(design)
        if not np.isfinite(determinant) or abs(determinant) < 1e-12:
            continue

        coefficients = np.linalg.solve(design, sample_z)
        residuals = np.abs(z - (xy_local @ coefficients[:2] + coefficients[2]))
        inliers = residuals <= tolerance
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            continue

        inlier_error = float(np.median(residuals[inliers]))
        if inlier_count > best_count or (
            inlier_count == best_count and inlier_error < best_error
        ):
            best_inliers = inliers
            best_count = inlier_count
            best_error = inlier_error

    if best_inliers is None:
        raise ValueError("RANSAC could not find a non-degenerate ground plane.")

    coefficients = _fit_plane_least_squares(xy_local[best_inliers], z[best_inliers])
    for _ in range(3):
        residuals = np.abs(z - (xy_local @ coefficients[:2] + coefficients[2]))
        refined_inliers = residuals <= tolerance
        if np.count_nonzero(refined_inliers) < 3:
            break
        if np.array_equal(refined_inliers, best_inliers):
            best_inliers = refined_inliers
            break
        best_inliers = refined_inliers
        coefficients = _fit_plane_least_squares(xy_local[best_inliers], z[best_inliers])

    return coefficients.astype(np.float64, copy=False), best_inliers


def _nanmedian(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else np.nan


def build_ground_model(
    xyz: np.ndarray,
    *,
    grid_size: float,
    cell_percentile: float,
    ransac_tolerance: float,
    residual_limit: float,
    median_filter_size: int,
    max_sample_points: int,
    min_cell_points: int,
    ransac_iterations: int,
    random_seed: int = 42,
    lower_outlier_percentile: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate a local ground surface from an unclassified XYZ point cloud.

    The deterministic sample is clipped below its global 0.1th percentile,
    reduced to a low percentile per coarse XY cell, fitted with a RANSAC trend
    plane, and accepted within ``residual_limit`` of that plane.  Local
    residuals receive a NaN-aware median pass, nearest-neighbour hole filling,
    and a final median smoothing pass before they are added back to the trend.

    Returns ``(ground_xy_origin, ground_z_grid, diagnostics)``.  The origin is
    the XY coordinate of ``ground_z_grid[0, 0]`` (the first cell centre), and
    the grid is indexed as ``ground_z_grid[y_index, x_index]``.
    """

    xyz_array = np.asarray(xyz, dtype=np.float64)
    if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
        raise ValueError(f"xyz must have shape (n, 3), got {xyz_array.shape!r}.")
    if len(xyz_array) == 0:
        raise ValueError("xyz must contain at least one point.")

    _validate_positive("grid_size", grid_size)
    _validate_positive("ransac_tolerance", ransac_tolerance)
    _validate_positive("residual_limit", residual_limit)
    if not 0 <= cell_percentile <= 100:
        raise ValueError("cell_percentile must be between 0 and 100.")
    if not 0 <= lower_outlier_percentile < 100:
        raise ValueError("lower_outlier_percentile must be in [0, 100).")
    if median_filter_size < 1 or median_filter_size % 2 == 0:
        raise ValueError("median_filter_size must be a positive odd integer.")
    if max_sample_points < 3:
        raise ValueError("max_sample_points must be at least 3.")
    if min_cell_points < 1:
        raise ValueError("min_cell_points must be at least 1.")
    if ransac_iterations < 1:
        raise ValueError("ransac_iterations must be at least 1.")

    finite_mask = np.all(np.isfinite(xyz_array), axis=1)
    finite_xyz = xyz_array if np.all(finite_mask) else xyz_array[finite_mask]
    if len(finite_xyz) < 3:
        raise ValueError("xyz must contain at least three finite points.")

    finite_count = len(finite_xyz)
    sample_count = min(finite_count, int(max_sample_points))
    if sample_count == finite_count:
        sampled_xyz = finite_xyz
    else:
        sample_indices = (
            np.arange(sample_count, dtype=np.int64) * np.int64(finite_count)
        ) // np.int64(sample_count)
        sampled_xyz = finite_xyz[sample_indices]

    lower_clip_z = float(
        np.percentile(sampled_xyz[:, 2], lower_outlier_percentile)
    )
    sampled_xyz = sampled_xyz[sampled_xyz[:, 2] >= lower_clip_z]
    if len(sampled_xyz) < 3:
        raise ValueError("Too few points remain after removing low-Z outliers.")

    xy_min = finite_xyz[:, :2].min(axis=0)
    xy_max = finite_xyz[:, :2].max(axis=0)
    cell_corner_origin = np.floor(xy_min / grid_size) * grid_size
    grid_shape_xy = (
        np.floor((xy_max - cell_corner_origin) / grid_size).astype(np.int64) + 1
    )
    grid_width, grid_height = (int(grid_shape_xy[0]), int(grid_shape_xy[1]))
    if grid_width < 1 or grid_height < 1:
        raise ValueError("The point-cloud XY extent produced an invalid ground grid.")

    ground_xy_origin = cell_corner_origin + grid_size * 0.5
    sample_grid_xy = np.floor(
        (sampled_xyz[:, :2] - cell_corner_origin) / grid_size
    ).astype(np.int64)
    sample_grid_xy[:, 0] = np.clip(sample_grid_xy[:, 0], 0, grid_width - 1)
    sample_grid_xy[:, 1] = np.clip(sample_grid_xy[:, 1], 0, grid_height - 1)
    linear_cells = sample_grid_xy[:, 1] * grid_width + sample_grid_xy[:, 0]

    sort_order = np.argsort(linear_cells, kind="stable")
    sorted_cells = linear_cells[sort_order]
    sorted_z = sampled_xyz[sort_order, 2]
    unique_cells, starts, counts = np.unique(
        sorted_cells,
        return_index=True,
        return_counts=True,
    )

    cell_count_grid = np.zeros((grid_height, grid_width), dtype=np.int32)
    cell_count_grid.ravel()[unique_cells] = counts.astype(np.int32)
    cell_percentile_grid = np.full((grid_height, grid_width), np.nan, dtype=np.float64)

    populated = counts >= min_cell_points
    populated_cells = unique_cells[populated]
    populated_starts = starts[populated]
    populated_counts = counts[populated]
    for cell, start, count in zip(populated_cells, populated_starts, populated_counts):
        cell_percentile_grid.ravel()[cell] = np.percentile(
            sorted_z[start : start + count],
            cell_percentile,
        )

    candidate_mask = np.isfinite(cell_percentile_grid)
    candidate_y, candidate_x = np.nonzero(candidate_mask)
    candidate_z = cell_percentile_grid[candidate_mask]
    candidate_xy_local = np.column_stack(
        (candidate_x * grid_size, candidate_y * grid_size)
    )
    candidate_cell_count = len(candidate_z)
    if candidate_cell_count < 3:
        raise ValueError(
            "Fewer than three cells meet min_cell_points; increase the ground "
            "grid size or lower GROUND_MIN_CELL_POINTS."
        )

    plane_coefficients, ransac_inliers = _ransac_plane(
        candidate_xy_local,
        candidate_z,
        ransac_tolerance,
        ransac_iterations,
        random_seed,
    )

    x_local = np.arange(grid_width, dtype=np.float64) * grid_size
    y_local = np.arange(grid_height, dtype=np.float64) * grid_size
    plane_grid = (
        plane_coefficients[0] * x_local[None, :]
        + plane_coefficients[1] * y_local[:, None]
        + plane_coefficients[2]
    )
    candidate_plane_z = plane_grid[candidate_mask]
    candidate_residuals = candidate_z - candidate_plane_z
    accepted_candidates = np.abs(candidate_residuals) <= residual_limit

    accepted_mask = np.zeros_like(candidate_mask)
    accepted_mask[candidate_y[accepted_candidates], candidate_x[accepted_candidates]] = True
    accepted_cell_count = int(np.count_nonzero(accepted_mask))
    if accepted_cell_count == 0:
        raise ValueError("No ground cells remain within residual_limit of the RANSAC plane.")

    residual_grid = np.full_like(cell_percentile_grid, np.nan)
    residual_grid[accepted_mask] = (
        cell_percentile_grid[accepted_mask] - plane_grid[accepted_mask]
    )
    median_residual_grid = generic_filter(
        residual_grid,
        _nanmedian,
        size=median_filter_size,
        mode="nearest",
    )
    median_residual_grid[~accepted_mask] = np.nan

    valid_residual_mask = np.isfinite(median_residual_grid)
    if not np.any(valid_residual_mask):
        median_residual_grid = residual_grid.copy()
        valid_residual_mask = np.isfinite(median_residual_grid)

    nearest_indices = distance_transform_edt(
        ~valid_residual_mask,
        return_distances=False,
        return_indices=True,
    )
    filled_residual_grid = median_residual_grid[tuple(nearest_indices)]
    smoothed_residual_grid = median_filter(
        filled_residual_grid,
        size=median_filter_size,
        mode="nearest",
    )
    ground_z_grid = (plane_grid + smoothed_residual_grid).astype(np.float64, copy=False)

    ransac_inlier_mask = np.zeros_like(candidate_mask)
    ransac_inlier_mask[candidate_y[ransac_inliers], candidate_x[ransac_inliers]] = True
    diagnostics: dict[str, Any] = {
        "finite_point_count": int(finite_count),
        "sample_point_count": int(sample_count),
        "retained_sample_point_count": int(len(sampled_xyz)),
        "lower_clip_z": lower_clip_z,
        "candidate_cell_count": int(candidate_cell_count),
        "inlier_cell_count": int(np.count_nonzero(ransac_inliers)),
        "accepted_cell_count": accepted_cell_count,
        "ground_z_min": float(np.min(ground_z_grid)),
        "ground_z_max": float(np.max(ground_z_grid)),
        "plane_coefficients_local": plane_coefficients,
        "cell_count_grid": cell_count_grid,
        "cell_percentile_grid": cell_percentile_grid,
        "candidate_cell_mask": candidate_mask,
        "ransac_inlier_mask": ransac_inlier_mask,
        "accepted_cell_mask": accepted_mask,
    }
    return ground_xy_origin.astype(np.float64, copy=False), ground_z_grid, diagnostics


def interpolate_ground_z(
    xy: np.ndarray,
    origin: np.ndarray,
    grid_size: float,
    grid: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate ground Z, clipping queries to the DTM boundary."""

    _validate_positive("grid_size", grid_size)
    xy_array = np.asarray(xy, dtype=np.float64)
    if xy_array.ndim == 1:
        if xy_array.shape != (2,):
            raise ValueError(f"xy must have shape (n, 2) or (2,), got {xy_array.shape!r}.")
        original_shape: tuple[int, ...] = ()
        flat_xy = xy_array.reshape(1, 2)
    elif xy_array.ndim >= 2 and xy_array.shape[-1] == 2:
        original_shape = xy_array.shape[:-1]
        flat_xy = xy_array.reshape(-1, 2)
    else:
        raise ValueError(f"xy must have shape (n, 2) or (2,), got {xy_array.shape!r}.")
    if not np.all(np.isfinite(flat_xy)):
        raise ValueError("xy contains non-finite coordinates.")

    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (2,) or not np.all(np.isfinite(origin_array)):
        raise ValueError("origin must contain exactly two finite coordinates.")
    grid_array = np.asarray(grid, dtype=np.float64)
    if grid_array.ndim != 2 or grid_array.shape[0] < 1 or grid_array.shape[1] < 1:
        raise ValueError("grid must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(grid_array)):
        raise ValueError("grid must contain only finite ground elevations.")

    grid_x = np.clip(
        (flat_xy[:, 0] - origin_array[0]) / grid_size,
        0.0,
        grid_array.shape[1] - 1,
    )
    grid_y = np.clip(
        (flat_xy[:, 1] - origin_array[1]) / grid_size,
        0.0,
        grid_array.shape[0] - 1,
    )

    x0 = np.floor(grid_x).astype(np.int64)
    y0 = np.floor(grid_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, grid_array.shape[1] - 1)
    y1 = np.minimum(y0 + 1, grid_array.shape[0] - 1)
    fx = grid_x - x0
    fy = grid_y - y0

    z00 = grid_array[y0, x0]
    z10 = grid_array[y0, x1]
    z01 = grid_array[y1, x0]
    z11 = grid_array[y1, x1]
    interpolated = (
        z00 * (1.0 - fx) * (1.0 - fy)
        + z10 * fx * (1.0 - fy)
        + z01 * (1.0 - fx) * fy
        + z11 * fx * fy
    )
    return interpolated.reshape(original_shape)


__all__ = ["build_ground_model", "interpolate_ground_z"]
