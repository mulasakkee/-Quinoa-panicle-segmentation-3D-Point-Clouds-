from runtime_env import configure_runtime_environment

configure_runtime_environment()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from config import (
    FIGURE_DIR,
    MIN_BASIN_CELLS,
    MIN_LOCAL_SUPPORT,
    OUTPUT_DIR,
    PEAK_HEIGHT_PERCENTILE,
    PEAK_MIN_DISTANCE_M,
    SEED_FILE,
    SMOOTH_SIGMA_M,
    VOXEL_FILE,
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"读取地面归一化体素：{VOXEL_FILE}", flush=True)
    data = np.load(VOXEL_FILE)
    voxel_size = float(data["voxel_size"])
    grid_shape = data["grid_shape"].astype(np.int64)
    voxel_grid = data["voxel_grid"]
    voxel_z_max_agl = data["voxel_z_max_agl"]
    above_ground = data["voxel_above_ground_mask"].astype(bool)
    ground_clearance_m = float(data["ground_clearance_m"])

    if len(voxel_grid) != len(voxel_z_max_agl) or len(voxel_grid) != len(above_ground):
        raise ValueError("体素坐标、AGL 高度和非地面掩膜长度不一致。")
    if not np.any(above_ground):
        raise ValueError("没有体素高于地面阈值，请检查地面模型或 GROUND_CLEARANCE_M。")

    sigma = SMOOTH_SIGMA_M / voxel_size
    min_distance = max(1, round(PEAK_MIN_DISTANCE_M / voxel_size))
    grid_width, grid_height = int(grid_shape[0]), int(grid_shape[1])
    cell_count = grid_width * grid_height

    candidate_indices = np.flatnonzero(above_ground).astype(np.int32)
    candidate_xy = voxel_grid[candidate_indices, :2]
    candidate_linear = candidate_xy[:, 1] * grid_width + candidate_xy[:, 0]

    height_flat = np.full(cell_count, -np.inf, dtype=np.float32)
    np.maximum.at(height_flat, candidate_linear, voxel_z_max_agl[candidate_indices])
    height_map = height_flat.reshape(grid_height, grid_width)

    density_map = np.bincount(
        candidate_linear,
        minlength=cell_count,
    ).reshape(grid_height, grid_width)
    valid_mask = density_map > 0

    height_values = np.where(valid_mask, height_map, 0.0)
    local_support = gaussian_filter(valid_mask.astype(np.float32), sigma=sigma)
    smoothed_height = gaussian_filter(height_values, sigma=sigma)
    smoothed_height = np.divide(
        smoothed_height,
        local_support,
        out=np.full_like(smoothed_height, np.nan),
        where=local_support > 0,
    )
    watershed_mask = valid_mask & (local_support >= MIN_LOCAL_SUPPORT)
    if not np.any(watershed_mask):
        raise ValueError("非地面候选区域为空，请检查地面阈值和 MIN_LOCAL_SUPPORT。")

    threshold = np.percentile(
        smoothed_height[watershed_mask],
        PEAK_HEIGHT_PERCENTILE,
    )
    peak_image = np.where(watershed_mask, smoothed_height, -np.inf)
    seed_yx = peak_local_max(
        peak_image,
        min_distance=min_distance,
        threshold_abs=threshold,
        labels=watershed_mask.astype(np.uint8),
        exclude_border=False,
    )

    if len(seed_yx) == 0:
        raise ValueError("未检测到顶部种子，请检查 AGL 掩膜或峰值参数。")

    markers = np.zeros_like(height_map, dtype=np.int32)
    markers[tuple(seed_yx.T)] = np.arange(1, len(seed_yx) + 1)
    watershed_map = watershed(
        -np.nan_to_num(smoothed_height),
        markers,
        mask=watershed_mask,
    )

    basin_sizes = np.bincount(
        watershed_map.ravel(),
        minlength=len(seed_yx) + 1,
    )
    seed_yx = seed_yx[basin_sizes[1:] >= MIN_BASIN_CELLS]
    if len(seed_yx) == 0:
        raise ValueError("所有种子的二维分水岭区域都过小。")

    markers.fill(0)
    markers[tuple(seed_yx.T)] = np.arange(1, len(seed_yx) + 1)
    watershed_map = watershed(
        -np.nan_to_num(smoothed_height),
        markers,
        mask=watershed_mask,
    )

    top_voxel_flat = np.full(cell_count, -1, dtype=np.int32)
    candidate_top_mask = (
        voxel_z_max_agl[candidate_indices]
        == height_flat[candidate_linear]
    )
    top_voxel_flat[candidate_linear[candidate_top_mask]] = candidate_indices[
        candidate_top_mask
    ]
    seed_linear = seed_yx[:, 0] * grid_width + seed_yx[:, 1]
    seed_voxel_indices = top_voxel_flat[seed_linear]
    if np.any(seed_voxel_indices < 0):
        raise ValueError("部分二维种子没有对应的非地面顶部体素。")
    if not np.all(above_ground[seed_voxel_indices]):
        raise AssertionError("检测到的种子包含地面体素。")

    np.savez(
        SEED_FILE,
        grid_size=voxel_size,
        height_reference="AGL",
        ground_clearance_m=ground_clearance_m,
        height_map=height_map,
        density_map=density_map,
        smoothed_height=smoothed_height,
        watershed_mask=watershed_mask,
        watershed_map=watershed_map,
        seed_yx=seed_yx,
        seed_voxel_indices=seed_voxel_indices,
    )

    print(f"高度图尺寸：{grid_width} × {grid_height}", flush=True)
    print(
        f"非地面候选体素：{len(candidate_indices):,}；"
        f"离地阈值：{ground_clearance_m:.2f} m",
        flush=True,
    )
    print(f"有效顶部种子数：{len(seed_yx)}", flush=True)
    print(f"核心结果已保存：{SEED_FILE}", flush=True)

    extent = [0, grid_width * voxel_size, 0, grid_height * voxel_size]
    vmin, vmax = np.percentile(smoothed_height[watershed_mask], [1, 99])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    height_image = axes[0].imshow(
        smoothed_height,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Smoothed height above ground")
    axes[0].set_xlabel("Local X (m)")
    axes[0].set_ylabel("Local Y (m)")
    fig.colorbar(height_image, ax=axes[0], label="Height AGL (m)")

    density_image = axes[1].imshow(
        np.log1p(density_map),
        origin="lower",
        extent=extent,
    )
    axes[1].set_title("Above-ground voxel density")
    axes[1].set_xlabel("Local X (m)")
    axes[1].set_ylabel("Local Y (m)")
    fig.colorbar(density_image, ax=axes[1], label="log(1 + voxel count)")
    fig.tight_layout()
    height_figure = FIGURE_DIR / "01_height_density.png"
    fig.savefig(height_figure, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(
        smoothed_height,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].scatter(
        (seed_yx[:, 1] + 0.5) * voxel_size,
        (seed_yx[:, 0] + 0.5) * voxel_size,
        s=10,
        c="red",
    )
    axes[0].set_title(f"Top seeds above ground: {len(seed_yx)}")
    axes[0].set_xlabel("Local X (m)")
    axes[0].set_ylabel("Local Y (m)")

    axes[1].imshow(
        watershed_map,
        origin="lower",
        extent=extent,
        cmap="nipy_spectral",
    )
    axes[1].set_title("2D marker-controlled watershed")
    axes[1].set_xlabel("Local X (m)")
    axes[1].set_ylabel("Local Y (m)")
    fig.tight_layout()
    watershed_figure = FIGURE_DIR / "02_seeds_watershed.png"
    fig.savefig(watershed_figure, dpi=220)
    plt.close(fig)

    print(f"诊断图已保存：{height_figure}", flush=True)
    print(f"诊断图已保存：{watershed_figure}", flush=True)


if __name__ == "__main__":
    main()
