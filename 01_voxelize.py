from runtime_env import configure_runtime_environment

configure_runtime_environment()

import matplotlib

matplotlib.use("Agg")

import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from config import (
    FIGURE_DIR,
    GROUND_CELL_PERCENTILE,
    GROUND_CLEARANCE_M,
    GROUND_GRID_SIZE_M,
    GROUND_LOWER_OUTLIER_PERCENTILE,
    GROUND_MAX_SAMPLE_POINTS,
    GROUND_MEDIAN_FILTER_SIZE,
    GROUND_MIN_CELL_POINTS,
    GROUND_RANSAC_ITERATIONS,
    GROUND_RANSAC_SEED,
    GROUND_RANSAC_TOLERANCE_M,
    GROUND_RESIDUAL_LIMIT_M,
    INPUT_LAS,
    OUTPUT_DIR,
    VOXEL_FILE,
    VOXEL_SIZE,
)
from ground_model import build_ground_model, interpolate_ground_z


def read_rgb(las):
    dimensions = set(las.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(dimensions):
        rgb = np.column_stack((las.red, las.green, las.blue))
        rgb_scale = 65535 if rgb.max() > 255 else 255
        return rgb.astype(np.float32) / rgb_scale

    print("警告：LAS 没有 RGB 字段，体素 PLY 将使用灰色。", flush=True)
    return np.full((len(las.points), 3), 0.65, dtype=np.float32)


def count_points_above_ground(xyz, ground_origin, ground_grid, chunk_size=1_000_000):
    above_count = 0
    for start in range(0, len(xyz), chunk_size):
        end = min(start + chunk_size, len(xyz))
        ground_z = interpolate_ground_z(
            xyz[start:end, :2],
            ground_origin,
            GROUND_GRID_SIZE_M,
            ground_grid,
        )
        above_count += np.count_nonzero(
            xyz[start:end, 2] - ground_z > GROUND_CLEARANCE_M
        )
    return int(above_count)


def save_ground_diagnostic(ground_grid, voxel_height_agl, above_ground_mask):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    image = axes[0].imshow(ground_grid, origin="lower", cmap="terrain")
    axes[0].set_title("Estimated local ground surface")
    axes[0].set_xlabel("Ground-grid X cell")
    axes[0].set_ylabel("Ground-grid Y cell")
    fig.colorbar(image, ax=axes[0], label="Ground Z (m)")

    finite_agl = voxel_height_agl[np.isfinite(voxel_height_agl)]
    upper = max(0.2, float(np.percentile(finite_agl, 99.5)))
    axes[1].hist(
        np.clip(finite_agl, -0.15, upper),
        bins=180,
        color="#4472C4",
    )
    axes[1].axvline(
        GROUND_CLEARANCE_M,
        color="red",
        linestyle="--",
        label=f"clearance = {GROUND_CLEARANCE_M:.2f} m",
    )
    axes[1].set_title(
        f"Voxel height AGL; retained {above_ground_mask.mean() * 100:.1f}%"
    )
    axes[1].set_xlabel("Height above ground (m)")
    axes[1].set_ylabel("Voxel count")
    axes[1].legend()

    fig.tight_layout()
    target = FIGURE_DIR / "00_ground_model.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    return target


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"读取 LAS：{INPUT_LAS}", flush=True)
    las = laspy.read(INPUT_LAS)
    xyz = np.column_stack((las.x, las.y, las.z))

    if len(xyz) == 0:
        raise ValueError("输入 LAS 不包含点。")

    print("估计局部地面模型并计算离地高度（AGL）……", flush=True)
    ground_origin, ground_grid, ground_diagnostics = build_ground_model(
        xyz,
        grid_size=GROUND_GRID_SIZE_M,
        cell_percentile=GROUND_CELL_PERCENTILE,
        ransac_tolerance=GROUND_RANSAC_TOLERANCE_M,
        residual_limit=GROUND_RESIDUAL_LIMIT_M,
        median_filter_size=GROUND_MEDIAN_FILTER_SIZE,
        max_sample_points=GROUND_MAX_SAMPLE_POINTS,
        min_cell_points=GROUND_MIN_CELL_POINTS,
        ransac_iterations=GROUND_RANSAC_ITERATIONS,
        random_seed=GROUND_RANSAC_SEED,
        lower_outlier_percentile=GROUND_LOWER_OUTLIER_PERCENTILE,
    )

    print(f"以 {VOXEL_SIZE * 1000:.0f} mm 分辨率进行体素化……", flush=True)
    origin = xyz.min(axis=0)
    point_voxel_coords = np.floor((xyz - origin) / VOXEL_SIZE).astype(np.int32)
    grid_shape = point_voxel_coords.max(axis=0) + 1

    point_voxel_keys = np.ravel_multi_index(
        point_voxel_coords.T,
        tuple(grid_shape),
    )
    voxel_keys, point_to_voxel = np.unique(
        point_voxel_keys,
        return_inverse=True,
    )
    point_to_voxel = point_to_voxel.astype(np.int32)
    voxel_count = len(voxel_keys)

    voxel_grid = np.column_stack(
        np.unravel_index(voxel_keys, tuple(grid_shape))
    ).astype(np.int32)
    del point_voxel_coords, point_voxel_keys
    voxel_point_count = np.bincount(
        point_to_voxel,
        minlength=voxel_count,
    ).astype(np.int32)

    voxel_xyz = np.column_stack((
        np.bincount(point_to_voxel, weights=xyz[:, 0], minlength=voxel_count),
        np.bincount(point_to_voxel, weights=xyz[:, 1], minlength=voxel_count),
        np.bincount(point_to_voxel, weights=xyz[:, 2], minlength=voxel_count),
    ))
    voxel_xyz /= voxel_point_count[:, None]

    voxel_z_max = np.full(voxel_count, -np.inf)
    np.maximum.at(voxel_z_max, point_to_voxel, xyz[:, 2])

    rgb = read_rgb(las)
    voxel_rgb = np.column_stack((
        np.bincount(point_to_voxel, weights=rgb[:, 0], minlength=voxel_count),
        np.bincount(point_to_voxel, weights=rgb[:, 1], minlength=voxel_count),
        np.bincount(point_to_voxel, weights=rgb[:, 2], minlength=voxel_count),
    ))
    voxel_rgb = (voxel_rgb / voxel_point_count[:, None]).astype(np.float32)
    del rgb

    voxel_ground_z = interpolate_ground_z(
        voxel_xyz[:, :2],
        ground_origin,
        GROUND_GRID_SIZE_M,
        ground_grid,
    )
    voxel_height_agl = voxel_xyz[:, 2] - voxel_ground_z
    voxel_z_max_agl = voxel_z_max - voxel_ground_z
    voxel_above_ground_mask = voxel_z_max_agl > GROUND_CLEARANCE_M

    if len(point_to_voxel) != len(xyz):
        raise AssertionError("point_to_voxel 与原始点数不一致。")
    if int(voxel_point_count.sum()) != len(xyz):
        raise AssertionError("体素点数总和与原始点数不一致。")
    if point_to_voxel.min() < 0 or point_to_voxel.max() >= voxel_count:
        raise AssertionError("point_to_voxel 超出体素索引范围。")

    source_stat = INPUT_LAS.stat()
    np.savez(
        VOXEL_FILE,
        voxel_size=VOXEL_SIZE,
        origin=origin,
        grid_shape=grid_shape,
        voxel_keys=voxel_keys,
        voxel_grid=voxel_grid,
        voxel_xyz=voxel_xyz,
        voxel_z_max=voxel_z_max,
        voxel_rgb=voxel_rgb,
        voxel_point_count=voxel_point_count,
        point_to_voxel=point_to_voxel,
        ground_xy_origin=np.asarray(ground_origin, dtype=np.float64),
        ground_grid_size=GROUND_GRID_SIZE_M,
        ground_z_grid=np.asarray(ground_grid, dtype=np.float32),
        ground_clearance_m=GROUND_CLEARANCE_M,
        voxel_ground_z=voxel_ground_z.astype(np.float32),
        voxel_height_agl=voxel_height_agl.astype(np.float32),
        voxel_z_max_agl=voxel_z_max_agl.astype(np.float32),
        voxel_above_ground_mask=voxel_above_ground_mask,
        source_point_count=np.int64(len(xyz)),
        source_xyz_min=xyz.min(axis=0),
        source_xyz_max=xyz.max(axis=0),
        source_file_size=np.int64(source_stat.st_size),
        source_mtime_ns=np.int64(source_stat.st_mtime_ns),
    )

    above_point_count = count_points_above_ground(
        xyz,
        ground_origin,
        ground_grid,
    )

    above_cloud = o3d.geometry.PointCloud()
    above_cloud.points = o3d.utility.Vector3dVector(voxel_xyz[voxel_above_ground_mask])
    above_cloud.colors = o3d.utility.Vector3dVector(voxel_rgb[voxel_above_ground_mask])
    voxel_ply = OUTPUT_DIR / "voxels_above_ground.ply"
    if not o3d.io.write_point_cloud(str(voxel_ply), above_cloud, write_ascii=False):
        raise OSError(f"无法写出点云：{voxel_ply}")

    diagnostic_path = save_ground_diagnostic(
        ground_grid,
        voxel_height_agl,
        voxel_above_ground_mask,
    )

    print(f"原始点数：{len(xyz):,}", flush=True)
    print(f"{VOXEL_SIZE * 1000:.0f} mm 体素数：{voxel_count:,}", flush=True)
    print(
        f"离地高 > {GROUND_CLEARANCE_M:.2f} m 的点："
        f"{above_point_count:,} ({above_point_count / len(xyz) * 100:.2f}%)",
        flush=True,
    )
    print(
        f"非地面体素：{np.count_nonzero(voxel_above_ground_mask):,} "
        f"({voxel_above_ground_mask.mean() * 100:.2f}%)",
        flush=True,
    )
    print(
        "地面模型：候选格 {candidate_cell_count:,}，RANSAC 内点 {inlier_cell_count:,}，"
        "接受格 {accepted_cell_count:,}，Z 范围 {ground_z_min:.3f}–{ground_z_max:.3f} m".format(
            **ground_diagnostics
        ),
        flush=True,
    )
    print(f"已保存：{VOXEL_FILE}", flush=True)
    print(f"已保存：{voxel_ply}", flush=True)
    print(f"已保存：{diagnostic_path}", flush=True)


if __name__ == "__main__":
    main()
