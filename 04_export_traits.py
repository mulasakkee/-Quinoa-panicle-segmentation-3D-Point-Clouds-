"""Export panicle traits and optional point-cloud products.

The trait CSV files are deliberately written before any PLY export.  This keeps
the numerical result available even when a large point-cloud export is skipped
or interrupted.
"""

from runtime_env import configure_runtime_environment

configure_runtime_environment()

import csv
from pathlib import Path

import laspy
import numpy as np

from config import (
    EXPORT_INDIVIDUAL_INSTANCES,
    EXPORT_INSTANCES_COLORED,
    EXPORT_INSTANCES_WITH_CONTEXT,
    INPUT_LAS,
    INSTANCE_DIR,
    MIN_INSTANCE_POINTS,
    MIN_INSTANCE_VOXELS,
    OUTPUT_DIR,
    RAW_LABEL_FILE,
    VOXEL_FILE,
)
from ground_model import interpolate_ground_z


TRAIT_FIELDS = [
    "instance_id",
    "original_point_count",
    "voxel_count",
    "visible_height_cm",
    "major_width_cm",
    "minor_width_cm",
    "projected_area_cm2",
    "occupied_voxel_volume_cm3",
    "base_agl_m",
    "top_agl_m",
    "centroid_agl_m",
    "base_z_m",
    "top_z_m",
    "centroid_x_m",
    "centroid_y_m",
    "centroid_z_m",
]

SUMMARY_TRAITS = {
    "original_point_count": "points",
    "voxel_count": "voxels",
    "visible_height_cm": "cm",
    "major_width_cm": "cm",
    "minor_width_cm": "cm",
    "projected_area_cm2": "cm2",
    "occupied_voxel_volume_cm3": "cm3",
    "base_agl_m": "m",
    "top_agl_m": "m",
    "centroid_agl_m": "m",
}


def _scalar(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} 应为标量，实际 shape={array.shape}。")
    result = float(array.reshape(()))
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须为有限数。")
    return result


def _require_shape(array: np.ndarray, shape: tuple, name: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} shape 应为 {shape}，实际为 {array.shape}。")


def _validate_inputs(
    voxel_data: np.lib.npyio.NpzFile,
    raw_labels: np.ndarray,
    point_count: int,
) -> dict:
    required_keys = {
        "voxel_size",
        "voxel_grid",
        "voxel_xyz",
        "voxel_point_count",
        "point_to_voxel",
        "ground_xy_origin",
        "ground_grid_size",
        "ground_z_grid",
        "ground_clearance_m",
        "voxel_ground_z",
        "voxel_height_agl",
        "voxel_z_max_agl",
        "voxel_above_ground_mask",
    }
    missing = sorted(required_keys.difference(voxel_data.files))
    if missing:
        raise KeyError(f"{VOXEL_FILE} 缺少字段：{', '.join(missing)}。请先重新运行 01。")

    voxel_grid = np.asarray(voxel_data["voxel_grid"])
    voxel_xyz = np.asarray(voxel_data["voxel_xyz"])
    voxel_point_count = np.asarray(voxel_data["voxel_point_count"])
    point_to_voxel = np.asarray(voxel_data["point_to_voxel"])
    voxel_count = len(voxel_grid)

    if voxel_grid.ndim != 2 or voxel_grid.shape[1] != 3:
        raise ValueError(f"voxel_grid 应为 (n, 3)，实际为 {voxel_grid.shape}。")
    _require_shape(voxel_xyz, (voxel_count, 3), "voxel_xyz")
    _require_shape(voxel_point_count, (voxel_count,), "voxel_point_count")
    _require_shape(raw_labels, (voxel_count,), "voxel_labels_raw")
    _require_shape(point_to_voxel, (point_count,), "point_to_voxel")

    if not np.issubdtype(point_to_voxel.dtype, np.integer):
        raise TypeError("point_to_voxel 必须是整数数组。")
    if point_count and (
        int(point_to_voxel.min()) < 0 or int(point_to_voxel.max()) >= voxel_count
    ):
        raise ValueError("point_to_voxel 含有超出 voxel_grid 范围的索引。")
    if not np.issubdtype(raw_labels.dtype, np.integer) or np.any(raw_labels < 0):
        raise ValueError("voxel_labels_raw 必须是非负整数数组。")
    if np.any(voxel_point_count < 0):
        raise ValueError("voxel_point_count 不能包含负数。")
    if int(np.sum(voxel_point_count, dtype=np.int64)) != point_count:
        raise ValueError("voxel_point_count 总和与 LAS 点数不一致，请从 01 重新运行。")

    mapped_counts = np.bincount(point_to_voxel, minlength=voxel_count)
    if not np.array_equal(mapped_counts, voxel_point_count.astype(np.int64, copy=False)):
        raise ValueError("point_to_voxel 与 voxel_point_count 不一致，请勿混用不同运行的结果。")

    if "source_point_count" in voxel_data.files:
        source_point_count = int(np.asarray(voxel_data["source_point_count"]).reshape(()))
        if source_point_count != point_count:
            raise ValueError(
                f"LAS 有 {point_count} 点，但 voxels.npz 记录 {source_point_count} 点。"
            )

    voxel_size = _scalar(voxel_data["voxel_size"], "voxel_size")
    ground_grid_size = _scalar(voxel_data["ground_grid_size"], "ground_grid_size")
    ground_clearance_m = _scalar(
        voxel_data["ground_clearance_m"], "ground_clearance_m"
    )
    if voxel_size <= 0 or ground_grid_size <= 0 or ground_clearance_m < 0:
        raise ValueError("voxel_size、ground_grid_size 必须为正，ground_clearance_m 不能为负。")

    ground_xy_origin = np.asarray(voxel_data["ground_xy_origin"], dtype=np.float64)
    ground_z_grid = np.asarray(voxel_data["ground_z_grid"], dtype=np.float64)
    _require_shape(ground_xy_origin, (2,), "ground_xy_origin")
    if ground_z_grid.ndim != 2 or min(ground_z_grid.shape) < 1:
        raise ValueError(f"ground_z_grid 必须是非空二维数组，实际为 {ground_z_grid.shape}。")
    if not np.all(np.isfinite(ground_xy_origin)) or not np.all(np.isfinite(ground_z_grid)):
        raise ValueError("地面模型包含 NaN 或无穷值。")

    per_voxel_fields = (
        "voxel_ground_z",
        "voxel_height_agl",
        "voxel_z_max_agl",
        "voxel_above_ground_mask",
    )
    for name in per_voxel_fields:
        _require_shape(np.asarray(voxel_data[name]), (voxel_count,), name)

    voxel_ground_z = np.asarray(voxel_data["voxel_ground_z"], dtype=np.float64)
    voxel_height_agl = np.asarray(voxel_data["voxel_height_agl"], dtype=np.float64)
    voxel_z_max_agl = np.asarray(voxel_data["voxel_z_max_agl"], dtype=np.float64)
    if not (
        np.all(np.isfinite(voxel_ground_z))
        and np.all(np.isfinite(voxel_height_agl))
        and np.all(np.isfinite(voxel_z_max_agl))
    ):
        raise ValueError("体素 AGL 字段包含 NaN 或无穷值。")

    return {
        "voxel_size": voxel_size,
        "voxel_grid": voxel_grid,
        "voxel_xyz": voxel_xyz,
        "voxel_point_count": voxel_point_count.astype(np.int64, copy=False),
        "point_to_voxel": point_to_voxel.astype(np.int64, copy=False),
        "ground_xy_origin": ground_xy_origin,
        "ground_grid_size": ground_grid_size,
        "ground_z_grid": ground_z_grid,
        "ground_clearance_m": ground_clearance_m,
        "voxel_height_agl": voxel_height_agl,
        "voxel_above_ground_mask": np.asarray(
            voxel_data["voxel_above_ground_mask"], dtype=bool
        ),
    }


def _point_agl(
    xyz: np.ndarray,
    ground_xy_origin: np.ndarray,
    ground_grid_size: float,
    ground_z_grid: np.ndarray,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Calculate point AGL in bounded-memory chunks."""
    result = np.empty(len(xyz), dtype=np.float32)
    for start in range(0, len(xyz), chunk_size):
        end = min(start + chunk_size, len(xyz))
        ground_z = interpolate_ground_z(
            xyz[start:end, :2],
            ground_xy_origin,
            ground_grid_size,
            ground_z_grid,
        )
        result[start:end] = xyz[start:end, 2] - ground_z
    return result


def _filter_and_relabel(
    raw_labels: np.ndarray,
    voxel_above_ground_mask: np.ndarray,
    voxel_point_count: np.ndarray,
    point_to_voxel: np.ndarray,
    point_agl: np.ndarray,
    ground_clearance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = raw_labels.astype(np.int64, copy=True)
    labels[~voxel_above_ground_mask] = 0

    label_count = int(labels.max(initial=0)) + 1
    instance_voxel_count = np.bincount(labels, minlength=label_count)
    instance_point_count = np.bincount(
        labels,
        weights=voxel_point_count,
        minlength=label_count,
    )
    keep = (
        (instance_voxel_count >= MIN_INSTANCE_VOXELS)
        & (instance_point_count >= MIN_INSTANCE_POINTS)
    )
    keep[0] = False

    label_map = np.zeros(label_count, dtype=np.int32)
    kept_ids = np.flatnonzero(keep)
    label_map[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
    voxel_labels = label_map[labels]
    original_labels = voxel_labels[point_to_voxel]

    # A voxel can straddle the cutoff plane.  This point-level guard guarantees
    # that points at or below the selected clearance never receive an instance.
    original_labels[point_agl <= ground_clearance_m] = 0

    if original_labels.size:
        actual_point_count = np.bincount(
            original_labels, minlength=int(voxel_labels.max(initial=0)) + 1
        )
        final_keep = actual_point_count >= MIN_INSTANCE_POINTS
        final_keep[0] = False
        if not np.all(final_keep[1:]):
            final_map = np.zeros(len(final_keep), dtype=np.int32)
            ids = np.flatnonzero(final_keep)
            final_map[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
            voxel_labels = final_map[voxel_labels]
            original_labels = final_map[original_labels]

    if np.any(voxel_labels[~voxel_above_ground_mask] != 0):
        raise AssertionError("内部错误：地面体素仍有非零标签。")
    if np.any(original_labels[point_agl <= ground_clearance_m] != 0):
        raise AssertionError("内部错误：地面净空线以下仍有非零点标签。")
    return voxel_labels, original_labels


def _xy_spans_analytic(xy: np.ndarray) -> tuple[float, float]:
    """Robust XY spans along analytic PCA axes, without BLAS/LAPACK calls."""
    center_x = float(np.mean(xy[:, 0]))
    center_y = float(np.mean(xy[:, 1]))
    dx = xy[:, 0] - center_x
    dy = xy[:, 1] - center_y

    covariance_xx = float(np.mean(dx * dx))
    covariance_yy = float(np.mean(dy * dy))
    covariance_xy = float(np.mean(dx * dy))
    angle = 0.5 * np.arctan2(
        2.0 * covariance_xy,
        covariance_xx - covariance_yy,
    )
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    axis_1 = dx * cos_angle + dy * sin_angle
    axis_2 = -dx * sin_angle + dy * cos_angle
    span_1 = float(np.diff(np.percentile(axis_1, [0.1, 99.9]))[0])
    span_2 = float(np.diff(np.percentile(axis_2, [0.1, 99.9]))[0])
    return max(span_1, span_2), min(span_1, span_2)


def _instance_ranges(original_labels: np.ndarray) -> tuple[np.ndarray, list]:
    order = np.argsort(original_labels, kind="stable")
    sorted_labels = original_labels[order]
    boundaries = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(order)]
    ranges = []
    for start, end in zip(starts, ends):
        instance_id = int(sorted_labels[start])
        if instance_id:
            ranges.append((instance_id, int(start), int(end)))
    return order, ranges


def _calculate_traits(
    xyz: np.ndarray,
    point_agl: np.ndarray,
    point_to_voxel: np.ndarray,
    voxel_grid: np.ndarray,
    voxel_size: float,
    order: np.ndarray,
    instance_ranges: list,
) -> list[dict]:
    rows = []
    total = len(instance_ranges)
    for number, (instance_id, start, end) in enumerate(instance_ranges, start=1):
        point_indices = order[start:end]
        points = xyz[point_indices]
        agl = point_agl[point_indices]
        instance_voxels = np.unique(point_to_voxel[point_indices])
        projected_cells = np.unique(voxel_grid[instance_voxels, :2], axis=0)

        major_width, minor_width = _xy_spans_analytic(points[:, :2])
        base_z, top_z = np.percentile(points[:, 2], [0.1, 99.9])
        base_agl, top_agl = np.percentile(agl, [0.1, 99.9])
        rows.append(
            {
                "instance_id": instance_id,
                "original_point_count": len(point_indices),
                "voxel_count": len(instance_voxels),
                "visible_height_cm": float(top_agl - base_agl) * 100.0,
                "major_width_cm": major_width * 100.0,
                "minor_width_cm": minor_width * 100.0,
                "projected_area_cm2": len(projected_cells)
                * voxel_size**2
                * 10_000.0,
                "occupied_voxel_volume_cm3": len(instance_voxels)
                * voxel_size**3
                * 1_000_000.0,
                "base_agl_m": float(base_agl),
                "top_agl_m": float(top_agl),
                "centroid_agl_m": float(np.mean(agl)),
                "base_z_m": float(base_z),
                "top_z_m": float(top_z),
                "centroid_x_m": float(np.mean(points[:, 0])),
                "centroid_y_m": float(np.mean(points[:, 1])),
                "centroid_z_m": float(np.mean(points[:, 2])),
            }
        )
        if number == 1 or number % 50 == 0 or number == total:
            print(f"性状计算进度：{number}/{total}", flush=True)
    return rows


def _write_traits(rows: list[dict], output_dir: Path) -> None:
    trait_file = output_dir / "panicle_traits.csv"
    with trait_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=TRAIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = [
        "trait",
        "unit",
        "instance_count",
        "mean",
        "std",
        "min",
        "p25",
        "median",
        "p75",
        "max",
    ]
    summary_file = output_dir / "panicle_traits_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for trait, unit in SUMMARY_TRAITS.items():
            values = np.asarray([row[trait] for row in rows], dtype=np.float64)
            percentiles = np.percentile(values, [0, 25, 50, 75, 100])
            writer.writerow(
                {
                    "trait": trait,
                    "unit": unit,
                    "instance_count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(percentiles[0]),
                    "p25": float(percentiles[1]),
                    "median": float(percentiles[2]),
                    "p75": float(percentiles[3]),
                    "max": float(percentiles[4]),
                }
            )


def _palette(instance_count: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    palette = rng.uniform(
        0.15, 1.0, size=(instance_count + 1, 3)
    ).astype(np.float64)
    palette[0] = 0.45
    return palette


def _write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    # Keep Open3D out of the CSV-only path.  If its native dependency fails,
    # the trait tables have already been saved by the caller.
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    success = o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)
    if not success:
        raise OSError(f"Open3D 写出失败：{path}")


def _read_rgb(las: laspy.LasData) -> np.ndarray | None:
    dimension_names = set(las.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(dimension_names):
        return None
    rgb = np.column_stack((las.red, las.green, las.blue))
    rgb_scale = 65535.0 if int(rgb.max(initial=0)) > 255 else 255.0
    return rgb.astype(np.float64) / rgb_scale


def _export_ply(
    las: laspy.LasData,
    xyz: np.ndarray,
    voxel_xyz: np.ndarray,
    voxel_labels: np.ndarray,
    original_labels: np.ndarray,
    order: np.ndarray,
    instance_ranges: list,
    output_dir: Path,
    instance_dir: Path,
) -> None:
    instance_count = int(voxel_labels.max(initial=0))
    palette = _palette(instance_count)

    if EXPORT_INSTANCES_COLORED:
        export_mask = original_labels > 0
        path = output_dir / "instances_colored.ply"
        print(f"正在写出仅含实例点的 {path.name} ...", flush=True)
        _write_point_cloud(
            path,
            xyz[export_mask],
            palette[original_labels[export_mask]],
        )
        print(f"已保存：{path}", flush=True)

    if EXPORT_INSTANCES_WITH_CONTEXT:
        path = output_dir / "instances_with_context.ply"
        print(f"正在写出含灰色背景体素的 {path.name} ...", flush=True)
        _write_point_cloud(path, voxel_xyz, palette[voxel_labels])
        print(f"已保存：{path}", flush=True)

    if EXPORT_INDIVIDUAL_INSTANCES:
        instance_dir.mkdir(parents=True, exist_ok=True)
        rgb = _read_rgb(las)
        total = len(instance_ranges)
        for number, (instance_id, start, end) in enumerate(instance_ranges, start=1):
            point_indices = order[start:end]
            if rgb is None:
                colors = np.broadcast_to(
                    palette[instance_id], (len(point_indices), 3)
                )
            else:
                colors = rgb[point_indices]
            path = instance_dir / f"panicle_{instance_id:04d}.ply"
            _write_point_cloud(path, xyz[point_indices], colors)
            if number == 1 or number % 25 == 0 or number == total:
                print(f"单实例 PLY 进度：{number}/{total}", flush=True)


def main() -> None:
    input_las = Path(INPUT_LAS)
    voxel_file = Path(VOXEL_FILE)
    raw_label_file = Path(RAW_LABEL_FILE)
    output_dir = Path(OUTPUT_DIR)
    instance_dir = Path(INSTANCE_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"读取 LAS：{input_las}", flush=True)
    las = laspy.read(input_las)
    xyz = np.column_stack((las.x, las.y, las.z))

    print("读取并校验体素、地面模型和原始标签 ...", flush=True)
    raw_labels = np.load(raw_label_file, allow_pickle=False)
    with np.load(voxel_file, allow_pickle=False) as voxel_data:
        data = _validate_inputs(voxel_data, raw_labels, len(xyz))

    print("计算原始点离地高度（AGL） ...", flush=True)
    point_agl = _point_agl(
        xyz,
        data["ground_xy_origin"],
        data["ground_grid_size"],
        data["ground_z_grid"],
    )
    voxel_labels, original_labels = _filter_and_relabel(
        raw_labels,
        data["voxel_above_ground_mask"],
        data["voxel_point_count"],
        data["point_to_voxel"],
        point_agl,
        data["ground_clearance_m"],
    )
    instance_count = int(voxel_labels.max(initial=0))
    if instance_count == 0:
        raise ValueError(
            "地面去除和小实例筛选后没有保留实例；请检查 01/03 输出及筛选阈值。"
        )

    np.save(output_dir / "voxel_labels.npy", voxel_labels)
    np.save(output_dir / "original_labels.npy", original_labels)

    order, instance_ranges = _instance_ranges(original_labels)
    if len(instance_ranges) != instance_count:
        raise AssertionError("实例标签不连续，或某个体素实例没有对应的离地点。")

    print(f"开始计算 {instance_count} 个实例的性状 ...", flush=True)
    rows = _calculate_traits(
        xyz,
        point_agl,
        data["point_to_voxel"],
        data["voxel_grid"],
        data["voxel_size"],
        order,
        instance_ranges,
    )
    _write_traits(rows, output_dir)

    point_coverage = np.count_nonzero(original_labels) / len(original_labels) * 100.0
    print(f"保留的可见穗头实例数：{instance_count}", flush=True)
    print(f"原始点标签覆盖率：{point_coverage:.2f}%", flush=True)
    print(
        "已优先保存：voxel_labels.npy、original_labels.npy、"
        "panicle_traits.csv、panicle_traits_summary.csv",
        flush=True,
    )

    if (
        EXPORT_INSTANCES_COLORED
        or EXPORT_INSTANCES_WITH_CONTEXT
        or EXPORT_INDIVIDUAL_INSTANCES
    ):
        _export_ply(
            las,
            xyz,
            data["voxel_xyz"],
            voxel_labels,
            original_labels,
            order,
            instance_ranges,
            output_dir,
            instance_dir,
        )
    else:
        print("配置已关闭全部 PLY 导出。", flush=True)

    print("04 性状导出完成。", flush=True)


if __name__ == "__main__":
    main()
