from runtime_env import configure_runtime_environment

configure_runtime_environment()

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from config import (
    DOWNWARD_VERTICAL_COST,
    MAX_FALLBACK_DROP_M,
    MAX_FALLBACK_RISE_M,
    MAX_PATH_COST,
    MAX_VERTICAL_GAP_VOXELS,
    RAW_LABEL_FILE,
    SEED_FILE,
    UPWARD_PENALTY,
    VOXEL_FILE,
    WATERSHED_CROSS_PENALTY,
)


def find_pairs(keys, coords, grid_shape, offset):
    dx, dy, dz = offset
    mask = np.ones(len(keys), dtype=bool)

    for axis, step in enumerate(offset):
        if step > 0:
            mask &= coords[:, axis] < grid_shape[axis] - step
        elif step < 0:
            mask &= coords[:, axis] >= -step

    source = np.flatnonzero(mask)
    stride_x = np.int64(grid_shape[1]) * np.int64(grid_shape[2])
    stride_y = np.int64(grid_shape[2])
    key_offset = np.int64(dx) * stride_x + np.int64(dy) * stride_y + np.int64(dz)
    target_keys = keys[source] + key_offset
    target = np.searchsorted(keys, target_keys)

    inside = target < len(keys)
    source = source[inside]
    target_keys = target_keys[inside]
    target = target[inside]
    matched = keys[target] == target_keys

    return source[matched].astype(np.int32), target[matched].astype(np.int32)


def main():
    print(f"读取体素：{VOXEL_FILE}", flush=True)
    voxel_data = np.load(VOXEL_FILE)
    seed_data = np.load(SEED_FILE)

    grid_shape = voxel_data["grid_shape"].astype(np.int64)
    voxel_keys = voxel_data["voxel_keys"]
    voxel_grid = voxel_data["voxel_grid"]
    voxel_height_agl = voxel_data["voxel_height_agl"]
    above_ground = voxel_data["voxel_above_ground_mask"].astype(bool)

    watershed_map = seed_data["watershed_map"]
    seed_voxel_indices = seed_data["seed_voxel_indices"].astype(np.int64)

    voxel_count = len(voxel_keys)
    if not (
        len(voxel_grid)
        == len(voxel_height_agl)
        == len(above_ground)
        == voxel_count
    ):
        raise ValueError("体素数组长度不一致。")
    if len(seed_voxel_indices) == 0:
        raise ValueError("没有三维传播种子。")
    if np.any(seed_voxel_indices < 0) or np.any(seed_voxel_indices >= voxel_count):
        raise ValueError("种子体素索引越界。")
    if not np.all(above_ground[seed_voxel_indices]):
        raise ValueError("部分顶部种子位于地面掩膜内。")
    if voxel_grid[:, 0].max() >= watershed_map.shape[1] or voxel_grid[:, 1].max() >= watershed_map.shape[0]:
        raise ValueError("体素 XY 索引超出二维分水岭图范围。")

    baseline_labels = watershed_map[voxel_grid[:, 1], voxel_grid[:, 0]]
    active_mask = (baseline_labels > 0) & above_ground
    active_indices = np.flatnonzero(active_mask)
    if len(active_indices) == 0:
        raise ValueError("地面排除后没有可参与三维传播的体素。")

    active_keys = voxel_keys[active_mask]
    active_grid = voxel_grid[active_mask]
    active_height_agl = voxel_height_agl[active_mask]
    active_baseline = baseline_labels[active_mask]
    node_count = len(active_keys)

    seed_nodes = np.searchsorted(active_indices, seed_voxel_indices)
    valid_seed_nodes = seed_nodes < node_count
    if not np.all(valid_seed_nodes):
        raise ValueError("部分顶部种子不在非地面二维分水岭有效区域内。")
    if np.any(active_indices[seed_nodes] != seed_voxel_indices):
        raise ValueError("部分顶部种子未映射到三维图节点。")
    seed_nodes = seed_nodes.astype(np.int32)

    offsets = [
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, -1, 0),
    ]
    offsets.extend(
        (dx, dy, 1)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    )
    offsets.extend([
        (0, 0, 2),
        (1, 0, 2),
        (-1, 0, 2),
        (0, 1, 2),
        (0, -1, 2),
    ])
    offsets.extend((0, 0, dz) for dz in range(3, MAX_VERTICAL_GAP_VOXELS + 1))

    rows = []
    cols = []
    weights = []

    print(f"构建非地面三维图：{node_count:,} 个节点……", flush=True)
    for dx, dy, dz in offsets:
        source, target = find_pairs(
            active_keys,
            active_grid,
            grid_shape,
            (dx, dy, dz),
        )
        if len(source) == 0:
            continue

        cross_penalty = (
            active_baseline[source] != active_baseline[target]
        ).astype(np.float32) * WATERSHED_CROSS_PENALTY

        horizontal_cost = np.hypot(dx, dy)
        downward_cost = horizontal_cost + dz * DOWNWARD_VERTICAL_COST
        upward_cost = horizontal_cost + dz * DOWNWARD_VERTICAL_COST * UPWARD_PENALTY

        rows.extend((source, target))
        cols.extend((target, source))
        weights.extend((
            cross_penalty + upward_cost,
            cross_penalty + downward_cost,
        ))

    if not rows:
        raise ValueError("非地面候选体素之间没有可用图边。")

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    weight = np.concatenate(weights).astype(np.float32)
    graph = csr_matrix(
        (weight, (row, col)),
        shape=(node_count, node_count),
    )

    print(f"运行多源 Dijkstra：{len(seed_nodes)} 个种子……", flush=True)
    _, _, sources = dijkstra(
        graph,
        directed=True,
        indices=seed_nodes,
        return_predecessors=True,
        min_only=True,
        limit=MAX_PATH_COST,
    )

    source_labels = np.zeros(node_count, dtype=np.int32)
    source_labels[seed_nodes] = np.arange(1, len(seed_nodes) + 1)
    active_labels = np.zeros(node_count, dtype=np.int32)
    reachable = sources >= 0
    active_labels[reachable] = source_labels[sources[reachable]]

    seed_height_agl = np.zeros(len(seed_nodes) + 1, dtype=np.float32)
    seed_height_agl[1:] = voxel_height_agl[seed_voxel_indices]
    unreachable = ~reachable
    fallback_labels = active_baseline[unreachable]
    vertical_drop = seed_height_agl[fallback_labels] - active_height_agl[unreachable]
    fallback = (
        (vertical_drop <= MAX_FALLBACK_DROP_M)
        & (vertical_drop >= -MAX_FALLBACK_RISE_M)
    )
    unreachable_indices = np.flatnonzero(unreachable)
    active_labels[unreachable_indices[fallback]] = fallback_labels[fallback]

    voxel_labels = np.zeros(voxel_count, dtype=np.int32)
    voxel_labels[active_indices] = active_labels
    voxel_labels[~above_ground] = 0
    if np.any(voxel_labels[~above_ground]):
        raise AssertionError("地面体素仍然带有实例标签。")

    np.save(RAW_LABEL_FILE, voxel_labels)

    print(f"参与三维传播的非地面体素数：{node_count:,}", flush=True)
    print(f"有向图边数：{graph.nnz:,}", flush=True)
    print(f"Dijkstra 直接分配比例：{reachable.mean() * 100:.2f}%", flush=True)
    print(f"三维传播后的非零标签体素数：{np.count_nonzero(voxel_labels):,}", flush=True)
    print(f"地面体素非零标签数：{np.count_nonzero(voxel_labels[~above_ground])}", flush=True)
    print(f"已保存：{RAW_LABEL_FILE}", flush=True)


if __name__ == "__main__":
    main()
