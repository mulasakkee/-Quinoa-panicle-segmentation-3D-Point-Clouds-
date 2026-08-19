# 藜麦穗部三维点云实例分割与性状提取

这是一个面向三维点云处理练习和研究验证的藜麦穗部候选实例分割流程。项目从 LAS 点云建立局部地面模型，将绝对高程转换为离地高度（AGL，height above ground），再依次完成顶部种子检测、三维标签传播和实例性状导出。

仓库地址：[Quinoa-panicle-segmentation-3D-Point-Clouds](https://github.com/mulasakkee/-Quinoa-panicle-segmentation-3D-Point-Clouds-)

> 仓库不包含原始三维点云，也不包含运行生成的 `results*` 文件。使用者需要准备自己的 LAS 数据。

## 项目目的与适用范围

本项目主要用于：

- 练习植物三维点云的地面归一化、体素化、实例分割和性状计算；
- 从已经裁剪到试验小区或植株区域的点云中，快速获得穗部候选实例和初步性状；
- 为后续人工检查、参数试验或更严格的穗部算法提供基线结果。

更适合地面可见、坐标单位为米、Z 轴向上，并且冠层顶部存在可识别局部高点的数据。若地面完全被遮挡、点云噪声很强、植株严重交叠，或坐标单位不是米，需要先预处理或重新调整参数。

## 方法概览

处理流程必须严格按照 `01 → 02 → 03 → 04` 的顺序运行：

1. **`01_voxelize.py`：地面归一化与体素化**
   估计局部数字地面模型（DTM），计算点和体素的 AGL 高度，以 8 mm 分辨率体素化，并生成非地面候选掩膜。
2. **`02_detect_seeds.py`：顶部种子检测**
   只使用非地面体素建立 AGL 高度图和密度图，经过平滑、局部峰值检测和二维标记控制分水岭获得顶部种子。
3. **`03_segment_3d.py`：三维标签传播**
   在非地面候选体素上建立有向邻接图，以顶部种子执行多源 Dijkstra 传播；保存前再次强制地面体素标签为 0。
4. **`04_export_traits.py`：过滤、性状计算与导出**
   按最小体素数和最小原始点数过滤实例，重新连续编号，计算点级 AGL 性状，并导出 CSV 和可选 PLY 文件。

### 局部地面归一化

`01_voxelize.py` 调用 `ground_model.py`，默认执行以下步骤：

1. 从输入点云确定性抽样，最多使用 2,000,000 个点；
2. 排除全局最低 0.1% 的异常低点；
3. 在 20 cm 的 XY 粗格网内取 Z 的第 5 百分位作为地面候选；
4. 使用 RANSAC 拟合缓坡趋势平面；
5. 保留趋势面附近的候选格点，进行中值平滑并填补空洞，形成局部 DTM；
6. 双线性插值得到每个点和体素位置的地面高程；
7. 计算 `height_agl = z - ground_z(x, y)`，仅允许高于离地阈值的体素进入后续分割。

该步骤对高度进行地面归一化，但不会旋转或重新定向原始坐标系。

## 三个不能混淆的尺度

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `VOXEL_SIZE` | `0.008 m` | 三维体素边长，即 **8 mm** |
| `GROUND_CLEARANCE_M` | `0.08 m` | 进入分割的离地清除阈值，即 **8 cm** |
| `GROUND_GRID_SIZE_M` | `0.20 m` | 估计局部地面的粗格网边长，即 **20 cm** |

体素大小仍是 8 mm；8 cm 是地面以上的过滤高度，二者用途不同。

## 仓库文件

| 文件 | 作用 |
|---|---|
| `config.py` | 集中配置输入、输出、空间尺度、分割和导出参数 |
| `ground_model.py` | 构建局部地面模型并进行地面高程插值 |
| `runtime_env.py` | 在 Windows 下补充 Conda 动态库搜索路径 |
| `01_voxelize.py` | 地面建模、AGL 计算和体素化 |
| `02_detect_seeds.py` | 二维高度图、顶部种子和分水岭 |
| `03_segment_3d.py` | 非地面体素的三维图传播 |
| `04_export_traits.py` | 实例过滤、性状统计和点云导出 |
| `run_pipeline.ps1` | Windows PowerShell 一键按顺序运行 01–04 |
| `environment.yml` | Conda 环境定义 |
| `tests/test_ground_model.py` | 地面模型和边界插值的单元测试 |

## 安装

### 1. 克隆仓库

```powershell
git clone https://github.com/mulasakkee/-Quinoa-panicle-segmentation-3D-Point-Clouds-.git
Set-Location ".\-Quinoa-panicle-segmentation-3D-Point-Clouds-"
```

### 2. 创建 Conda 环境

推荐安装 Miniforge 或 Miniconda，然后在仓库根目录运行：

```powershell
conda env create -f environment.yml
conda activate quinoa3d
```

环境使用 Python 3.11，主要依赖为：

- NumPy、SciPy
- laspy
- scikit-image
- Matplotlib
- Open3D
- Pillow

如 `environment.yml` 后续有变化，可用以下命令更新已有环境：

```powershell
conda env update -n quinoa3d -f environment.yml --prune
```

## 数据准备

在仓库根目录创建本地 `data` 文件夹，并放入自己的 LAS 点云，例如：

```text
项目根目录/
├─ data/
│  └─ your_point_cloud.las
├─ config.py
├─ 01_voxelize.py
└─ ...
```

输入数据应满足：

- 文件格式为 LAS；
- XYZ 坐标单位为米；
- Z 轴表示向上的高程方向；
- 场景中包含足够且分布较均匀的可见地面点；
- 推荐先裁剪到目标试验小区或目标植株区域；
- RGB 字段不是必需的；没有 RGB 时，相关 PLY 会使用灰色显示。

不要将大型点云、运行结果或缓存提交到仓库。项目的 `.gitignore` 应持续排除 `data/`、`results*/`、`*.las`、`*.laz`、生成的 PLY 和 Python 缓存。

## 配置输入与输出

打开 `config.py`，修改输入文件名：

```python
INPUT_LAS = PROJECT_DIR / "data" / "your_point_cloud.las"
```

默认输出目录为：

```python
OUTPUT_DIR = PROJECT_DIR / "results_ground_normalized"
```

所有脚本都从 `config.py` 读取路径和参数，不依赖启动命令所在的当前目录。

## 运行方法

### Windows：一键运行

在已激活的 `quinoa3d` 环境中执行：

```powershell
$pythonExe = (Get-Command python).Source
.\run_pipeline.ps1 -PythonExe $pythonExe
```

运行脚本会先进行 NumPy 和 Matplotlib 环境检查，然后按顺序执行四个阶段；任一阶段返回错误时会停止。

### 逐步运行

Windows、Linux 和 macOS 均可在已激活环境中逐步执行：

```bash
python 01_voxelize.py
python 02_detect_seeds.py
python 03_segment_3d.py
python 04_export_traits.py
```

不能跳过阶段或调换顺序。修改输入点云、地面参数、体素大小或上游算法后，应从 01 开始重新运行全部阶段，不能混用不同运行产生的 `.npz` 和 `.npy` 中间文件。

## 输出说明

默认在 `results_ground_normalized/` 中生成：

```text
results_ground_normalized/
├─ voxels.npz
├─ voxels_above_ground.ply
├─ maps_and_seeds.npz
├─ voxel_labels_raw.npy
├─ voxel_labels.npy
├─ original_labels.npy
├─ panicle_traits.csv
├─ panicle_traits_summary.csv
├─ instances_colored.ply
├─ instances_with_context.ply
├─ figures/
│  ├─ 00_ground_model.png
│  ├─ 01_height_density.png
│  └─ 02_seeds_watershed.png
└─ instances/
   ├─ panicle_0001.ply
   └─ ...
```

| 输出 | 内容 |
|---|---|
| `voxels.npz` | 体素坐标、点到体素映射、颜色、点数、DTM、AGL 和非地面掩膜等核心中间数据 |
| `voxels_above_ground.ply` | 仅保留离地阈值以上的体素中心，用于检查地面去除效果 |
| `maps_and_seeds.npz` | AGL 高度图、密度图、二维分水岭和种子索引 |
| `voxel_labels_raw.npy` | 03 输出的原始体素实例标签 |
| `voxel_labels.npy` | 04 过滤并连续编号后的体素标签 |
| `original_labels.npy` | 映射回每个原始 LAS 点的最终标签，0 表示背景或被排除点 |
| `panicle_traits.csv` | 每个保留实例一行的性状表 |
| `panicle_traits_summary.csv` | 主要性状的数量、均值、样本标准差、最小值、四分位数和最大值 |
| `instances_colored.ply` | 仅包含最终非零实例的原始点，不显示地面 |
| `instances_with_context.ply` | 显示全部体素作为场景上下文；标签 0（包括地面）显示为灰色 |
| `instances/panicle_XXXX.ply` | 每个实例单独导出的原始点云 |
| `figures/*.png` | 地面模型、AGL 高度/密度和种子/分水岭诊断图 |

可以在 `config.py` 中关闭不需要的 PLY 导出，以减少磁盘和内存占用：

```python
EXPORT_INSTANCES_COLORED = False
EXPORT_INSTANCES_WITH_CONTEXT = False
EXPORT_INDIVIDUAL_INSTANCES = False
```

## 性状字段定义

`panicle_traits.csv` 使用原始点和最终体素标签计算以下字段：

| 字段 | 单位 | 定义 |
|---|---:|---|
| `instance_id` | — | 过滤后连续编号的实例 ID |
| `original_point_count` | points | 实例包含的原始 LAS 点数 |
| `voxel_count` | voxels | 实例占据的三维体素数 |
| `visible_height_cm` | cm | AGL 第 99.9 百分位与第 0.1 百分位之差 |
| `major_width_cm` | cm | XY 主方向投影的较大稳健跨度（0.1%–99.9%） |
| `minor_width_cm` | cm | XY 主方向投影的较小稳健跨度（0.1%–99.9%） |
| `projected_area_cm2` | cm² | 占据的唯一 XY 体素格数乘以体素平面面积 |
| `occupied_voxel_volume_cm3` | cm³ | 占据的三维体素数乘以单体素体积；不是封闭表面体积 |
| `base_agl_m` | m | 点级 AGL 第 0.1 百分位 |
| `top_agl_m` | m | 点级 AGL 第 99.9 百分位 |
| `centroid_agl_m` | m | 点级 AGL 平均值 |
| `base_z_m` | m | 原始 Z 的第 0.1 百分位 |
| `top_z_m` | m | 原始 Z 的第 99.9 百分位 |
| `centroid_x_m` | m | 原始 X 平均值 |
| `centroid_y_m` | m | 原始 Y 平均值 |
| `centroid_z_m` | m | 原始 Z 平均值 |

这里的“可见高度”“投影面积”和“占据体素体积”都是基于采集到的可见点和当前实例标签计算的几何代理量，不等同于破坏性测量的真实生物量或封闭几何体积。

## 测试

在仓库根目录和 `quinoa3d` 环境中运行：

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- 含冠层点和异常低点时，对缓坡地面的恢复；
- 地面格网边界外查询的插值裁剪行为。

单元测试通过只说明这些基础函数满足测试条件，不能代替对真实点云分割结果的人工或标注验证。

## 参数与资源提示

所有关键参数均位于 `config.py`：

- 地面估计：`GROUND_GRID_SIZE_M`、`GROUND_CELL_PERCENTILE`、`GROUND_RANSAC_TOLERANCE_M`、`GROUND_RESIDUAL_LIMIT_M`；
- 离地过滤：`GROUND_CLEARANCE_M`；
- 种子检测：`SMOOTH_SIGMA_M`、`PEAK_MIN_DISTANCE_M`、`PEAK_HEIGHT_PERCENTILE`、`MIN_BASIN_CELLS`；
- 三维传播：`UPWARD_PENALTY`、`WATERSHED_CROSS_PENALTY`、`MAX_PATH_COST`、`MAX_FALLBACK_DROP_M`；
- 实例过滤：`MIN_INSTANCE_VOXELS`、`MIN_INSTANCE_POINTS`；
- 导出开关：三个 `EXPORT_*` 参数。

建议每次只修改一类参数，并结合三张诊断图和 PLY 进行检查。尤其应先确认地面模型合理，再调整顶部种子和三维传播参数。

8 mm 体素会在大范围或高密度点云上产生大量体素；三维图、点到体素映射和 PLY 导出可能占用较多内存、运行时间和磁盘空间。资源不足时可先裁剪小区域进行参数试验，或适当增大 `VOXEL_SIZE`，但修改体素大小后必须从 01 全部重跑。

## 重复运行注意事项

大部分同名结果会被覆盖，但程序不会在每次运行前自动清空 `results_ground_normalized/instances/`。如果新一次运行的实例数少于上一次，旧的 `panicle_XXXX.ply` 可能残留并被误认为新结果。

为保留可追溯性，推荐每次试验修改 `OUTPUT_DIR` 使用新的结果目录，或在确认不再需要旧结果后先将整个旧结果目录移动到备份位置。不要混合比较不同输入或不同参数生成的中间文件。

## 当前限制

- 当前版本解决的是局部地面归一化以及“地面被分到实例中”的问题；
- 去除地面并不等于已经严格提取穗部，03 的标签仍可能沿茎和叶片向下传播；
- 当输入包含完整植株时，导出的高度、宽度、面积和体积可能描述的是包含茎叶的候选实例，而不是真正的穗部；
- 二维顶部局部峰值不一定与真实穗一一对应，遮挡和相互接触可能造成欠分割或过分割；
- 当前没有提供针对特定品种、设备或田间条件的通用参数，也没有声明分割精度；
- 在视觉检查或人工标注验证完成前，不应将输出直接用于正式生物学结论。

后续可根据真实穗长加入“相对顶部种子的最大向下深度”、穗部高度先验、形状约束或监督模型，以进一步排除茎叶。

本代码用于学习、算法实验和研究验证，不替代经过标定、重复试验和统计验证的正式植物表型测量流程。
