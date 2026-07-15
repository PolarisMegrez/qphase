---
description: 结果与复现
---

# 结果与复现

QPhase 的设计核心之一是可复现性。每次仿真运行都会生成一个结构化的输出目录，其中不仅包含结果数据，还包含复现该结果所需的完整上下文信息。

## 目录结构

默认情况下，所有输出均存储在 `runs/` 目录下。

### 基于会话的执行

执行 `qphase run` 命令时，会启动一个新的 **会话 (Session)**。会话充当该命令中执行的所有任务的容器。

```text
runs/
└── 2025-12-31T05-23-05_281415/      # 会话目录 (时间戳 + UUID)
    ├── session_manifest.json        # 整个会话的元数据
    ├── vdp_sde/                     # 任务目录 (任务名称)
    │   ├── config_snapshot.yaml     # 该任务使用的完整配置快照
    │   ├── vdp_sde.npz              # SDE 结果归档
    │   └── qphase.log               # 执行日志
    └── vdp_viz/                     # 下游任务目录
        ├── config_snapshot.yaml
        └── plot.png
```

## 可复现性

### 配置快照 (`config_snapshot.yaml`)

每个任务目录都包含一个 `config_snapshot.yaml` 文件。这是运行该任务时使用的 **精确** 配置，包含：
*   合并后的全局配置与任务配置。
*   解析后的插件默认值。
*   系统环境信息（QPhase 版本、Python 版本、操作系统）。

要复现某个结果，只需直接运行此快照文件：

```bash
qphase run runs/2025-12-31.../vdp_sde/config_snapshot.yaml
```

### 会话清单 (`session_manifest.json`)

会话清单跟踪会话中所有任务的状态和关系。其用途包括：
*   调试失败的任务流。
*   以编程方式分析运行历史。
*   恢复中断的会话（高级用法）。

## 数据格式

结果数据的格式取决于所使用的引擎。

*   **SDE 引擎**：保存为 NumPy `.npz` 归档。顶层键包括 `t0`、`t1`、`dt`、`meta`、`analysis`，并在保留轨迹时包含形状为 `(n_traj, n_time, n_modes)` 的原始 `data`。
*   **Viz 引擎**：保存图像 (`.png`, `.pdf`) 或处理后的数据文件。

输出格式可在任务配置的引擎设置中进行调整。SDE artifacts 的详细模式说明参见 [`qphase_sde` 参考中的输出格式](../api/qphase_sde/output.zh.md)。

### SDE 分析载荷

当 SDE 任务配置包含 `analyser` 插件时，分析结果会按 analyser key 存储在 `analysis` 下。

*   `analysis["psd"]`：
    *   `axis`：一维频率轴。
    *   `psd`：形状为 `(n_frequency, n_modes)` 的 PSD 矩阵。
    *   `modes`：分析的模式索引列表。
    *   `kind`：`"complex"` 或 `"modular"`。
    *   `convention`：`"symmetric"`、`"unitary"` 或 `"pragmatic"`。
    *   `peaks`：将每个模式映射到序列化后的 `PeakInfo`，包含 `indices`、`frequencies`、`values` 和 `properties`。
*   `analysis["dist"]`：
    *   `distributions`：将每个模式映射到直方图结果。复数模式使用二维直方图（`hist`、`xedges`、`yedges`、`type="2d_complex"`）；实数模式使用一维直方图（`hist`、`edges`、`type="1d_real"`）。
    *   `modes`：分析的模式索引列表。
    *   `bins`：使用的分箱数。
    *   `density`：是否归一化为概率密度。
*   `analysis["pdist"]`：实验性的极坐标分布载荷，结构与 `dist` 类似（包含 `distributions`、`modes`、`bins_config`、`density`），需在配置中启用极坐标分布分析器。

如果未设置 `engine.sde.keep_traj`，引擎会在完成分析后丢弃原始轨迹以减小文件体积。此时 `.npz` 仍会包含 `meta`、`analysis`、`t0`、`t1` 和 `dt`。

## 后处理导出

跨 job 后处理现在通过调度工作流实现，使用 `analyser.lorentz_fitter` 插件并配合 `engine.sde.mode: analyze`。该分析器消费已有的 `analysis["psd"]` 数据，不会从轨迹重新计算 PSD。默认写出：

*   `fit_results.csv`：每个扫描值一行。每个 Lorentz 参数都有对应的 `_std` 协方差标准差列；`uncertainty_source` 表明协方差传播了 `psd_sem` 还是使用兼容旧结果的残差回退。PSD 不确定度不会改变 Lorentz 拟合权重。`status` 可为 `ok`（满足质量阈值）、`low_quality`（不满足质量阈值）或 `failed`（拟合失败）。
*   `psd_merged.csv`：以频率为索引、每个扫描值一列的 PSD 表；PSD 标准误可用时同时包含 `<scan_value>_sem` 列。
*   `dist_merged.npz`（实验性）：设置 `export_dist: true` 时写出。包含键 `dist_list`、`scan_params`、`__schema_version__` 和 `__created_by__`。
*   `pdist_merged.pkl`（实验性）：设置 `export_dist: true` 时写出。是一个 pickled 字典，包含 `rows`、`__schema_version__` 和 `__created_by__`。

常用参数包括 `output_dir`、`psd_key`、`fit_window`、`freq_min`、`freq_max`、`min_r2`、`min_peak_height`、`max_linewidth`、`uncertainty`、`export_dist`、`clip_by_std` 和 `clip_sigma`。默认的 `uncertainty: auto` 会在 PSD 标准误可用时使用它，同时兼容旧结果文件。设置 `clip_by_std: true` 会先按 **平方后的 PSD** 加权分布的均值 ± `clip_sigma` 个标准差裁剪频率窗口，有助于忽略远端长尾上的干扰峰，并加快宽频网格上的拟合。

对 Lorentz 线型，平方 PSD 加权标准差等于 `linewidth / 2`。因此默认 `clip_sigma: 10.0` 大约保留峰值两侧各 `5 × FWHM` 的范围，足以覆盖整条谱线，同时仍能排除很远处的干扰。

拟合结果表还包含 `amplitude`（高于基线的峰高）、`peak_intensity`（总峰高）、`R2`、`status`、`error` 和 `warning`。当输入数据的平方 PSD 加权标准差与 Lorentz 期望（`std = linewidth / 2`）相差超过两倍时，`warning` 字段会写入警告，提示数据可能不是单峰 Lorentz。

示例工作流：

```yaml
- name: sim
  save: true
  engine:
    sde: { t0: 0.0, t1: 1.0, dt: 0.01, n_traj: 8, seed: 42 }
  model:
    kerr_3pa:
      epsilon: [0.025, 0.05, 0.1]
  analyser:
    psd: { modes: [0], kind: complex, find_peaks: true }

- name: fit
  input: sim
  aggregate_input:
    on: epsilon
  engine:
    sde: { mode: analyze }
  analyser:
    lorentz_fitter:
      scan_param: epsilon
      mode: 0
```
