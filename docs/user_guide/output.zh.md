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

*   **SDE 引擎**：保存为 NumPy `.npz` 归档。归档包含 `t0`、`dt`、`meta`、`analysis`，并在保留轨迹时包含形状为 `(n_traj, n_time, n_modes)` 的原始 `data`。
*   **Viz 引擎**：保存图像 (`.png`, `.pdf`) 或处理后的数据文件。

输出格式可在任务配置的引擎设置中进行调整。

### SDE 分析载荷

当 SDE 任务配置包含 `analyser` 插件时，分析结果会按 analyser key 存储在 `analysis` 下。

*   `analysis["psd"]`：包含 `axis`、`psd`、`modes`、`kind`、`convention` 和 `peaks`。PSD 矩阵形状为 `(n_frequency, n_modes)`。
*   `analysis["dist"]`：按模式索引的笛卡尔相空间分布。
*   `analysis["pdist"]`：按模式索引的极坐标分布。

如果未设置 `engine.sde.keep_traj`，引擎会在完成分析后丢弃原始轨迹以减小文件体积。此时 `.npz` 仍会包含 `meta`、`analysis`、`t0` 和 `dt`。

## 后处理导出

使用 `qphase postprocess` 可以把已保存的 PSD 分析结果转换为稳定的 CSV 文件：

```bash
qphase postprocess runs/2026-03-17T21-03-06_088ab0 --scan-param epsilon --mode 0
```

该命令消费已有的 `analysis["psd"]` 数据，不会从轨迹重新计算 PSD。默认写出：

*   `fit_results.csv`：每个 job 一行，包含扫描参数、Lorentz 拟合的 `center`、`linewidth`、`base`、`peak_intensity`、`R2`、`status` 和 `error`。
*   `psd_merged.csv`：以频率为索引、每个扫描值一列的 PSD 表。
*   `dist_merged.npz` 和 `pdist_merged.pkl`：传入 `--export-dist` 时写出的实验性分布合并文件。

常用参数包括 `--output-dir`、`--psd-key`、`--fit-window`、`--overwrite` 和 `--export-dist`。
