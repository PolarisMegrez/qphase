---
description: 后处理架构
---

# 后处理架构

后处理不再是一个独立的 core 命令，也不是一个独立的资源包。它被表达为一个普通的 scheduler job：使用 SDE engine 的 `mode: analyze` 加上 `analyser.lorentz_fitter` 插件。

## 设计原则

- **Core** (`qphase`) 在 `qphase.core.aggregation` 中提供通用的聚合与导出工具。
- **`qphase_sde`** 提供 SDE 专属的跨 job 分析器 `lorentz_fitter`，用于对聚合后的 PSD 数据进行 Lorentzian 拟合并写出合并结果。
- **`qphase postprocess` CLI 命令已被移除**，请改用 `qphase run <workflow.yaml>`。

## 工作流示例

```yaml
- name: sim
  save: true
  engine:
    sde:
      t1: 1.0
      dt: 0.01
      n_traj: 2
  model:
    kerr_3pa:
      epsilon: [0.025, 0.05]
  analyser:
    psd:
      modes: [0]
      kind: complex

- name: fit
  input: sim
  aggregate_input:
    on: epsilon
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: epsilon
      mode: 0
```

scheduler 会：

1. 将 `sim` job 按 `epsilon` 的每个取值展开为多个 job。
2. 把展开后的结果聚合为单个输入，传给 `fit` job。
3. 在 `analyze` 模式下运行 `analyser.lorentz_fitter`，在 `fit` job 的 run 目录中生成 `fit_results.csv` 和 `psd_merged.csv`。

## 输出文件

| 文件 | 生成者 | 内容 |
| :--- | :--- | :--- |
| `fit_results.csv` | `lorentz_fitter` | 每个扫描值一行，包含拟合参数及由协方差得到的标准差。 |
| `psd_merged.csv` | `lorentz_fitter` | 频率轴加上每个扫描值对应的 PSD 与可选 PSD SEM 列。 |
| `dist_merged.npz` | `lorentz_fitter`（可选） | 聚合后的 distribution payload。 |
| `pdist_merged.pkl` | `lorentz_fitter`（可选） | 聚合后的极坐标 distribution payload。 |

NPZ/PKL bundle 会通过 `qphase.core.aggregation` 写入 `__schema_version__` 和 `__created_by__` 元数据。

## 边界

- 单个 result 内的分析（单 job PSD、寻峰、distribution）属于 `analyser` 插件。
- 跨 result 的聚合、排序、带 schema version 的导出属于 `qphase.core.aggregation`。
- SDE 专属的曲线拟合与 payload 提取属于 `qphase_sde.analyser.lorentz_fitter`。
