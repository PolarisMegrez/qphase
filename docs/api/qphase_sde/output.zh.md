---
layout: default
title: 输出格式
parent: qphase_sde
grand_parent: API 参考
nav_order: 5
---

# 输出格式

SDE 引擎产生两类 artifacts：单次运行的归档文件与合并后的分析 bundle。

## 单次运行归档（`.npz`）

当 `save: true` 时，每个 SDE 任务写入一个 NumPy 归档，包含：

| 键 | 类型 | 说明 |
| :-- | :-- | :-- |
| `t0` | `float` | 起始时间。 |
| `dt` | `float` | 保存后的采样间隔（`dt * save_stride`）。 |
| `meta` | `object` | 元数据，如模型参数与丢弃原因。 |
| `analysis` | `object` | 以分析器名称为键的载荷。 |
| `data` | `ndarray` | 原始轨迹，形状 `(n_traj, n_saved, n_modes)`。仅在 `keep_traj: true` 或分析器需要时存在。 |

在 Python 中加载：

```python
import numpy as np
archive = np.load("run.npz", allow_pickle=True)
meta = archive["meta"].item()
psd = archive["analysis"].item().get("psd")
```

## PSD 输出

`psd` 分析器存储：

*   `axis` — 频率或角频率向量。
*   `psd` — 每个模式的 PSD 均值。
*   `psd_std` / `psd_sem` — 跨轨迹样本标准差和均值标准误。

当分析器在不同参数值的多个任务中运行时，调度器可将其聚合成一张表，供 `lorentz_fitter` 分析器使用。

## Lorentz 拟合输出

`lorentz_fitter` 根据 `export` 选项写入最多三种 artifacts：

*   `fit_results.csv` — 每个扫描点一行，列说明参见 [分析器](./analyzers.zh.md)。
*   `psd_merged.csv` — 合并 PSD 以及加权拟合使用的 `<scan_value>_sem` 列。
*   `fit_results.npz` / `fit_results.pkl` — 相同数据的替代格式。

## 分布输出

*   `dist_merged.npz` — 跨扫描聚合时由 `dist` 分析器保存。
*   `pdist_merged.pkl` — 跨扫描聚合时由 `pdist` 分析器保存。

运行目录布局详情参见 [用户指南：输出](../../user_guide/output.zh.md)。
