---
layout: default
title: 输出格式
parent: qphase_sde
grand_parent: API 参考
nav_order: 5
---

# 输出格式

SDE engine 每个 job 产生一个逻辑结果。非扫描结果是 `SDEResult`；扫描结果是带
命名 axes 和惰性 point view 的 `SDEScanResult` dataset。

## Single dataset 归档（`.npz`）

使用 `storage_layout: single` 时，每个 SDE job 写入一个 NumPy 归档，包含：

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
*   `orientation` 及其公式字段 — 频率轴的符号约定。缺少该元数据的历史结果按
    `phase_increasing`（原始前向 FFT 方向）解释。

对于 scan，PSD payload 仍附属于一个逻辑 SDE dataset 的命名 point。下游
`mode: analyze` job 将该 dataset 一次传给 `lorentz_fitter`。

使用 `storage_layout: sharded` 时，同一个逻辑 dataset 被拆成一个 job 目录内有限
数量的 `shard_*.npz`。`artifact_manifest.json` 记录 shape 和 loader，
`SDEScanResult.load_dataset` 恢复完整 view。

## Lorentz 拟合输出

`lorentz_fitter` 根据 `export` 选项写入最多三种 artifacts：

*   `fit_results.csv` — 每个扫描点一行，列说明参见 [分析器](./analyzers.zh.md)。
*   `psd_merged.csv` — 合并 PSD 以及不确定度传播使用的 `<scan_value>_sem` 列。
*   `fit_results.npz` / `fit_results.pkl` — 相同数据的替代格式。

## 带限载频输出

当名称包含在 `export` 中时，`band_limited_carrier` 写出三张可审计表：

*   `carrier_results.csv`：局部可辨识状态以及独立的 scan-tracked 载频；数据不足以
    支持唯一平台时，局部或 tracked 频率允许为 `NaN`。
*   `carrier_candidates.csv`：全部带宽/延迟拟合，包括相位残差、频率漂移、衰减率、
    延迟边界和拒绝原因。
*   `carrier_platforms.csv`：scan tracking 之前保留的全部频率平台，包括支持度和评分。

其中 diagnostic uncertainty 是条件于估计模型的敏感度，不是 trajectory SEM；输出
metadata 会明确记录该边界。

`finite_delay_carrier` 写出 `finite_delay_carrier.csv`，每个扫描点、读出和探测器速率
一行，包含读出名称与类型、direct detector carrier、SDE 零延迟极限、有限延迟修正、
相干权重和数值延迟范围。

`spectral_ridge` 写出 `spectral_ridge.csv`（每个扫描点和读出一条选中谱脊）以及
`spectral_ridge_candidates.csv`（完整多尺度候选集）。选中表包含峰位、局部与尺度
不确定度、曲率诊断、PSD SEM 峰位置信区间、相对峰高平台边界、竞争谱脊歧义边界、
路径候选编号与状态。统计置信区间与候选选择歧义分别保存。
候选行还包含 `retained_for_association`、`retention_tier`、
`tracking_path_ranks` 与 `tracking_best_cost_delta`。

## Allan 输出

`allan_variance` 载荷保留在逻辑 SDE dataset 内，包含重叠与非重叠 Allan 方差、逐轨迹
估计、SEM 和有效窗口计数；当 `keep_traj: false` 时不包含原始复轨迹。下游
`allan_scaling` analyser 写出：

*   `allan_points.csv`：每个微扰点一行，包含检测到的白 FM tau 范围、
    `tau * sigma_A^2`、SEM、有效非重叠窗口数、平均角频率及门控状态。
*   `allan_scaling.json`：选定的共同 tau/epsilon 窗口、纯幂律 Allan 拟合、频移响应拟合、
    bootstrap 区间、正规形预期及明确的门控失败原因。

## 分布输出

*   `dist_merged.npz` — scan dataset 的可选 distribution 合并导出。
*   `pdist_merged.pkl` — scan dataset 的可选 polar-distribution 合并导出。

运行目录布局详情参见 [用户指南：输出](../../user_guide/output.zh.md)。
