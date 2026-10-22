---
layout: default
title: 输出格式
parent: qphase_sde
grand_parent: API 参考
nav_order: 5
---

# 输出格式

SDE engine 每个 job 产生一个逻辑结果：由命名 typed data products（轨迹与各分析器 product）组成的 `SDEDataBundle`。扫描结果是同一个 bundle，其 product 携带命名参数轴；`point_view(index)` 给出逐点的惰性视图。

## Artifact 目录

每个保存的 SDE job 写入一个 Artifact v4 目录：

| 条目 | 说明 |
| :-- | :-- |
| `artifact_manifest.json` | 经过校验的 manifest（`qphase.artifact/4`）：完整 product schema、`sde.bundle/1` bundle 描述符（scan 网格、product 角色）、provenance、逐变量 payload 引用。 |
| `<stem>.npz` | `storage_layout: single`：每个 product 一个文件，包含其全部变量。 |
| `<stem>__<variable>.npz` / `<stem>__<variable>__<NNNN>.npz` | 默认/sharded 布局：每个文件一个 `"data"` 键，变量按字节目标分块。 |

payload 数组以原生 dtype 存储——恢复时不需要 `allow_pickle`。在 Python 中加载：

```python
from qphase.data import load_bundle

bundle = load_bundle("runs/2026/08/<session-id>/sim")
psd = bundle.products["psd"]      # 惰性 backed；尚未读取 payload
point = bundle.point_view((0,))   # bundle 的单扫描点视图
```

## PSD 输出

`psd` 分析器存储：

*   `axis` — 频率或角频率向量。
*   `psd` — 每个模式的 PSD 均值。
*   `psd_std` / `psd_sem` — 跨轨迹样本标准差和均值标准误。
*   `orientation` 及其公式字段 — 频率轴的符号约定。缺少该元数据的历史结果按
    `phase_increasing`（原始前向 FFT 方向）解释。

对于 scan，PSD product 携带逻辑 SDE bundle 的命名 scan 轴。下游
`mode: analyze` job 将该 bundle 整体交给 `lorentz_fitter`。

使用 `storage_layout: sharded` 时，同一个逻辑 dataset 被拆成同一 job 目录内
按字节限定的分块文件。manifest 记录分块布局与 `npz/3` 存储适配器 id；
`qphase.data.load_bundle` 与物理布局无关地惰性恢复完整 bundle。

## Lorentz 拟合输出

`lorentz_fitter` 持久化一个 typed statistics product（以 analyser label 命名），其字段
保存每个扫描点一行的拟合结果——列说明参见 [分析器](./analyzers.zh.md)——以及与参考
频率轴对齐的合并 PSD 与 PSD SEM 曲线。不再写旁路文件；用
`qphase.data.load_products(job_dir)` 加载。

## 带限载频输出

`band_limited_carrier` 持久化三个可审计的 typed product（`carrier`、
`carrier.candidates`、`carrier.platforms`）：

*   `carrier`：局部可辨识状态以及独立的 scan-tracked 载频；数据不足以支持唯一平台时，
    局部或 tracked 频率允许为 `NaN`。
*   `carrier.candidates`：全部带宽/延迟拟合，包括相位残差、频率漂移、衰减率、
    延迟边界和拒绝原因。
*   `carrier.platforms`：scan tracking 之前保留的全部频率平台，包括支持度和评分。

其中 diagnostic uncertainty 是条件于估计模型的敏感度，不是 trajectory SEM；product
metadata 会明确记录该边界。

ridge-conditioned 模式下，三个 product 还会保存 `ridge_candidate_index`、
`ridge_retention_tier`、ridge 中心不确定度、最近 ridge 带宽上限、
`ridge_carrier_correction` 与 `ridge_conditioned_uncertainty_upper`；每个扫描点允许有
多行结果。

`finite_delay_carrier` 持久化一个 typed `carrier` product，每个扫描点、读出和探测器
速率一行，包含读出名称与类型、direct detector carrier、SDE 零延迟极限、有限延迟修正、
相干权重和数值延迟范围。

`spectral_ridge` 持久化两个 typed product：`ridge`（每个扫描点和读出一条选中谱脊）与
`ridge.candidates`（完整多尺度候选集）。选中行包含峰位、局部与尺度不确定度、曲率诊断、
PSD SEM 峰位置信区间、相对峰高平台边界、竞争谱脊歧义边界、路径候选编号与状态。
候选行还包含 `retained_for_association`、`retention_tier`、`tracking_path_ranks` 与
`tracking_best_cost_delta`。

## Allan 输出

`allan_variance` 载荷保留在逻辑 SDE dataset 内，包含重叠与非重叠 Allan 方差、逐轨迹
估计、SEM 和有效窗口计数；当 `keep_traj: false` 时不包含原始复轨迹。下游
`allan_scaling` analyser 持久化一个 typed product，含两部分：

*   `rows`：每个微扰点一行，包含检测到的白 FM tau 范围、`tau * sigma_A^2`、SEM、
    有效非重叠窗口数、平均角频率及门控状态。
*   `summary`：选定的共同 tau/epsilon 窗口、纯幂律 Allan 拟合、频移响应拟合、
    bootstrap 区间、正规形预期及明确的门控失败原因。

distribution 与 polar-distribution 结果分别是各自 analyser 的 typed product；
`lorentz_fitter` 不再把它们合并成 NPZ/PKL 导出。

运行目录布局详情参见 [用户指南：输出](../../user_guide/output.zh.md)。
