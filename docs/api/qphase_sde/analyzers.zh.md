---
layout: default
title: 分析器
parent: qphase_sde
grand_parent: API 参考
nav_order: 4
---

# 分析器

分析器在 SDE 积分循环结束后运行，生成存储在 `SDEResult.analysis` 中的载荷。它们也可以在 `mode: analyze` 下调用，用于后处理。

## `psd`

估算所选模式的功率谱密度（PSD）。

### 配置

```yaml
analyser:
  psd:
    modes: [0]
    kind: complex
    expected_freq_max: 0.34
    find_peaks: true
```

| 键 | 类型 | 说明 |
| :-- | :-- | :-- |
| `modes` | `list[int]` | 要分析的模式索引。 |
| `kind` | `str` | PSD 变体，如 `complex`、`real`、`imag`。 |
| `expected_freq_max` | `float \| None` | 输出频率轴单位下预期的最大物理频率；达到 Nyquist 上限时分析直接失败。 |
| `find_peaks` | `bool` | 是否报告峰值位置。 |

### 频率轴

频率网格取决于保存后的轨迹：

```text
f = np.fft.fftfreq(n_saved, dt * save_stride) * 2 * pi
```

对于窄峰，应选择 `save_stride` 使 Nyquist 频率远高于峰值。
对于角频率约定，`omega_Nyquist = pi / (dt * save_stride)`。增大 `t1` 只能改善
分辨率 `2*pi/t1`，不会扩大该带宽。设置 `expected_freq_max` 可以把原本静默的
混叠错误转化为明确的配置失败。

### 输出载荷

分析器导出：

*   `axis` — 频率或角频率轴。
*   `psd` — 每个请求模式的 PSD 均值。
*   `psd_std` — 跨轨迹样本标准差（`ddof=1`）。
*   `psd_sem` — 均值标准误，即 `psd_std / sqrt(n_traj)`。
*   `uncertainty` — 标识 `psd_sem`、独立统计单元和样本数的元数据。

对于 Welch 和 multitaper，先在每条轨迹内部平均 segment 或 taper，再跨轨迹
计算不确定度，因此不会把相关 segment 当成独立样本。只有一条轨迹时，
`psd_std` 和 `psd_sem` 为 `NaN`，并标记不确定度不可用。

当 `find_peaks: true` 时，元数据还包含检测到的峰位与高度。

## `dist`

计算所选模式的边缘分布。

## `pdist`

计算所选可观测量的成对或高维分布。

## `lorentz_fitter`

对聚合后的 PSD 数据拟合 Lorentz 曲线。这是一个用于 `mode: analyze` 的**跨 job** 分析器。

### 配置

```yaml
analyser:
  lorentz_fitter:
    scan_param: omega_a
    mode: 0
    uncertainty: auto
    fit_window: [0.1, 0.2]
    freq_min: -0.1
    freq_max: 0.1
    clip_by_std: true
    clip_sigma: 10.0
    min_r2: 0.5
    export:
      - fit_results.csv
      - psd_merged.csv
```

| 键 | 类型 | 说明 |
| :-- | :-- | :-- |
| `scan_param` | `str` | 用于合并 PSD 的扫描参数。 |
| `mode` | `int` | 要拟合的模式索引。 |
| `uncertainty` | `auto \| required \| off` | `auto` 将 `psd_sem` 传播到参数协方差，旧载荷自动回退；`required` 拒绝缺少 SEM 的载荷；`off` 使用残差协方差。该选项不会改变拟合权重。 |
| `fit_window` | `list[float] \| None` | 手动 `[min, max]` 频率窗口。为 `None` 时，窗口由 `freq_min`/`freq_max` 或寻峰结果推导。 |
| `freq_min` / `freq_max` | `float \| None` | 可选的全局频率边界。 |
| `clip_by_std` | `bool` | 启用基于平方 PSD 加权的裁剪，忽略远端尾部。 |
| `clip_sigma` | `float` | 裁剪掉距离平方加权均值超过 `clip_sigma * std` 的样本。 |
| `min_r2` | `float` | 可接受的最小 `R^2`。 |
| `min_peak_height` | `float \| None` | 最小拟合峰高。 |
| `max_linewidth` | `float \| None` | 最大可接受 FWHM 线宽。 |
| `export` | `list[str]` | 要写入的 artifacts。默认为 `fit_results.csv`。 |

### 输出字段（`fit_results.csv`）

| 列 | 含义 |
| :-- | :-- |
| `scan_param` | 来自 `aggregate_input` 的扫描值。 |
| `center` | Lorentz 峰中心（rad/s）。 |
| `center_std` | 拟合峰中心的标准差。 |
| `linewidth` | 半高全宽（FWHM）。 |
| `linewidth_std` | `2 * gamma` 传播后的标准差。 |
| `base` | 常数基线。 |
| `base_std` | 基线标准差。 |
| `amplitude` | Lorentz 振幅。 |
| `amplitude_std` | 振幅标准差。 |
| `peak_intensity` | `amplitude + base`。 |
| `peak_intensity_std` | 包含 amplitude/base 协方差的标准差。 |
| `R2` | 决定系数。 |
| `reduced_chi2` | 使用 `psd_sem` 时的约化卡方；否则为 `NaN`。 |
| `uncertainty_source` | `psd_sem_sandwich` 或兼容旧结果的 `residual_covariance`。 |
| `status` | `ok` 或 `failed`。 |
| `error` | 拟合失败时的错误信息。 |
| `warning` | 诊断信息，如 std/FWHM 不匹配。 |

### 裁剪原理

PSD 数据通常覆盖由 `dt` 决定的很宽频率范围，而峰很窄。分析器用 `(PSD - min(PSD))^2` 作为权重计算频率轴的均值与标准差，然后丢弃 `mean ± clip_sigma * std` 之外的样本。这样可以去除无关尾部，同时保留峰及附近足够的连续谱，以估计稳定基线。

当平方加权 `std` 偏离 Lorentz 期望 `std ≈ linewidth / 2` 超过 2 倍时，会发出警告，这可能意味着存在多峰或频率分辨率不足。

PSD 不确定度不会改变无权 `curve_fit` 的目标函数或拟合参数。fitter 会在拟合参数
处计算 Lorentzian Jacobian，再通过 heteroscedastic sandwich covariance 传播
`psd_sem`。该计算把不同频点近似为独立样本；窗函数、频谱泄漏和有限轨迹动力学
可能使相邻频点相关，因此这些标准差使用的是对角输入协方差近似，而不是完整的
频谱协方差模型。
