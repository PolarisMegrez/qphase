---
layout: default
title: 分析器
parent: qphase_sde
grand_parent: API 参考
nav_order: 4
---

# 分析器

分析器在 SDE 积分循环结束后运行，其载荷成为该 job `SDEDataBundle` 的命名 product。它们也可以在 `mode: analyze` 下调用，用于后处理。

## `psd`

估算所选模式的功率谱密度（PSD）。

### 配置

```yaml
analyser:
  psd:
    modes: [0]
    kind: complex
    orientation: phase_decreasing
    expected_freq_max: 0.34
    find_peaks: true
    estimator:
      periodogram:
        window: null
```

| 键 | 类型 | 说明 |
| :-- | :-- | :-- |
| `modes` | `list[int]` | 要分析的模式索引。 |
| `kind` | `str` | PSD 变体，如 `complex`、`real`、`imag`。 |
| `orientation` | `str` | 正频率方向。`phase_decreasing` 将 `exp(-i*omega*t)` 映射到 `+omega`；`phase_increasing` 保留前向 FFT 频率轴。 |
| `expected_freq_max` | `float \| None` | 输出频率轴单位下预期的最大频率幅值；达到 Nyquist 上限时分析直接失败。 |
| `find_peaks` | `bool` | 是否报告峰值位置。 |
| `estimator` | 子插件选择 | 必须且只能选择 `periodogram`、`welch`、`multitaper` 之一。 |

Estimator 对比：

| Estimator | 子配置字段 | 分辨率与开销 |
| :-- | :-- | :-- |
| `periodogram` | `window` | 使用完整保存时长；频率分辨率最高，单 trajectory FFT 最大。 |
| `welch` | `window`、`nperseg`、`noverlap`、`nfft` | 方差较低，segment FFT 内存有界；物理分辨率由 `nperseg * sample_dt` 决定，`nfft > nperseg` 只插值频点。 |
| `multitaper` | `nw`、`k_tapers` | 保持完整时长分辨率并平均 taper；每个 taper 需要额外 FFT。 |

三者均支持跨 trajectory batch 的在线聚合。Estimator 是 `analyser.psd` 的
子插件，可通过 `qphase list --parent analyser.psd` 发现，并通过
`qphase config schema analyser.psd/estimator.welch` 查询 schema。

### 频率轴

Estimator 首先执行 NumPy/CuPy 标准前向 DFT，随后按配置重排输出频率轴：

```text
omega_fft = np.fft.fftfreq(n_saved, dt * save_stride) * 2 * pi
phase_decreasing: omega = -omega_fft，并按升序重排
phase_increasing: omega = +omega_fft，并按升序重排
```

QPhase 默认采用 `phase_decreasing`。对于
`C(tau) = <a^dagger(t) a(t+tau)>`，它对应量子光学发射谱
`S(omega) = integral C(tau) exp(+i*omega*tau) d tau`，因此轨迹
`a(t) ~ exp(-i*omega0*t)` 的峰位是正载频 `+omega0`。
`phase_increasing` 用于一般信号处理或严格复现前向 FFT 轴。该选项只改变频率轴方向，
不改变 PSD 归一化、线宽或积分功率。

配置输入也接受俗名：`physical` 等价于 `phase_decreasing`，`fft` 等价于
`phase_increasing`。它们只用于输入；序列化配置和结果元数据始终使用正式名称。
此处 `physical` 仅表示 QPhase 默认约定，并不表示跨领域唯一的物理约定。

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
*   `orientation`、`positive_frequency_time_dependence` 与 `spectrum_kernel`
    — 明确记录频率符号定义的元数据。

`lorentz_fitter` 会把相同元数据传递到结果，并在 `fit_results.csv` 中增加
`orientation` 列；若输入混用了两种方向，则明确拒绝，而非静默拟合不兼容的频率轴。

对于 Welch 和 multitaper，先在每条轨迹内部平均 segment 或 taper，再跨轨迹
计算不确定度，因此不会把相关 segment 当成独立样本。只有一条轨迹时，
`psd_std` 和 `psd_sem` 为 `NaN`，并标记不确定度不可用。

当 `find_peaks: true` 时，元数据还包含检测到的峰位与高度。

## `coherence_carrier`

从固定实验读出通道的短延迟一阶相干函数估计载频。插件随模拟运行，支持
trajectory batching；设置 `keep_traj: false` 时不会向磁盘保存轨迹：

```yaml
analyser:
  coherence_carrier:
    modes: [0]
    include_trace: true
    channels:
      bright: ["0+0j", "0.70710678+0j", "0.70710678+0j"]
    polynomial_order: 2
    minimum_lag_points: 4
    maximum_lag_points: 12
```

对半正定读出矩阵 `W`，估计量定义为

```text
C_W(tau) = mean(alpha(t)^dagger W alpha(t + tau))
omega_W = orientation_sign * Im[C_W'(0+) / C_W(0)].
```

`modes` 生成裸模式投影，`include_trace` 在已记录模式子空间使用 `W=I`；固定相干读出
`c=l^dagger alpha` 使用 `W=l l^dagger`。通道向量按物理模式编号排列，约定与 CAM
的 `coherence_pole_spectrum` 后处理器相同。权重非零的模式必须全部包含在
`engine.sde.record_modes` 中。需要与 CAM 的完整 trace 对应时，必须记录模型的全部模式。

该观测量与 CAM 具有直接对应。对漂移
`d alpha/dt = -i H(R) alpha + noise`，平稳性与 CAM 矩闭合给出

```text
C_W'(0+) = -i Tr[W H(R) R].
```

因此，在默认 `phase_decreasing` 方向下，输出退化为广义 Rayleigh 商

```text
Re Tr[W H(R) R] / Tr[W R].
```

实现只计算少量短延迟相关函数，以约束通过零延迟的局部相位多项式拟合嵌套窗口，
并在逐轨迹 jackknife 误差尺度内选择与较短窗口一致的最大窗口。点估计始终先对轨迹
相关函数作系综平均再取比；逐轨迹量仅用于 jackknife。输出包括频率及其 SEM、
`recorded_modes`、该基下的读出矩阵、选定延迟、全部嵌套候选、相位残差、首个延迟点
相干度和 Nyquist 占比诊断。

该估计器**不**分解长时极点、不计算线宽，也不假定 Lorentz 线型。它要求采样轨迹
平稳、短延迟分辨率足够且读出强度为正。`nyquist_fraction` 较高、嵌套窗口不稳定或
首延迟相干度很低时，应减小保存采样间隔。输出是有限探测带宽下可操作的一阶相干载频；
多个谱分量共存时，它不保证等于每个谱峰的中心。

## `band_limited_carrier`

该插件从已有 PSD dataset 估计经过实验带宽筛选后的长时一阶相干载频。它作为
`engine.sde.mode: analyze` 的下游 analyser 运行，不改变 `coherence_carrier`
短延迟插件：

```yaml
- name: carrier
  input: {from: sim, mode: dataset}
  engine:
    sde: {mode: analyze}
  analyser:
    band_limited_carrier:
      scan_param: omega_a
      readout: trace
      freq_min: -0.75
      freq_max: 0.2
      bandwidth_multipliers: [0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0]
      minimum_lag_span: 24.0
      max_phase_fit_rms: 0.05
      tracking_enabled: true
```

`readout` 可以是已记录的物理 mode，也可以是 `trace`，即所有已记录 mode 的
非相干 PSD 之和。搜索频带属于测量定义的一部分，应排除无关的远端频带。

估计器先扣除稳健基线，再以平方剩余谱的标准差定义谱集中宽度；对理想、未截断的
Lorentz 线型，该宽度等于 HWHM。随后生成嵌套余弦边缘通带并分别重建带限
`G^(1)`。每个带宽内搜索连续延迟子区间，只接受加权相位残差和二次频率漂移均通过
门限的线性相位窗口；之后要求频率在有限 `log(bandwidth)` 跨度内形成平台。
算法不再在失败时回退到 `1x spectral_width`：不可辨识点返回 `NaN`，而不是任意
候选值。

局部状态包括 `ok`、`ambiguous_multiband` 和 `no_bandwidth_plateau`；候选层还会
记录 `nonlinear_phase` 等具体拒绝原因。只有唯一局部平台时才填写 `frequency`。
`carrier_candidates.csv` 保存全部带宽/延迟估计，`carrier_platforms.csv` 保存所有
竞争平台。

启用 `tracking_enabled` 后，analyser 还会按扫描轴顺序，利用局部平台质量和
divided-difference 曲率跟踪连续谱组分，输出 `tracked_frequency`、
`tracked_platform_index` 与 `tracked_status`，但不覆盖局部结果。路径代价禁止使用
CAM、目标幂指数或理论频率；平台缺失会中断路径，过大曲率标为
`discontinuous_path`，不会强制连接。

`regression_std` 是给定延迟窗口下的 HAC 条件回归误差，`bandwidth_std` 是同一平台
内的带宽敏感度，均不是 trajectory SEM。`diagnostic_uncertainty` 仅组合这两个条件
诊断量。正式采样不确定度仍需逐轨迹充分统计或独立重复运行；仅凭当前平均 PSD 无法
事后重建。

可选的 `center.spectral_ridge` 模式会以每条数据驱动 retained ridge 为中心，分别执行
相同的延迟—带宽平台检验：

```yaml
band_limited_carrier:
  scan_param: omega_c
  readout: trace
  freq_min: -0.3
  freq_max: -0.1
  tracking_enabled: false
  center:
    maximum_neighbor_fraction: 0.45
    spectral_ridge:
      scan_param: omega_c
      readouts: [trace]
      minimum_prominence_fraction: 0.03
      tracking_path_count: 2
```

算法先把 ridge 中心作为粗略本地振荡器频率，再执行相位解缠。每条 retained ridge
输出一行。通带上限不超过最近多尺度 ridge 距离的 `maximum_neighbor_fraction`；该邻峰
集合也包含最终关联时排除的候选。若仍有多个延迟—带宽平台，则选择最接近数据驱动
ridge 中心者，全程不读取 CAM 或理论频率。`ridge_carrier_correction` 是相对 ridge
中心的精细相位频率修正。

`ridge_conditioned_uncertainty_upper` 保守相加条件载频诊断误差与 ridge 中心标准差。
由于二者来自同一系综 PSD，该字段是诊断上界，不是 trajectory-bootstrap SEM。

## `spectral_ridge`

该插件不假设 Lorentz 线型，也不使用模型目标频率。它构造一维 Gaussian scale
space，以局部二次拟合精化峰位，将多个平滑尺度共同支持的峰聚类为候选谱脊，并可仅
依据峰证据和频率连续性在参数扫描中跟踪一条路径。

```yaml
analyser:
  spectral_ridge:
    scan_param: omega_c
    readouts: [0, 1, 2, trace]
    freq_min: -0.3
    freq_max: -0.1
    tracking_gap_factor: 1.5
```

`tracking_gap_factor` 必须显式启用，用于在遗漏分岔点等显著扫描轴缺口处分段；普通
不规则或对数扫描应保持未设置。谱脊选择绝不读取 CAM 频率。

tracking 保留代价最低的 `tracking_path_count` 条纯数据路径，并用 Huber 跳变损失限制
真实快速频移对连续性代价的影响。严格候选满足相对峰高、尺度支持和曲率门槛；属于
与最优路径代价相差不超过 `tracking_max_cost_delta` 的高峰候选会标记为
`continuity_rescued`。若两类均为空，则恰好保留一个证据分数最高的
`fallback_low_confidence` 候选。候选 CSV 明确保存保留层级、路径排名和最佳路径代价差；
这些字段只定义可供下游关联的候选，不使用或暗示任何模型目标。

输出分别记录局部峰位传播误差、跨平滑尺度漂移、曲率及其显著性、基于 PSD SEM 的
峰位置信区间，以及描述性的相对峰高平台。另一个歧义包络覆盖峰高达到最强候选
`plateau_fraction` 的全部多尺度候选；它是模型选择诊断而非置信区间。
`frequency_bin_covariance: diagonal` 对无窗
periodogram 使用常见的频率 bin 渐近对角近似；`conservative` 将所有频率 bin 误差视为
完全相关。由于结果未保存跨 mode PSD 协方差，trace SEM 始终采用各 mode SEM 之和的
保守上界。

## `finite_delay_carrier`

该插件从完整保存 PSD 重建一阶相干函数 `G(tau)`，并对探测器速率 `kappa` 计算

```text
Omega(kappa) = integral exp(-2*kappa*tau)
                 Im[conj(G) dG/dtau] dtau
               / integral exp(-2*kappa*tau) |G|^2 dtau。
```

```yaml
analyser:
  finite_delay_carrier:
    scan_param: omega_c
    readouts: [0, 1, 2, trace]
    detector_rates: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    maximum_lag: 4096.0
```

`readouts` 可在一次 dataset 遍历中同时计算多个已记录裸 mode 和非相干 trace；仅选择
一个读出时仍可使用单数形式 `readout`。相干叠加通道需要交叉谱信息；若积分时仍有原始
轨迹，应使用 `coherence_carrier.channels` 接口。

即使多个 pole 相互干涉，该量仍是完整的有限带宽探测器载频，不要求选定某一 pole
或假定 Lorentz 线型。增大 `kappa` 时，direct SDE 结果趋于真实 SDE instantaneous
coherence carrier；矩闭合失效时，该极限一般不等于 CAM Rayleigh 商。

对应 CAM postprocessor 使用相同速率和全部闭合 CAM pole residue，其零延迟极限
严格等于 generalized Rayleigh quotient。`finite_delay_carrier.csv` 保存探测器速率、
载频、读出名称与类型、瞬时极限、有限延迟修正、相干权重和数值延迟范围。仅凭系综
平均 PSD 无法得到正式 sampling uncertainty。

## `coherence_matrix`

计算系综一阶相干矩阵

```text
R_ij = mean_trajectory,time(alpha_i * conj(alpha_j))
rho_R = R / Tr(R)
P_R = Tr(rho_R^2).
```

这里的 `P_R` 是归一化模态相干矩阵的纯度，不是完整多体量子密度算符的纯度。

```yaml
analyser:
  coherence_matrix:
    modes: [0, 1, 2]
    time_blocks: 8
    min_block_samples: 32
    time_chunk_samples: 8192
    confidence_level: 0.95
```

输出包括 `matrix`、`normalized_matrix`、本征值、`purity`、有效秩、谱熵、
主模占比、连通协方差和归一化一阶相干度。矩阵元 SEM 以每条独立轨迹的
时间平均为统计单位；纯度不确定度使用逐轨迹留一 jackknife，输出
`purity_sem` 和 `purity_ci`。

连续 `time_blocks` 用于报告矩阵、纯度、迹和漂移距离，以检查稳态收敛；
插件不会把相关时间块当作独立样本。该 analyser 支持 trajectory batching，
可以与 `keep_traj: false` 配合，只保留紧凑矩阵统计量。`time_chunk_samples`
限制后端 workspace，选择模式时不会再物化第二份完整轨迹。

插件不自动实施相空间 ordering 修正。对于 Wigner 轨迹，输出遵循当前模型和
CAM 方程使用的原始 amplitude 约定；若需要 normal-order 修正，应由显式、
model-aware 的分析实现，而不是在通用插件中静默减去 `1/2`。

`R` 所需的全部模式必须包含在 `engine.sde.record_modes` 中。省略 `modes`
表示分析所有已记录模式。

## `moment_statistics`

该 analyser 从记录的复振幅计算通用的 c-number 占据数矩：

```text
n_i = |alpha_i|^2
G2_ij = mean_trajectory,time(n_i n_j)
g2_ij = G2_ij / (mean(n_i) mean(n_j)).
```

```yaml
analyser:
  moment_statistics:
    modes: [0, 1, 2]
    time_blocks: 16
    time_chunk_samples: 8192
```

输出包括各模式占据数、四阶矩、全部跨模式占据数乘积、协方差、归一化 `g2`、
逐轨迹 SEM 和连续时间块平稳性诊断。`g2_sem` 使用逐轨迹留一 jackknife。
插件支持 trajectory batching，并在合并全部批次后重新计算非线性比值。

`time_chunk_samples` 限制后端临时内存。插件不会物化完整的 `|alpha|^2` 时序，
所以达到该分块长度后，其 workspace 不随总观测时间继续增长。插件不自动执行
Wigner 到 normal ordering 的修正，输出明确遵循模型原始 c-number 约定。

## `quadratic_moments`

统计命名 Hermitian 二次型的一至四阶矩：

```text
x_o = alpha^dagger Q_o alpha - center_o
    = Tr[Q_o (alpha alpha^dagger - R_ref,o)]。
```

```yaml
analyser:
  quadratic_moments:
    modes: [0, 1]
    max_order: 4
    time_blocks: 16
    time_chunk_samples: 8192
    observables:
      population_difference:
        matrix:
          - ["1+0j", "0+0j"]
          - ["0+0j", "-1+0j"]
        center: 0.0
      coherent_quadrature:
        matrix:
          - ["0+0j", "0.5+0.25j"]
          - ["0.5-0.25j", "0+0j"]
        reference_matrix:
          - ["1+0j", "0+0j"]
          - ["0+0j", "1+0j"]
```

每个矩阵必须为 Hermitian，并使用 `modes` 声明的物理模式顺序。
`center` 与 `reference_matrix` 互斥；参考矩阵会转换为
`center=Tr(Q R_ref)`。结果包含 raw moments、central moments、cumulants、
逐轨迹 raw moments 和连续时间块统计。raw moment SEM 以独立轨迹的时间均值
为样本；累积量不确定度使用逐轨迹 leave-one-out jackknife。

插件支持 trajectory batching，并在全部批次合并后重新计算非线性累积量。
后端工作区受 `time_chunk_samples` 限制，因此可与 `keep_traj: false` 配合，
不会物化完整二次型时序。时间块只用于平稳性诊断，不视为独立样本；插件也不
自动进行相空间排序修正。

## `dist`

计算所选模式的边缘分布。

## `pdist`

计算所选可观测量的成对或高维分布。

## `trajectory_diagnostics`

在不预设 Lorentz 或其他谱线模型的前提下计算时间域诊断量。它用于在解释 PSD 拟合线宽
之前区分非平稳性、轨迹间差异、相干衰减和相位频率噪声。

```yaml
analyser:
  trajectory_diagnostics:
    modes: [0]
    orientation: phase_decreasing
    block_durations: [100.0, 1000.0]
    coherence: true
    coherence_max_lag: 500.0
    allan: true
    allan_taus: null
    allan_points: 24
    allan_min_windows: 8
    amplitude_floor: 0.0
```

相位增量均值和复数 block spectrum 峰值遵循与 `psd` 相同的 `orientation`。
复数 coherence 保留原始时域相位；Allan 方差由相位二阶差分的平方构成，因而不受整体频率反号影响。

输出的 `mode_results[mode]` 包含：

*   `block_statistics`：逐轨迹、非重叠分块的复振幅、振幅、功率与角频率均值。
*   `phase_increment`：逐轨迹平均角频率、最大保存相位步，以及落在 Nyquist 相位边界
    10% 范围内的步数比例。
*   `coherence`：随延迟变化的复数 `g1`、归一化 `g1` 与跨轨迹 SEM 模长。
*   `allan`：基于重叠相位二阶差分的角频率 Allan 方差、逐轨迹结果、系综均值与跨轨迹 SEM。

所有配置时长都是物理时间，必须与保存后的采样间隔 `dt * save_stride` 对齐。自动 Allan
时标使用对数分布的整数采样点。SEM 只把不同轨迹作为独立统计单元，不会把重叠时间窗
计作额外独立样本。

若 Lorentz 窄核的 HWHM 接近软模衰减率，它描述的是稳态中每次随机扰动后的恢复，而不只是
配置初态的一次性弛豫。设置非零 engine `t0` 可在 PSD 和本 analyser 之前排除该初态瞬态。

首版实现会在主机端物化传入的轨迹系综，尚不支持跨 trajectory batch 在线聚合，因此应先
用于减量诊断任务；流式聚合留待后续阶段。

## `allan_variance`

当不需要 coherence 等完整轨迹诊断时，本 analyser 专门计算角频率 Allan 统计量。它与
`trajectory_diagnostics` 不同，支持 trajectory batching，可与 `keep_traj: false` 配合：

```yaml
analyser:
  allan_variance:
    modes: [0]
    orientation: phase_decreasing
    points: 40
    min_windows: 8
    min_independent_windows: 4
    transfer_chunk_samples: 8192
```

`orientation` 控制逐轨迹平均角频率的符号。Allan 方差本身不随该配置改变。
`allan_scaling` 会将方向信息传递到逐点结果、汇总与导出，并拒绝混合方向的 scan。

每个 tau 同时输出既有的重叠估计和非重叠相位二阶差分估计，包括逐轨迹结果、跨轨迹
SEM、实际有效非重叠窗口数以及每条轨迹的名义窗口数。此处“独立”只表示时间块不重叠；
有色动力学仍可能使相邻块相关。因此，不应把所有块当作可交换样本，指数不确定度优先
使用逐轨迹 bootstrap。

使用设备后端时，插件按模式逐个传输，并用 `transfer_chunk_samples` 限制每次传输的
时间块长度。这不会改变 Allan 定义，同时避免在主机端复制全部已记录模式。

## `allan_scaling`

在 `mode: analyze` 中消费逻辑 SDE scan dataset。它逐参数点检测长时白 FM 区，再对连续
微扰点取白区交集，计算 `N_A = tau * angular_frequency_allan_variance`，并拟合不带背景的
`N_A = C * abs(epsilon) ** (-q)`。指数区间由逐轨迹 bootstrap 给出。相位增量的平均频率
单独按 `omega = omega0 + A * abs(epsilon) ** p` 拟合。线性模型仅作为判断非线性是否
可分辨的零假设；幂律模型中不加入额外线性修正项。

```yaml
analyser:
  allan_scaling:
    scan_param: omega_c
    critical_value: 1.0
    mode: 0
    min_scaling_points: 5
    target_scaling_decades: 1.0
    normal_form: {n: 3, k: 1, m: 0, observable_order: 2}
```

对于正规形 `epsilon**k * x**m + x**n = 0`，预期频移指数为
`observable_order * k / (n - m)`；在临界投影噪声保持规则白噪声时，预期 Allan 强度指数为
`2 * (n - observable_order) * k / (n - m)`。最终 `status: ok` 同时要求 epsilon
跨度充分、Allan 幂律拟合合格、频移非线性可辨认，且指数符合配置的正规形。插件输出
`allan_points.csv` 和
`allan_scaling.json`。旧 `trajectory_diagnostics` 结果仍可读取，但其独立窗口数会明确标记
为估算值，而不是实测非重叠计数。

## `lorentz_fitter`

对逻辑 SDE scan dataset 的 PSD point view 拟合 Lorentz 曲线。这是一个用于
`mode: analyze` 的下游分析器。

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
| `scan_param` | 从逻辑 SDE dataset 读取的命名 scan axis。 |
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
