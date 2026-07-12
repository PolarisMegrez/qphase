---
layout: default
title: 引擎
parent: qphase_sde
grand_parent: API 参考
nav_order: 1
---

# SDE 引擎

SDE 引擎（`qphase_sde.engine.Engine`）负责协调积分循环、数据存储与可选的逐步分析。

## `EngineConfig`

在任务配置中，这些键位于 `engine.sde` 下：

| 键 | 类型 | 说明 |
| :-- | :-- | :-- |
| `dt` | `float` | 积分步长，必须足够小以保证稳定。 |
| `t0` | `float` | 起始时间。 |
| `t1` | `float` | 结束时间。 |
| `n_traj` | `int` | 系综轨迹数。 |
| `seed` | `int \| None` | 随机种子，用于可复现。 |
| `ic` | `Any \| None` | 初始条件。 |
| `save_stride` | `int` | 每 `N` 个积分步保存一次，见下文。 |
| `keep_traj` | `bool \| None` | 分析后是否保留原始轨迹。 |

## `save_stride` 与内存控制

`save_stride` 允许积分器使用保证稳定的小 `dt`，但只保存（并用于 FFT）每 `N` 个样本。保存后的轨迹有效采样间隔为 `dt * save_stride`，这会收窄 PSD 的 Nyquist 频率，但**不改变**真正的频率分辨率：

```text
df = 1 / t1                                    # 频率分辨率（不变）
f_Nyquist = pi / (dt * save_stride)            # Nyquist 频率（降低）
```

存储轨迹的粗略内存估算：

```text
内存 ~ n_traj * (t1 / (dt * save_stride)) * n_modes * 单个元素字节数
```

对于窄的低频峰，选择 `save_stride` 时应使 `f_Nyquist` 远高于感兴趣的最高频率，以避免混叠。例如 `dt=0.1`、峰位在 `0.1` rad/s 附近时，`save_stride=50` 给出 `f_Nyquist ~ 0.63` rad/s，已经足够。

```yaml
engine:
  sde:
    t0: 0.0
    t1: 10000.0
    dt: 0.1
    save_stride: 50
    n_traj: 100
```

## `mode: analyze`

设置 `engine.sde.mode: analyze` 后，引擎不会对上游输入执行新的仿真，而是直接运行配置的分析器。这常用于跨 job 后处理，例如对聚合后的 PSD 做 Lorentz 拟合：

```yaml
- name: fit
  input: sim
  aggregate_input:
    on: params.epsilon
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: epsilon
      mode: 0
```

## `SDEResult`

引擎返回并保存 `SDEResult`，格式为 NumPy `.npz`：

*   `trajectory` — `TrajectorySet`，若分析后丢弃原始数据则为 `None`。
*   `analysis` — 以分析器名称为键的载荷，如 `psd`、`dist`、`pdist`。
*   `meta` — 元数据，包括模型 `params`、`t0`、`dt` 以及丢弃原因。

保存的归档包含 `t0`、`dt`、`meta`、`analysis`；在保留轨迹时还包括形状为 `(n_traj, n_time, n_modes)` 的 `data`。
