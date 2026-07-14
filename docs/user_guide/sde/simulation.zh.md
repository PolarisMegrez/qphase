---
description: SDE 仿真指南
---

# SDE 仿真指南

`qphase_sde` 包提供了一个强大的计算引擎，用于在相空间中求解随机微分方程 (SDE)。其模块化设计允许用户灵活切换不同的积分方案和噪声模型。

## 概述

SDE 引擎主要求解以下形式的方程：

$$ d\mathbf{y} = \mathbf{a}(\mathbf{y}, t) dt + \mathbf{b}(\mathbf{y}, t) d\mathbf{W} $$

其中：
*   $\mathbf{y}$ 为状态向量（例如相空间坐标）。
*   $\mathbf{a}(\mathbf{y}, t)$ 为 **漂移 (Drift)** 向量。
*   $\mathbf{b}(\mathbf{y}, t)$ 为 **扩散 (Diffusion)** 矩阵。
*   $d\mathbf{W}$ 为维纳过程增量（噪声）。

## 仿真配置

要使用 SDE 引擎，需在任务配置文件（如 `job.yaml`）中进行相应配置。

```yaml
engine:
  sde:
    dt: 1e-3              # 时间步长
    t_max: 10.0           # 总仿真时间
    n_traj: 1000          # 轨迹数量
    integrator:           # 积分器配置
      name: "srk"         # 使用通用 SRK 求解器
      method: "heun"      # 具体方法 (heun, euler)
      tol: 1e-4           # 自适应步进的容差
    backend: "numpy"      # 计算后端 (numpy, torch, cupy)
```

### 关键参数

| 参数 | 类型 | 描述 |
| :--- | :--- | :--- |
| `dt` | `float` | 基础时间步长。对于固定步长求解器，此为实际步长；对于自适应求解器，此为初始步长猜测值。 |
| `t_max` | `float` | 仿真的结束时间（从 t=0 开始）。 |
| `n_traj` | `int` | 并行仿真的轨迹数量。 |
| `integrator` | `dict` | 数值求解器的配置。 |
| `backend` | `str` | 指定使用的计算后端。 |

## 积分器

框架支持多种积分方案。选择积分器时需要在精度、稳定性和计算开销之间权衡。

### 可用方法

| 积分器 | 随机解释 | 强收敛阶 | 每步漂移/扩散求值次数 | 典型使用场景 |
| :-- | :-- | :-- | :-- | :-- |
| `euler_maruyama` | Itô | 0.5 | 1 | 大样本集合、加性噪声、速度优先于精度。 |
| `heun` (SRK) | Stratonovich | ~1.0 | 2 | 乘性噪声、中等精度、参数扫描。 |
| `milstein` | Itô | 1.0 | 1 + Jacobian | 需要对角/可交换乘性噪声且需要强一阶精度。 |

#### Euler–Maruyama

*   **更新规则**：`dy = a(y)·dt + L(y)·dW`。
*   **优点**：每步只需一次漂移和扩散求值；单步计算成本最低；与 CuPy 上的融合漂移+扩散核函数配合最好。
*   **缺点**：强收敛阶仅为 0.5，误差随 `dt` 线性累积，对刚性或多噪声系统需要很小的时间步。当扩散项强依赖状态且 `dt` 较大时，可能出现不稳定。
*   **适用场景**：加性或弱乘性噪声；主要关注统计矩的长时间轨迹；需要最小化 kernel 启动次数的 GPU 批量任务。

#### 随机 Heun（SRK 方法 `heun`）

*   **更新规则**：使用 `y` 和预测值 `y_bar` 处的漂移与扩散做预测–校正。
*   **优点**：在 Stratonovich 解释下强收敛约 1.0 阶；对状态相关扩散比 Euler–Maruyama 更稳定；不需要 Jacobian。
*   **缺点**：每步需要两次漂移/扩散求值，计算量约为 Euler–Maruyama 的两倍。若 `dt` 过大，预测步可能放大瞬态。
*   **适用场景**：需要 Stratonovich 解释的乘性噪声；希望在不实现 Jacobian 的情况下获得比 EM 更好的路径精度。

#### Milstein

*   **更新规则**：`dy = a·dt + L·dW + 0.5·G·(dW² − dt)`，其中 `G` 是由扩散 Jacobian 构造的修正项。
*   **优点**：在 Itô 意义下强收敛 1.0 阶；无需像 Heun 那样做第二次求值即可捕捉乘性噪声的主导修正。
*   **缺点**：需要模型提供 `model.diffusion_jacobian`；Jacobian 求值可能较昂贵，且目前不在融合 CuPy 核函数覆盖范围内，因此 GPU 加速优势较小。当模型使用复噪声基且未提供兼容 Jacobian 时，当前实现会回退到 Euler–Maruyama。
*   **适用场景**：对角或可交换乘性噪声，需要 Itô 计算且强一阶精度。

### 稳定性与开销总结

*   **计算开销（低 → 高）**：`euler_maruyama` < `milstein`（Jacobian 便宜）≈ `heun` < `milstein`（Jacobian 昂贵）。
*   **强精度（低 → 高）**：`euler_maruyama` (0.5) < `heun` (~1.0) ≈ `milstein` (1.0)。
*   **乘性噪声稳定性**：`euler_maruyama` 对 `dt` 最敏感；`heun` 和 `milstein` 可容忍更大的步长。
*   **GPU 批量化**：`euler_maruyama` 最受益于融合核函数，因为每步只需一次融合漂移+扩散求值。`heun` 目前需要两次融合求值；若实现完全融合的 Heun 核函数，可缩小这一差距。

### 自适应步进

`srk` 积分器支持基于 Richardson 外推（步长加倍法）的 **自适应步进**。这允许求解器在误差较高时（例如快速动力学期间）自动减小步长 `dt`，并在系统趋于稳定时增大步长，从而在保证精度的同时提高效率。

要启用自适应步进，只需在积分器配置中提供 `tol` (tolerance) 参数。

```yaml
integrator:
  name: "srk"
  method: "heun"
  tol: 1e-5  # 启用自适应步进，目标误差设为 1e-5
```

**注意**：即使启用了自适应步进，引擎也会对结果进行插值，以按照 `dt` 和 `return_stride` 定义的固定间隔保存数据。这确保了输出数据始终位于规则的时间网格上，便于后续分析。

## 自定义模型

若需仿真自定义系统，用户需要定义一个实现 `SDEModel` 协议的模型类。这涉及指定 `drift` 和 `diffusion` 函数。

有关如何编写和注册自定义模型的详细信息，请参阅 [插件开发](../../dev_guide/plugin_development.md) 指南。

## 批量化参数扫描

当任务包含参数扫描（例如 `model.omega_a: [0.001, 0.002, 0.003]`）时，调度器通常会将其展开为多个独立的仿真任务。如果这些任务具有**相同的引擎、积分器、后端和时间网格**，仅在模型参数上不同，`qphase_sde` 可以将它们融合为一次批量仿真。

在批量运行中：

* 扫描值会被广播为单个 `(n_scan * n_traj,)` 的集合体。
* 后端的一次调用会同时推进所有扫描点上的所有轨迹。
* 结果会自动拆分回原始的逐扫描任务，每个任务仍保留自己的运行目录和清单条目。

这在 GPU 上效果尤为明显：每个时间步较小的 CPU 启动开销被大量轨迹均摊，单个 CuPy kernel 启动即可更新整个集合体。

批量执行是**自动的**，不需要额外配置。对模型的唯一要求是其参数既能接受标量，也能接受一维数组，以便规划器将扫描值广播到整个集合体。

## Kernelized Terms（CuPy 核函数）

部分内置模型为 CuPy 后端提供了可选的**融合漂移+扩散核函数**。当核函数可用且任务使用 `backend: cupy` 时，积分器会优先使用核函数路径，而不是分别调用 Python 的 `drift` 和 `diffusion` 方法。这样可以消除 Python 层调度开销，并将两项计算融合到一次 CUDA 启动中。

当前支持 CuPy 核函数的模型：

* `model.vdp_2mode`
* `model.kerr_3mode`

核函数化是**自动的**；任务文件中不需要显式开关。如果模型声明支持当前后端，积分器就会使用它；否则会回退到标准 Python 实现。因此，在 `numpy` 和 `cupy` 之间切换 `backend` 始终能产生正确结果。

### 示例：CuPy 核函数工作流

```yaml
engine:
  sde:
    t0: 0.0
    t1: 2000.0
    dt: 0.1
    n_traj: 20
backend:
  cupy:
    float_dtype: float32
    device: cuda
model:
  vdp_2mode:
    omega_a: [0.001, 0.00251189, 0.01]
    omega_b: 0.0
    gamma_a: 2.0
    gamma_b: 1.0
    Gamma: 0.00001
    g: 0.5
    D: 1.0
```

由于 `omega_a` 有三个取值且后端为 CuPy，调度器会对这三个扫描点进行批量化，`vdp_2mode` 的核函数将以融合的 CUDA kernel 同时计算全部 `3 * 20` 条轨迹。
