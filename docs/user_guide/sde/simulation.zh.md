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

框架通过 `GenericSRK` (Stochastic Runge-Kutta) 类支持多种积分方案。

### 可用方法

*   **`euler` (Euler-Maruyama)**：
    *   **阶数**：强收敛 0.5 阶，弱收敛 1.0 阶。
    *   **适用场景**：简单的加性噪声，或对计算速度要求较高且不需要极高精度的场景。
    *   **随机积分解释**：Ito。

*   **`heun` (Stochastic Heun)**：
    *   **阶数**：强收敛约 1.0 阶，弱收敛 2.0 阶。
    *   **适用场景**：需要 Stratonovich 解释的乘性噪声系统。
    *   **随机积分解释**：Stratonovich。

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
