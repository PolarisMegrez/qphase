---
layout: default
title: 积分器
parent: qphase_sde
grand_parent: API 参考
nav_order: 2
---

# 积分器

SDE 引擎对积分器没有强制要求。任何实现 `Integrator` 协议的对象都可以在 `qphase_sde.integrators` 下注册，并通过任务配置中的 `integrator.<name>` 选择。

## `Integrator` 协议

积分器必须暴露：

```python
class Integrator(Protocol):
    name: str

    def step(
        self,
        model: SDEModel,
        t: Real,
        y: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """返回 t+dt 时刻的状态。"""
```

`step` 接收：

*   `model` — `SDEModel` 实例。
*   `t` — 当前时间。
*   `y` — 当前状态数组。
*   `dt` — 积分步长。
*   `rng` — `numpy` 随机数生成器。

积分器自行决定如何计算确定项和随机项。关于 `SDEModel.drift`、`SDEModel.diffusion` 与 `SDEModel.noise_dim` 的详情，参见 [模型](./models.zh.md)。

## `GenericSRK`

`GenericSRK` 通过不同的 Butcher 表实现一族随机 Runge-Kutta 方法。

### 配置

```yaml
integrator:
  srk:
    method: heun   # 或 euler
```

### 方法对比

| 方法 | 强收敛阶 | 随机解释 | 每步求值次数 | 适用场景 |
| :-- | :-- | :-- | :-- | :-- |
| `euler` | 0.5 | Itô | 1 | 加性噪声、速度优先。 |
| `heun` | ~1.0 | Stratonovich | 2 | 乘性噪声、中等精度。 |

独立的 `integrator.euler_maruyama` 插件提供与 SRK `euler` 方法相同的数值行为，适用于希望使用专用积分器命名空间而不是通用 SRK 分派器的场景。

引擎在每个积分区间调用 `GenericSRK.step(...)`。

## `CayleyMaruyama`

`integrator.cayley_maruyama` 是面向矩阵漂移 `A(y,t) @ y` 的固定步长 Itô
积分器：

```text
(I - dt*A_n/2) y_(n+1) = (I + dt*A_n/2) y_n + B_n dW_n
```

`A_n` 与 `B_n` 都由左端点状态计算。对于中性振荡本征模，Cayley 变换严格
保持放大因子的模长，避免显式 Euler 引入虚假的径向增益。

```yaml
integrator:
  cayley_maruyama:
    fused: auto       # auto、required 或 off
    chunk_steps: 128  # 1 表示禁用多步融合
    max_modes: 16     # 可配置至 64
```

通用路径使用 backend 的批量线性求解，支持任意小规模模式数。模型可以提供
专用 fused step 或 chunk kernel。GPU 生产任务建议使用 `fused: required`，避免
加速实现缺失时静默回退到通用路径。

`ChunkIntegrator` 是可选能力。只有固定步长任务且 model/backend 支持相同
scheme 时，SDE engine 才会启用多步融合；已有积分器仍使用普通 `step()` 路径。

对于很长的 `complex64` 轨迹，即使已消除 Euler 偏差，舍入误差累积仍可能留下
很小的频率残差。在 VDP 的 `omega_a=0.001` 验证中，该残差约为 `5e-6`；同一
fused kernel 使用 `complex128` 时与 Cayley 色散关系符合到机器精度。只有当该
残差比 GPU 吞吐与轨迹内存更重要时，才建议改用 `float64`。

## 注册自定义积分器

创建一个插件包，并在 `pyproject.toml` 中配置入口点：

```toml
[project.entry-points."qphase.integrators"]
my_srk = "my_pkg.integrators:MySRK"
```

类必须满足 `Integrator` 协议。完整插件打包说明参见 [开发者指南](../../dev_guide/plugin_development.zh.md)。
