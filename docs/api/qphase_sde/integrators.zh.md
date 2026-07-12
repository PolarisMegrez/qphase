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
    method: heun   # 或其他支持的方法
```

支持的 `method` 值包括 `heun` 与 `qphase_sde` 内置的其他 SRK 变体。引擎在每个积分区间调用 `GenericSRK.step(...)`。

## 注册自定义积分器

创建一个插件包，并在 `pyproject.toml` 中配置入口点：

```toml
[project.entry-points."qphase.integrators"]
my_srk = "my_pkg.integrators:MySRK"
```

类必须满足 `Integrator` 协议。完整插件打包说明参见 [开发者指南](../../dev_guide/plugin_development.zh.md)。
