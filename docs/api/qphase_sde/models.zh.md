---
layout: default
title: 模型
parent: qphase_sde
grand_parent: API 参考
nav_order: 3
---

# SDE 模型

模型插件描述如下形式的随机微分方程：

```text
dy = drift(t, y) dt + diffusion(t, y) dW
```

模型在 `qphase_sde.models` 下注册，并通过任务配置中的 `model.<name>` 选择。

## `SDEModel` 协议

```python
class SDEModel(Protocol):
    dim: int
    noise_dim: int

    def drift(self, t: float, y: np.ndarray) -> np.ndarray:
        ...

    def diffusion(self, t: float, y: np.ndarray) -> np.ndarray:
        ...
```

*   `dim` — 状态空间维度。
*   `noise_dim` — 每步独立维纳增量的数量。
*   `drift` 返回形状为 `(dim,)` 的向量。
*   `diffusion` 返回形状为 `(dim, noise_dim)` 的矩阵。

引擎每步计算 `drift` 和 `diffusion`，并将其传给所选积分器。

## 内置模型

### `vdp_level3`

一种用于窄峰基准测试的 Van der Pol / Kerr 风格双模模型。

关键参数：

*   `omega_a`、`omega_b` — 模式频率。
*   `gamma_a`、`gamma_b` — 阻尼率。
*   `Gamma` — 非线性增益参数。
*   `g` — 耦合强度。
*   `D` — 噪声强度。

典型用法：

```yaml
model:
  vdp_level3:
    omega_a: 0.00251189
    omega_b: 0.0
    gamma_a: 2.0
    gamma_b: 1.0
    Gamma: 0.00001
    g: 0.5
    D: 1.0
```

### `kerr_3pa` 与 `kerr_3mode`

Kerr 非线性三光子吸收与三模模型。参数列表参见 [模型源码](https://github.com/your-org/qphase/tree/main/models) 或包参考。

## 添加新模型

继承 `SDEModel` 并注册入口点：

```toml
[project.entry-points."qphase.models"]
my_model = "my_pkg.models:MyModel"
```

模型类由任务配置中的 `model.<name>` 块实例化。
