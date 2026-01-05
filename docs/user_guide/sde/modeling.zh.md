---
description: SDE 物理建模
---

# SDE 物理建模

QPhase SDE 采用三层架构来对物理系统进行建模。这种分层设计实现了物理定义（量子/相空间）与数值实现（SDE）的解耦，使得模型定义更加清晰且易于扩展。

## 三层架构

### 第一层：主方程 (Master Equation)
**对应类：** `qphase_sde.model.MasterEquation`

使用哈密顿量 ($\hat{H}$) 和 Lindblad 塌缩算符 ($\hat{L}_k$) 在希尔伯特空间中描述系统。这是最基础的物理描述层级。

*   **适用场景：** 从第一性原理 (First Principles) 定义物理系统。
*   **核心组件：** 哈密顿量、Lindblad 算符。

### 第二层：相空间模型 (Phase Space Model)
**对应类：** `qphase_sde.model.PhaseSpaceModel`

在相空间（例如 Wigner、P-表示、Q-函数）中描述系统动力学。该层级由 Fokker-Planck 方程 (FPE) 的 Kramers-Moyal 展开系数定义。

$$ \frac{\partial P}{\partial t} = \sum_{n=1}^\infty \frac{(-1)^n}{n!} \frac{\partial^n}{\partial \alpha^n} [D_n(\alpha) P] $$

*   **适用场景：** 解析推导、研究相空间分布特性。
*   **核心组件：** 漂移向量 ($D_1$)、扩散张量 ($D_2$)，以及可能的高阶项 ($D_3, \dots$)。

### 第三层：随机模型 (SDE Model)
**对应类：** `qphase_sde.model.DiffusiveSDEModel`、`qphase_sde.model.JumpSDEModel`

描述用于数值仿真的随机轨迹。这是仿真引擎直接使用的层级。

*   **DiffusiveSDEModel (Langevin)：** 适用于仅包含一阶和二阶项的系统（高斯噪声）。
    $$ d\mathbf{y} = \mathbf{a}(\mathbf{y}, t) dt + \mathbf{b}(\mathbf{y}, t) d\mathbf{W} $$
*   **JumpSDEModel：** 适用于将高阶项映射为跳跃过程的系统。

## 建模工作流

1.  **定义物理模型：** 首先定义一个包含漂移 ($D_1$) 和扩散 ($D_2$) 系数的 `PhaseSpaceModel`（第二层）。
2.  **自动转换：** 使用 `qphase_sde.model.fpe_to_sde()` 函数将 FPE 模型自动转换为 `DiffusiveSDEModel`（第三层）。
3.  **执行仿真：** 将生成的第三层模型传递给 `Engine` 进行计算。

此外，高级用户也可以直接定义 `DiffusiveSDEModel`，以便手动控制噪声分解方式。

## 示例：范德波尔振荡器 (Van der Pol Oscillator)

### 定义第二层模型
```python
from qphase_sde.model import PhaseSpaceModel

def drift_fn(y, t, p):
    # ... 计算 D1 ...
    return d1

def diffusion_fn(y, t, p):
    # ... 计算 D2 ...
    return d2

fpe_model = PhaseSpaceModel(
    name="vdp_fpe",
    n_modes=1,
    terms={1: drift_fn, 2: diffusion_fn},
    params={...}
)
```

### 转换为 SDE 模型
```python
from qphase_sde.model import fpe_to_sde

sde_model = fpe_to_sde(fpe_model)
```
