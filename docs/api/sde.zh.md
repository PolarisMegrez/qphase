---
layout: default
title: SDE API
parent: API 参考
nav_order: 2
---

# SDE API 参考

本节记录了 `qphase_sde` 包，它提供了用于随机微分方程仿真的核心引擎和组件。

## 引擎

### `class qphase_sde.engine.Engine`

主仿真驱动程序。它协调积分循环、管理数据存储并处理进度报告。

**配置 (`EngineConfig`)：**

*   `dt` (`float`)：时间步长。
*   `t_max` (`float`)：仿真时长。
*   `n_traj` (`int`)：轨迹数量。
*   `integrator` (`dict`)：积分器设置。
*   `backend` (`str`)：后端名称。

**方法：**

#### `run(model: SDEModel, ...) -> SDEResult`

为给定模型执行仿真。

---

## 积分器

### `protocol qphase_sde.integrator.Integrator`

所有数值求解器必须实现的接口。

**方法：**

*   `step(y, t, dt, model, noise, backend) -> dy`：执行单个固定时间步。
*   `step_adaptive(y, t, dt, tol, model, noise, backend, rng) -> (y_next, t_next, dt_next, error)`：（可选）执行自适应时间步。

### `class qphase_sde.integrator.GenericSRK`

支持多种方法和自适应步进的通用随机龙格-库塔求解器。

**参数：**

*   `method` (`str`)：要使用的积分方案（`"euler"`、`"heun"`）。
*   `tol` (`float`, 可选)：自适应步进的误差容差。

---

## 模型

`qphase_sde` 包支持分层建模方法。

### 第一层：主方程

#### `class qphase_sde.model.MasterEquation`

在希尔伯特空间中表示系统动力学。

**属性：**
*   `hamiltonian`：哈密顿算符。
*   `lindblad_ops`：Lindblad 塌缩算符列表。

### 第二层：相空间 (FPE)

#### `class qphase_sde.model.PhaseSpaceModel`

通过 Kramers-Moyal 系数在相空间中表示系统动力学。

**属性：**
*   `terms` (`dict[int, Any]`)：将阶数 $n$ 映射到系数 $D_n(\alpha)$ 的字典。
    *   $n=1$：漂移向量。
    *   $n=2$：扩散张量。

### 第三层：随机 (SDE)

#### `protocol qphase_sde.model.SDEModel`

定义引擎消耗的物理系统的接口。

**属性：**

*   `n_modes` (`int`)：状态向量的维度。
*   `noise_dim` (`int`)：噪声向量的维度。
*   `noise_basis` (`str`)：`"real"` 或 `"complex"`。

**方法：**

*   `drift(y, t, params) -> Any`：计算漂移向量 $\mathbf{a}(\mathbf{y}, t)$。
*   `diffusion(y, t, params) -> Any`：计算扩散矩阵 $\mathbf{b}(\mathbf{y}, t)$。

#### `class qphase_sde.model.DiffusiveSDEModel`

朗之万型 SDE（连续高斯噪声）的具体实现。

#### `class qphase_sde.model.JumpSDEModel`

跳跃-扩散 SDE 的具体实现。

### 转换器

#### `qphase_sde.model.fpe_to_sde(fpe: PhaseSpaceModel) -> DiffusiveSDEModel`

将二阶 PhaseSpaceModel 转换为 DiffusiveSDEModel。
*   漂移 $A = D_1$
*   扩散 $B = \sqrt{D_2}$

---

## 噪声规范

定义驱动系统的噪声属性。

**属性：**

*   `kind` (`str`)：`"independent"` 或 `"correlated"`。
*   `dim` (`int`)：噪声通道数。
*   `covariance` (`Any`, 可选)：相关噪声的协方差矩阵。

---

## 分析器

### `protocol qphase_sde.analyser.AnalyzerProtocol`

分析插件的接口。

**方法：**

*   `analyze(data: Any, backend: BackendBase) -> ResultProtocol`：对仿真数据执行分析。
