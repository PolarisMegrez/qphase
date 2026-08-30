---
layout: default
title: SDE API
parent: API 参考
nav_order: 2
---

# SDE API 参考

本节记录了 `qphase_sde` 包，它提供了用于随机微分方程仿真的核心引擎和组件。

更详细、按主题组织的参考请参见专门的 [`qphase_sde` 章节](./qphase_sde/index.zh.md)。

## 引擎

### `class qphase_sde.engine.Engine`

主仿真驱动程序。它协调积分循环、管理数据存储并处理进度报告。

**配置 (`EngineConfig`)：**

*   `dt` (`float`)：时间步长。
*   `t0` (`float`)：观测开始时刻；`[0, t0)` 会正常积分作为预热，但不保存。
*   `t1` (`float`)：积分和观测的结束时刻。
*   `n_traj` (`int`)：轨迹数量。
*   `seed` (`int | None`)：随机种子。
*   `ic` (`Any | None`)：初始条件。
*   `save_stride` (`int`)：每 N 步保存一次。
*   `keep_traj` (`bool | None`)：分析后是否保留原始轨迹。

**方法：**

#### `run(...) -> SDEDataBundle`

执行已配置的 SDE 任务并返回其结果 bundle。引擎要求 `backend`、`model` 和 `integrator` 插件，并接受可选的 `analyser` 插件。

### `class qphase_sde.result.SDEDataBundle`

引擎返回的结果：一组命名 typed data products 加上 job 级 provenance。

*   `products`：命名 dataset——`trajectories` 时间序列 product，以及每个分析器载荷对应的 product（例如 `psd` 对应的 `spectral` product）。
*   `provenance`：job 级 SDE provenance 记录。
*   `axes` / `shape`：命名 scan 轴坐标与网格形状（单点 job 为空）。
*   `point_view(index)`：单个扫描点的惰性视图；`metadata["params"]` 报告该点的扫参取值。
*   `metadata`：job 元数据（模型 `params`、扫描点信息）以及 JSON provenance 记录。

调度器通过 `qphase.data` 把 bundle 持久化为 Artifact v4 目录——`artifact_manifest.json` 加 `npz/3` payload 分块。`qphase.data.load_bundle(job_dir)` 以惰性 products 恢复 `SDEDataBundle`（不涉及 `allow_pickle`）；core 的 `load_result` 在 resume 时走同一 manifest 路径。

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

### PSD 分析器

`qphase_sde.analyser.PsdAnalyzer` 消费 `TrajectorySet` 并写出 PSD 载荷：

*   `axis`：频率轴。
*   `psd`：形状为 `(n_frequency, n_modes)` 的 PSD 矩阵。
*   `modes`：被分析的模式索引。
*   `peaks`：可选的 PSD 分析器内部寻峰结果。

PSD 分析器的寻峰只针对单个 job。跨 job 的 Lorentz 线型拟合通过 SDE 引擎的 `mode: analyze` 配合 `analyser.lorentz_fitter` 插件完成。

## 后处理

跨 job 后处理现在以调度工作流的形式实现：

```yaml
- name: sim
  save: true
  scan:
    axes:
      omega_a:
        target: model.kerr_2mode.omega_a
        values: [0.9, 1.1]
  engine:
    sde: { ... }
  model:
    kerr_2mode:
      omega_a: 0.9
      omega_b: 1.0
      chi: 0.01
      gamma_a: 0.1
      gamma_b: 0.1
      g: 0.1
  analyser:
    psd:
      modes: [0]

- name: fit
  input:
    from: sim
    mode: dataset
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: omega_a
      mode: 0
```

`lorentz_fitter` 分析器读取逻辑 SDE scan dataset，对每个扫描值拟合一条
Lorentz 曲线，并将 `fit_results.csv` 和 `psd_merged.csv` 写入该 job 的 run
目录。Dataset view 与通用导出工具位于 core。
### 自适应带限载频

`analyser.band_limited_carrier` 是 `mode: analyze` 下的 PSD 后处理插件。它在不假设
Lorentz 线型时，自适应估计经过频带筛选的长时 `G^(1)` 载频，并输出带宽敏感度
与相位回归诊断。新版会拒绝不可辨识的单载频点，并将局部平台与跨扫描连续跟踪分开
输出。它不替代随积分执行的短延迟 `coherence_carrier`。

`finite_delay_carrier` 是互补的探测器定义量：它用指数探测器权重积分完整重建
coherence，不选择单一 pole；高探测速率极限为 direct SDE instantaneous carrier。

当实验可观测量定义为 PSD 最大值而非相干相位平均时，可使用不依赖模型的
`spectral_ridge`。它不强加单峰线型，而是报告尺度稳定性、曲率和峰位区间。
