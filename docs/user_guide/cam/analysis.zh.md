# 相干振幅矩阵

workspace 内的 `qphase_cam` 资源包求解稳态矩阵方程

\[
\mathcal{L}(R)=-iH(R)R+iRH(R)^\dagger+D(R)=0.
\]

`engine.cam` 需要一个 backend、一个支持 CAM capability 的 model 和一个
`cam_solver`，并可配置任意多个 `cam_postprocessor`。完整示例见
`configs/jobs/vdp_2mode_cam.yaml`、`kerr_2mode_cam.yaml` 和
`kerr_3mode_cam.yaml`。

## 参数扫描

CAM 使用 core 的 `ScanSpec`。插件配置中的模型参数保持标量，命名 scan axis
显式指向这些参数：

```yaml
scan:
  combine: cartesian
  axes:
    omega_a:
      target: model.vdp_2mode.omega_a
      linspace: {start: -0.3, stop: 0.3, num: 101}
    gamma_b:
      target: model.vdp_2mode.gamma_b
      linspace: {start: 0.2, stop: 1.1, num: 101}
```

axis 支持 `values`、`linspace` 与 `logspace`，并可选择 Cartesian 或 zipped
组合。Cartesian 结果维度按 axis 声明顺序排列。上例是 shape 为 `(101, 101)` 的
一个逻辑 job，而不是 10,201 个 scheduler job。

`multistability` 与 `batched_newton` 分别使用自身的 tile 或 batch 策略消费 grid；
`steady_state` 与 `deflation` 使用 CAM pointwise helper。`continuation` 会拒绝外部
`ScanSpec`，因为 continuation 坐标及其自适应步长定义了另一类拓扑。

## 求解器

| 求解器 | 多稳态 | 后端 | Jacobian | 适用场景 |
| --- | --- | --- | --- | --- |
| `steady_state` | 否 | NumPy | root 可选；Cholesky 不使用 | 从单个初值求一个稳态；`auto` 先尝试 root，再尝试保证半正定的 Cholesky。 |
| `multistability` | 是 | NumPy | 可选 | 推荐的自动多稳态搜索。使用多个初值求解、聚类去重，并检查模型容量。 |
| `deflation` | 是 | NumPy | 必需 | 排斥已经找到的根；适合普通多初值搜索总是回到同一根的情况。 |
| `batched_newton` | 提供多个初值时可以 | NumPy 或 CuPy | 必需 | 使用准备好的种子集合进行大规模参数扫描；默认只有单位阵初值，通常最多找到一个根。 |
| `continuation` | 追踪一个 sheet | NumPy | 必需 | 从已知根开始进行伪弧长延拓并经过折叠；不负责自动寻找全部稳态。 |

对于未知的多稳态系统，优先使用 `multistability`。`guess_bounds: auto` 会从
对角平衡方程估计尺度，并生成普通与重尾 Hermitian 初值；这只是自动初值生成，
不能证明所有根都已找到。获得代表性稳态后，可以将其作为 `batched_newton` 的
初值集合以加速大规模扫描，或作为 `continuation` 的起点研究 sheet 与折叠。
当多初值搜索持续收敛到同一个根时，再考虑 `deflation`。

`method: root` 在不受约束的 Hermitian 空间内求解，可能得到非物理数学根。
`method: cholesky` 保证结果半正定，但会遗漏非 PSD 根，并且吸引域可能不同。

对于大型 scan，`tile_workers` 表示请求的进程数，`n_tiles` 控制有界的 scan task
数量。`n_tiles` 应大于 worker 数，以维持负载均衡；VDP 101 x 101 job 使用 24 个
请求 worker 和 288 个 tile，与迁移前 scanner 一致。在 Windows 上，solver 会限制
BLAS 线程，并根据可用物理内存及 `SystemConfig.scan_runtime.resources` 下调实际
进程数。若 spawned pool 因内存压力终止，会自动用更少 worker 重试，而不是立即使
逻辑 job 失败。result metadata 会记录请求/实际 worker 数、tile 数和重试次数。

## 后端支持

CAM engine 目前仅通过 `batched_newton` 支持 CuPy。VDP2、Kerr2 的解析
Jacobian 和 Kerr3 的符号 Jacobian 均支持 backend 数组。其他求解器依赖 SciPy
或 CPU 伪弧长逻辑，会明确拒绝 CuPy。

| 组件 | NumPy | CuPy |
| --- | --- | --- |
| `steady_state` | 支持 | 不支持 |
| `multistability` | 支持，包括多进程 tile | 不支持 |
| `deflation` | 支持 | 不支持 |
| `batched_newton` | 支持 | 支持 |
| `continuation` | 支持 | 不支持 |
| Rayleigh/Hamiltonian/physicality 后处理 | 支持 | 支持，但结果先传回 CPU |
| Jacobian spectrum | 支持 | 支持 |
| Bifurcation refinement | 支持续延拓结果 | 不支持 |

`batched_newton` 在构造 `CAMResult` 时会将收敛状态转换为 NumPy 数组。因此，
即使 Newton 迭代和 Jacobian 线性求解在 GPU 上运行，持久化和大部分后处理仍在
CPU 上完成。

## 物理解判定

`physicality` 对每个有效且收敛的解检查：

1. Hermitian：`R` 与 `R^dagger` 的差异不超过 `hermitian_tolerance`。
2. 半正定：`eigvalsh(R)` 的最小本征值不小于 `-psd_tolerance`。
3. 方程精度：保存的 Liouvillian 残差不超过 `residual_tolerance`。

只有三项均通过时，`is_physical` 才为真。这里要求的是正半定而非严格正定：
`R = alpha alpha^dagger` 可以具有零本征值，小于 `psd_tolerance` 的微小负值视为
数值误差。结果还分别保存 `is_hermitian`、`is_positive_semidefinite`、
`minimum_state_eigenvalue` 和 `residual_within_tolerance`。当前不检查 trace
归一化，也不检查 `D(R)` 是否半正定。

## 结果

结果使用模型规定的固定 solution 容量。`valid_mask` 与 `solution_count` 标记已填充
slot。每个参数点默认按 `real(R[0,0])` 排序；slot 不是跨参数点的全局 branch id，
相邻扫描点之间不隐含连续性。

首选标量频率字段是 `rayleigh_frequency`。完整复 Hamiltonian 谱保存在
`hamiltonian_eigenvalues`，其实部保存在 `mode_frequencies`；不再提供含义模糊的
`omega` 字段。逻辑数组 shape 为
`scan_shape + (capacity, n_modes, n_modes)`。NPZ 保存完整结果，配套 CSV 展平
参数、slot、矩阵元、残差、频率、稳定性和物理解字段。大型结果可保存为有限数量
的 shard；`artifact_manifest.json` 记录布局，`CAMResult.load_dataset` 会恢复相同
的逻辑 shape。
