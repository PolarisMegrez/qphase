# 相干振幅矩阵

workspace 内的 `qphase_cam` 资源包求解稳态矩阵方程

\[
\mathcal{L}(R)=-iH(R)R+iRH(R)^\dagger+D(R)=0.
\]

`engine.cam` 需要一个 backend、一个支持 CAM capability 的 model 和一个
`cam_solver`，并可配置任意多个 `cam_postprocessor`。完整示例见
`configs/workflows/vdp_2mode/vdp_2mode_cam.yaml`、
`configs/workflows/kerr_2mode/kerr_2mode_cam.yaml` 以及对应模型 Collection 下的
Workflow。

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
`ScanSpec`，因为 continuation 坐标及其自适应步长定义了另一类拓扑。`bifurcation`
则把外层 scan 解释为同一个逻辑 job 内的一组独立搜索 case。axis 可以指向固定模型
参数，也可以指向 perturbation parameter 的基准值，但不能与每个 case 内部自适应
搜索的 `controls` 重叠。

## 求解器

| 求解器 | 多稳态 | 后端 | Jacobian | 适用场景 |
| --- | --- | --- | --- | --- |
| `steady_state` | 否 | NumPy | root 可选；Cholesky 不使用 | 从单个初值求一个稳态；`auto` 先尝试 root，再尝试保证半正定的 Cholesky。 |
| `multistability` | 是 | NumPy | 可选 | 推荐的自动多稳态搜索。使用多个初值求解、聚类去重，并检查模型容量。 |
| `deflation` | 是 | NumPy | 必需 | 排斥已经找到的根；适合普通多初值搜索总是回到同一根的情况。 |
| `batched_newton` | 提供多个初值时可以 | NumPy 或 CuPy | 必需 | 使用准备好的种子集合进行大规模参数扫描；默认只有单位阵初值，通常最多找到一个根。 |
| `continuation` | 追踪一个 sheet | NumPy | 必需 | 从已知根开始进行伪弧长延拓并经过折叠；不负责自动寻找全部稳态。 |
| `bifurcation` | 可变候选数 | NumPy | fpgen 精确动力学 | 联立搜索二至四重不动点，并分析相对于一个指定物理参数的状态 scaling。 |

模型还可以按照 CAM 规范坐标顺序实现 `cam_residual_vector` 和
`cam_jacobian_vector`。这些可选回调用于避免 root 求解热点中的矩阵重建；未提供时，
求解器自动回退到标准 H/D 与 Jacobian capability。

对于未知的多稳态系统，优先使用 `multistability`。`guess_bounds: auto` 会从
对角平衡方程估计尺度，并生成普通与重尾 Hermitian 初值；这只是自动初值生成，
不能证明所有根都已找到。获得代表性稳态后，可以将其作为 `batched_newton` 的
初值集合以加速大规模扫描，或作为 `continuation` 的起点研究 sheet 与折叠。
当多初值搜索持续收敛到同一个根时，再考虑 `deflation`。

`method: root` 在不受约束的 Hermitian 空间内求解，可能得到非物理数学根。
`method: cholesky` 保证结果半正定，但会遗漏非 PSD 根，并且吸引域可能不同。

### 高阶不动点分岔

`cam_solver.bifurcation` 要求 `order-1` 个 controls 和恰好一个
`perturbation.parameter`。controls 用于定位临界参数值；perturbation 指定找到候选后
实际变化的物理参数，并允许同时属于 controls。分类时其他参数固定在候选值。可选的
外层 `ScanSpec` 用于改变各 case 的固定参数，不会展开为 scheduler 子 job。

每个 control 可用 `sampling: linear|log` 指定 discovery seed 的采样方式，默认为
`linear`。`log` 要求下界严格为正，适合搜索跨越多个数量级的阈值，例如弱耦合诱导的
`g^2/Delta` 速率。该选项只改变初值覆盖；refinement 仍在原物理控制变量和配置边界内
求解。

`strategy.auto` 会运行全部可用的一维线性约化，并与 full bordered 搜索结果取并集。
这比只选择首个序参量更昂贵，但不会把序参量选择变成隐含的覆盖假设。当前 discovery
只注册 `seeds`；在真正实现分支追踪前不提供名不副实的 continuation discovery。

默认 `bifurcation_classifier.scaling_signature` 对正规形
`epsilon^k x^m + x^n=0` 输出具名 `(n,k,m)` 和精确指数 `k/(n-m)`。
亚线性响应对应 `k<n-m`；transcritical 的 `(2,1,1)` 指数为 1，因此可以通过
`max_exponent: 0.999` 标记为不满足筛选。响应对象固定为完整 R 或增广状态矩阵，
具体实验读出属于 model-specific 后处理。

metadata 将覆盖度拆分为结构约化覆盖、有限数值搜索覆盖、物理域过滤和奇异约化路径
处理。这些字段用于审计搜索范围，不构成“所有候选均已找到”的证明。模型可以实现
`cam_bifurcation_scales()`，为初值和约化根归一化提供物理尺度；未实现时使用单位尺度。

对于大型 scan，`tile_workers` 表示请求的进程数。未配置 `n_tiles` 或 `tile_size` 时，
solver 默认生成约为 worker 数四倍且不少于 16 个 task，因此普通 job 不需要静态 tile
参数；只有在特定 workload benchmark 后才应覆盖分区策略。在 Windows 上，solver
会限制 BLAS 线程，并根据可用物理内存及 `SystemConfig.scan_runtime.resources` 下调
实际进程数。若 spawned pool 因内存压力终止，会自动用更少 worker 重试，而不是立即使
逻辑 job 失败。result metadata 会记录请求/实际 worker 数、tile 数和重试次数。

multistability scan 使用 continuation-assisted 搜索，而不是把各参数点当作互不相关
的 root。启用 `guess_bounds: auto` 时，它会合并扫描角点与中心的 bounds，执行
full-model 全局 seed 搜索，并把这些状态传给每个参数点。每个空间 tile 从中心附近
开始，并传播已收敛的相邻状态；空点使用 `retry_guesses` 加密搜索；首轮后，解数突变
点使用 `refine_guesses` 和周围状态重新求解。`n_guesses` 只统计每个点新生成的随机
guess，显式、全局和邻点 guess 会额外加入。默认使用解析或符号 Jacobian，并在同一
参数点的全部 root 尝试之间复用其回调。达到模型声明的解容量后，求解器会在连续得到
`capacity_patience` 个成功的重复解后停止（默认 10）；失败的 guess 不增加该计数。

## 后端支持

CAM engine 目前仅通过 `batched_newton` 支持 CuPy。VDP2、Kerr2、cross-Kerr2
和 Kerr3 的解析 Jacobian 均支持 backend 数组。其他求解器依赖 SciPy
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

可选的 `coherence_pole_spectrum` 后处理器用于诊断闭合 CAM 关联

```text
G_W(tau) = Tr[W exp(-i H tau) R]
```

的极点。它支持裸 mode 投影、非相干 trace（`W=I`）和固定相干通道
（`W=l l^dagger`）。每个测量通道选择 residue 不低于该通道最大 residue 的
`relative_residue_floor`、且衰减率最小的极点，并输出复本征值、频率、衰减率、
复 residue、相对 residue、有效 mask 和本征向量条件数。

```yaml
cam_postprocessor:
  coherence_pole_spectrum:
    modes: [0, 1, 2]
    include_trace: true
    channels:
      bright: ["0j", "1+0j", "1+0j"]
      dark: ["0j", "1+0j", "-1+0j"]
    relative_residue_floor: 0.001
```

该插件是独立 CAM 诊断能力，不替代 `rayleigh_frequency`。Rayleigh 商对应闭合
关联在 `tau=0+` 的加权相位斜率；所选 coherence pole 描述长时分量，多极点通道中
二者不必相等。workflow 必须根据声明的测量语义显式选择，资源包不会在二者之间
静默替换。

### Rayleigh 频率与 coherence pole

令 `W` 为固定的半正定测量矩阵。`W=I` 表示非相干 trace 测量，裸 mode 投影选择
单个模式，`W=l l^dagger` 表示相干读出 `c=l^dagger alpha`。对于固定 CAM 状态和
可对角化 Hamiltonian，

\[
H=\sum_j\lambda_j r_j l_j^\dagger,\qquad
l_j^\dagger r_k=\delta_{jk},
\]

闭合关联严格展开为

\[
G_{W,\mathrm{CAM}}^{(1)}(\tau)
=\operatorname{Tr}\!\left[W e^{-iH\tau}R\right]
=\sum_j A_j e^{-i\lambda_j\tau},\qquad
A_j=l_j^\dagger R W r_j.
\]

在 `phase_decreasing` 约定下，其 `tau=0+` 相位斜率为

\[
\omega_W^{R}
=\operatorname{Re}
\frac{\operatorname{Tr}(W H R)}{\operatorname{Tr}(W R)}
=\operatorname{Re}\frac{\sum_j A_j\lambda_j}{\sum_j A_j}.
\]

trace 情形 `W=I` 即 `rayleigh_frequency` 保存的数值。若第零个 pole 是可见度规则
选择的长时 pole，则二者差异严格为

\[
\omega_W^{R}-\operatorname{Re}\lambda_0
=\operatorname{Re}
\frac{\sum_{j\ne0}A_j(\lambda_j-\lambda_0)}{\sum_jA_j}.
\]

当 `abs(A_0) > sum(abs(A_j), j != 0)` 时，还有充分上界

\[
\left|\omega_W^{R}-\operatorname{Re}\lambda_0\right|
\le
\frac{\sum_{j\ne0}|A_j|\,|\lambda_j-\lambda_0|}
{|A_0|-\sum_{j\ne0}|A_j|}.
\]

以上结论只属于闭合 CAM 传播子。对于漂移为 `-i H(R_t) alpha_t` 的平稳 Ito
随机过程，定义

\[
G_{W,\mathrm{SDE}}^{(1)}(\tau)
=\operatorname{Tr}\!\left[W\,\mathbb{E}
\{\alpha(t+\tau)\alpha(t)^\dagger\}\right].
\]

适应的零均值噪声给出严格短时频率

\[
\omega_{W,\mathrm{SDE}}(0^+)
=\operatorname{Re}
\frac{\mathbb{E}\,\operatorname{Tr}[W H(R_t)R_t]}
{\mathbb{E}\,\operatorname{Tr}(W R_t)}.
\]

通常不能把期望值直接替换为 `H(R_CAM) R_CAM`。SDE 长时 pole 属于完整 Markov
生成元，而不属于 `H(R_CAM)`。因此可使用以下诊断分解：

\[
\omega_{W,\mathrm{SDE}}^{\mathrm{long}}-\omega_W^{R}
=\left(\omega_{W,\mathrm{SDE}}^{\mathrm{long}}
-\operatorname{Re}\lambda_0\right)
+\left(\operatorname{Re}\lambda_0-\omega_W^{R}\right).
\]

第二项是闭合 CAM 内部的多极点修正；第一项包含随机自能、矩闭合和有限噪声修正，
必须由随机数据估计。如果随机关联可以写成
`exp(-i omega_R tau) F(tau)` 且 `F` 为正实函数，则全部滞后上频率严格一致。
对称、零均值的 Gaussian 相位调制可以近似满足该条件，但仅有全局相位对称性并不
足以保证它。

CAM/SDE 频率对应必须在明确的适用域内解释。至少应检查平稳性、孤立可见 pole、
足够的衰减率间隙、有限的本征向量条件数，以及对频带和滞后窗口的稳健性。多稳态
切换、弱 residue、近缺陷本征系统、非 Gaussian 频率涨落或观测时间不足均会破坏
单 pole 解释。统计证据应使用不依赖 CAM 目标预先确定的测量与估计器、独立轨迹或
seed、逐轨迹 bootstrap 不确定度和明确的等效性容差；“未能拒绝频率不同”不能作为
严格恒等的证明。

`CAMBifurcationResult` schema 3 包含 candidate table 和通过 `candidate_index`
关联的 branch-response table。后者保存局部分支编号、`(n,k,m)`、指数分子/分母、
扰动侧、幅度和完整状态矩阵领先系数。可通过 `to_candidate_table()`、
`to_branch_table()` 与 `branch_view()` 读取。candidate CSV 包含规范状态坐标和数值
诊断，branch CSV 包含标量分支诊断；完整矩阵系数保存在 NPZ。

高精度验证明确区分重根方程的 `multiplicity_residual_norm` 与高精度重建后完整
CAM 动力学的 `verified_full_residual_norm`，并保存规范状态与搜索未知量的十进制
字符串。可选的 `local_response_validation` 后处理器固定临界 controls，只改变指定的
微扰参数；它沿每条实局部分支求解完整 residual，并记录连续性、物理性、Jacobian
稳定性、完整状态指数和 Rayleigh 频率有效指数。它还会在临界状态及每个收敛分支点
计算最大的 Hamiltonian Petermann 因子。响应表包含
`critical_hamiltonian_petermann_max`、`hamiltonian_petermann_max`、
`hamiltonian_petermann_effective_exponent` 和
`hamiltonian_petermann_fit_exponent`；指数约定为
`K_H ~ |epsilon|^(-q_K)`。与零相容表示未解析出 Hamiltonian Petermann 临界标度，
该量不同于 `stochastic_validity` 输出的 Jacobian 临界模条件数。
`rayleigh_visibility < 1e-3` 标记为
`weak_projection`，表示标量读出在有限窗口可能遮蔽状态的次线性渐近响应。验证失败
只影响响应状态，不删除数学候选。逐点结果另存为 `*_responses.csv`。

外层 bifurcation scan 返回 `CAMBifurcationScanResult`。它保存命名 case axes、展平的
candidate table 和 `candidate_offsets`，因此零候选与多候选 case 均可无歧义表示。
一个逻辑 job 只生成一份 NPZ，以及配套的 `cases`、`candidates` 和可选 `branches`
CSV，不会为每个 case 新建运行目录。

大型 bifurcation campaign 可以隔离普通的单 case 数值失败：

```yaml
engine:
  cam:
    case_failure_policy: record  # 默认 abort，也可设为 record
```

`record` 会生成空 case，并记录 `case_status`、异常类型与消息、展平/多维索引和扫描
参数。它不会吞掉 `MemoryError`、取消、损坏 checkpoint 或进程终止。chunk 持久化与
resume 仍由 core 的 `SystemConfig.scan_runtime.checkpoint` 管理；CAM engine 不发现或
覆盖 system config。
