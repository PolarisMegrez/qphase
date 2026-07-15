# qphase_sde Cayley-Maruyama 与 GPU 执行优化计划

> 状态：实现完成，短时数值与随机 PSD 验收通过；完整生产扫描待运行
>
> 范围：`qphase_sde` 资源包、工作区模型插件 `models/vdp_2mode.py` 及其 CuPy 实现
>
> 非目标：不修改 scheduler、registry、CLI 调度语义，不新增 engine，不改变现有 Euler-Maruyama 结果

## 1. 结论与边界

本次改造仍保持“一个资源包 = 一个 engine”：

- `qphase` 继续只负责发现并实例化 `engine.sde`、backend、model、integrator、analyser。
- `qphase_sde` 继续只暴露一个 `engine.sde`，Cayley-Maruyama 是该 engine 下的新 integrator 插件。
- 通用积分算法、能力协议和执行分支位于 `packages/qphase_sde/qphase_sde/`。
- VDP 模型的矩阵漂移和 CUDA 数值实现位于工作区模型插件 `models/`。
- 不在 `qphase` 核心中加入 VDP、Cayley 或 PSD 专用逻辑。

仓库当前 `vdp_2mode_cupy.yaml` 对应 `200000/0.1 = 2e6` 个积分步；用户描述的生产约束为约 `2e7` 个积分步。性能报告必须同时记录 `steps` 和 `trajectory-steps/s`，不能直接混用两种任务的 wall time。

“只增加一个积分器和一个 CuPy kernel”可解决当前频率系统误差，但不能显著提高 GPU 利用率。原因是当前 engine 仍会对每个时间步分别生成噪声、调用 integrator、启动多个 kernel 并检查保存条件。完整改造分为两个层次：

1. **数值正确性层**：Cayley-Maruyama 单步算法及 VDP fused single-step kernel。
2. **执行优化层**：integrator 可选 chunk-step 能力、VDP fused chunk kernel、精简轨迹记录。

第一层可以独立合并；第二层必须在第一层验证通过后实施。

## 2. 数值方法

### 2.1 更新公式

模型写成 Ito SDE：

```text
dY = A(Y, t) Y dt + B(Y, t) dW
A = -i H
```

Cayley-Maruyama 使用左端点 `A_n=A(Y_n,t_n)` 和 `B_n=B(Y_n,t_n)`：

```text
(I - dt*A_n/2) Y_(n+1)
    = (I + dt*A_n/2) Y_n + B_n*dW_n
```

它保留 Ito 左端点扩散，不进行 Heun 的第二次系数评估。对 `H` 的本征值 `h=x+iy`，确定性放大因子为：

```text
z = (1 - i*h*dt/2) / (1 + i*h*dt/2)
```

当 `y=0` 时严格有 `|z|=1`，因此不会产生 Euler-Maruyama 的虚假径向增益。相位误差为二阶：

```text
omega_discrete = 2*atan(omega*dt/2)/dt
```

当前参数下的无噪声参考结果：

| omega_a | dt | Euler 频差 | Cayley 频差 |
|---:|---:|---:|---:|
| 0.01165914 | 0.1 | +2.464e-3 | -2.64e-6 |
| 0.1 | 0.1 | +5.004e-3 | -2.99e-5 |
| 0.1 | 0.2 | +1.045e-2 | -1.20e-4 |

首个生产版本保持 `dt=0.1`。`dt=0.2` 只作为后续性能实验，不作为初始默认值。

### 2.2 模型能力

不把 `H` 强行加入现有 `SDEModel` 必选协议。新增一个可选的矩阵漂移能力，例如：

```python
@runtime_checkable
class MatrixDriftSDEModel(SDEModel, Protocol):
    def drift_matrix(self, y, t, params):
        """Return A(y,t), shape (..., n_modes, n_modes), with drift=A@y."""
```

理由：

- Euler、Milstein、SRK 继续只依赖 `drift()` 和 `diffusion()`。
- 只有 Cayley-Maruyama 要求矩阵漂移能力。
- 不把物理命名 `H` 固化到通用 SDE 协议中。
- 将来其他线性隐式或指数积分器也可复用 `drift_matrix()`。

通用实现通过 backend batched `solve()` 支持任意小规模模式数。默认 `max_modes=16`，配置允许提高到 64；典型目标为 2 到 10 个模式。超过配置上限时明确报错，不做隐式 fallback 到 Euler。

### 2.3 VDP 矩阵漂移

`models/vdp_2mode.py` 新增 `drift_matrix()`，返回：

```text
A00 = gamma_a/2 + Gamma*(1-|alpha|^2) - i*omega_a
A01 = -i*g
A10 = -i*g
A11 = -gamma_b/2 - i*omega_b
```

保留现有 `drift()` 作为其他积分器的公共接口，并增加测试保证：

```text
drift(y) == drift_matrix(y) @ y
```

## 3. 插件与能力接口

### 3.1 新 integrator 插件

新增：

```text
packages/qphase_sde/qphase_sde/integrator/cayley_maruyama.py
```

建议配置：

```yaml
integrator:
  cayley_maruyama:
    fused: auto       # auto | required | off
    chunk_steps: 128  # 1 表示禁用 chunk path
```

职责：

- `step()` 实现通用 NumPy/backend 数组路径，返回 `dy`，遵守现有 Integrator 协议。
- `fused=auto` 时优先使用模型提供的 fused step；不可用时走通用路径。
- `fused=required` 时能力缺失必须报错，防止生产任务静默降速。
- `fused=off` 用于参考测试和结果对照。
- `supports_adaptive_step()` 返回 `False`。
- 不修改 `engine.sde` manifest；integrator 仍是已有 required plugin namespace。

在 `packages/qphase_sde/pyproject.toml` 注册：

```toml
"integrator.cayley_maruyama" = "qphase_sde.integrator.cayley_maruyama:CayleyMaruyama"
```

### 3.2 模型专用 fused 能力

为避免将 VDP CUDA 公式放入通用 integrator，增加可选能力：

```python
model.supports_fused_step(scheme: str, backend) -> bool
model.fused_step(scheme, y, t, dt, params, noise, backend) -> dy
```

VDP 模型仅对 `scheme == "cayley_maruyama"` 且 backend 为 CuPy 时返回 `True`。具体 CUDA 源码按算法命名空间组织：

```text
models/kernels/base.py
models/kernels/euler_maruyama/vdp_2mode.py
models/kernels/cayley_maruyama/vdp_2mode.py
```

`models/_cupy_vdp_2mode.py` 仅保留 Euler kernel 的兼容导入。模型通过本地 `ModelKernelRegistry` 按 scheme、backend 和 operation 解析实现。

现有 `kernelized_terms()` 保留，Euler-Maruyama 路径不变。

该接口是可选协议，不加入 engine manifest，也不需要 scheduler 感知。兼容性由 integrator 在运行前验证。

### 3.3 chunk-step 能力

单步 fused kernel 仍然会产生每步一次 Python 调度和多次 GPU launch。为解决利用率问题，在 integrator 协议中增加可选能力，不改变现有 `step()`：

```python
class ChunkIntegrator(Protocol):
    def supports_chunk_step(self, model, backend) -> bool: ...

    def step_chunk(
        self,
        y,
        t,
        dt,
        model,
        noise,
        backend,
        *,
        n_steps: int,
        save_offsets: tuple[int, ...],
        record_modes: tuple[int, ...],
    ) -> ChunkStepResult: ...
```

`ChunkStepResult` 至少包含：

```text
final_state
saved_states  # (n_traj, n_saved_in_chunk, n_record_modes)
```

engine 仅在以下条件全部满足时进入 chunk path：

- 固定步长模式；
- integrator 声明支持 chunk；
- model/backend 组合存在实现；
- `chunk_steps > 1`。

否则保持当前逐步路径。该分支完全位于 `qphase_sde.engine`，不修改 scheduler。

## 4. CuPy kernel 设计

### 4.1 single-step kernel

每条轨迹一个线程，完成以下操作：

1. 读取 `alpha, beta` 到寄存器。
2. 计算 `|alpha|^2`、矩阵 `A` 和 `D_alpha/D_beta`。
3. 将 engine 已按 `sqrt(dt)` 缩放的四个实 Wiener 增量组合为两个复噪声：

```text
eta_a = sqrt(D_alpha/2) * (dW_0 + i*dW_2)
eta_b = sqrt(D_beta /2) * (dW_1 + i*dW_3)
Var(dW_k) = dt
```

4. 构造 Cayley 方程右端。
5. 用显式 `2x2` 复数公式求解。
6. 写出 `dy` 或 `y_next`。

必须保持现有扩散语义：

- `D_alpha` 和 `D_beta` 的公式不变；
- 负扩散仍 clip 到 0；
- 四个实 Wiener 增量的方差仍为 `dt`；
- scalar 参数和扫描后的 per-trajectory 参数均受支持；
- `complex64` 与 `complex128` 至少在 single-step path 中受支持。

### 4.2 chunk kernel

CuPy 在每个 chunk 前生成已经按 `sqrt(dt)` 缩放的 Wiener 增量：

```text
noise.shape = (chunk_steps, n_traj, noise_dim)
```

RawKernel 中每个线程：

- 将状态保存在寄存器中；
- 顺序消费该轨迹的 `chunk_steps` 组噪声；
- 在命中 `save_offsets` 时只写 `record_modes`；
- chunk 结束时写一次 final state。

初始 benchmark 候选：

```text
chunk_steps = 64, 128, 256
threads_per_block = 32, 64, 128
```

当前约 1500 条并行轨迹下，不预设 `256 threads/block` 最优。较小 block 能产生更多 block，改善 SM 覆盖。

首版不在 RawKernel 内实现随机数发生器。原因是：

- 复用 CuPy RNG，保持 seed 行为和维护成本可控；
- chunk noise buffer 很小，`128*1500*4*4` 约 3 MiB；
- 先验证 chunk 计算收益，再决定是否引入 Philox/curand 状态。

### 4.3 缓冲区与并发

- 为 noise chunk、final state 和 saved chunk 使用可复用缓冲区。
- 首版使用单 CUDA stream，避免在正确性阶段引入流同步问题。
- chunk path 稳定后再评估双缓冲：生成下一 chunk 噪声时执行当前 chunk。
- CUDA Graph 仅作为后续实验，不是 MVP；多步 kernel 已经消除了主要 launch 开销。

## 5. engine 内存与记录改造

### 5.1 修复轨迹 dtype

当前 `qphase_sde.engine` 使用：

```python
out = be.empty(..., dtype=complex)
```

CuPy 中这会分配 `complex128`，即使状态是 `complex64`。改为：

```python
out = be.empty(..., dtype=y.dtype)
```

这不会损失当前仿真精度，因为状态本身已经是 `complex64`；现有 `complex128` 历史只是保存时升精度，不能恢复已经丢失的信息。

### 5.2 选择记录模式

在 `EngineConfig` 增加：

```yaml
engine:
  sde:
    record_modes: [0]  # null 表示全部模式
```

要求：

- runtime 校验索引范围和非空列表；
- `TrajectorySet.meta["mode_indices"]` 保存物理模式到存储列的映射；
- analyser 按物理 mode 查找存储列；
- batch split 和结果保存必须保留映射；
- `record_modes: null` 完全保持旧行为。

不自动从 analyser 配置推断记录模式，避免 engine 隐式依赖具体 analyser 字段。

### 5.3 保存间隔

频率网格由以下两式控制：

```text
delta_omega = 2*pi/(t1-t0)
omega_Nyquist = pi/(dt*save_stride)
```

因此积分 `dt` 与 PSD 采样间隔必须独立设置。首轮完整任务建议：

```yaml
dt: 0.1
save_stride: 40
record_modes: [0]
```

保存间隔为 4，Nyquist 角频率约 0.785，高于当前目标峰值 0.33 和耦合尺度 0.5。是否采用 40 必须通过一次 `save_stride=20/40` 对照；不能仅凭 Nyquist 公式忽略噪声 alias 对基线和线宽的影响。

### 5.4 预期内存变化

按当前 15 个扫描点、每点 100 条轨迹、`t1=200000`、`dt=0.1` 计算：

| 记录方式 | 约占用 |
|---|---:|
| 2 modes, complex128, save_stride=20 | 4.47 GiB |
| 1 mode, complex64, save_stride=20 | 1.12 GiB |
| 1 mode, complex64, save_stride=40 | 0.56 GiB |

内存降低后再评估将两批 `omega_a` 合并，以增加独立轨迹链数量和 GPU occupancy。合批不是本次 API 改造的前置条件。

## 6. 文件修改范围

### qphase_sde

```text
packages/qphase_sde/pyproject.toml
packages/qphase_sde/qphase_sde/model.py
packages/qphase_sde/qphase_sde/integrator/base.py
packages/qphase_sde/qphase_sde/integrator/__init__.py
packages/qphase_sde/qphase_sde/integrator/cayley_maruyama.py        # new
packages/qphase_sde/qphase_sde/engine.py
packages/qphase_sde/qphase_sde/state.py                            # mode mapping if needed
packages/qphase_sde/qphase_sde/ops.py                              # optional 2x2 helper
```

### 工作区模型插件

```text
models/vdp_2mode.py
models/_cupy_vdp_2mode.py
models/kernels/base.py
models/kernels/cupy_utils.py
models/kernels/euler_maruyama/vdp_2mode.py
models/kernels/cayley_maruyama/vdp_2mode.py
```

### 配置、测试和文档

```text
configs/jobs/vdp_2mode_cupy.yaml
configs/jobs/vdp_2mode_smoke.yaml                                  # or new Cayley smoke job
tests/qphase_sde/test_cayley_maruyama.py                            # new
tests/qphase_sde/test_vdp_kernelized.py
tests/qphase_sde/test_engine.py
tests/qphase/test_save_stride.py
tests/qphase/test_batch_execution.py
docs/api/qphase_sde/integrators.md
docs/api/qphase_sde/integrators.zh.md
docs/api/qphase_sde/engine.md
docs/api/qphase_sde/engine.zh.md
```

除通用 backend 增加 batched `solve()` 外，明确不修改：

```text
packages/qphase/qphase/core/scheduler.py
packages/qphase/qphase/core/registry.py
packages/qphase/qphase/core/protocols.py
packages/qphase/qphase/commands/run.py
```

## 7. 测试计划

### 7.1 数值单元测试

新增 `test_cayley_maruyama.py`：

1. 纯复振子在无阻尼时保持模长。
2. 相位误差对 `dt` 呈二阶收敛。
3. 非 Hermitian 中性模不会出现 Euler 型人工径向增益。
4. `2x2` 通用实现与直接 NumPy 线性求解一致。
5. 左端点扩散的一步样本均值和协方差符合预期。
6. 缺少 `drift_matrix()` 时给出明确错误。
7. 对 1、2、5、10 个模式验证通用路径，并在超过配置上限时给出明确错误。

### 7.2 模型与 GPU 测试

扩展 `test_vdp_kernelized.py`：

1. `drift_matrix() @ y` 与现有 `drift()` 一致。
2. 相同 `y/dW/params` 下 CuPy fused single-step 与 NumPy Cayley 一致。
3. 标量参数和扫描参数数组均一致。
4. `complex64` 和支持时的 `complex128` 路径一致。
5. chunk kernel 与重复 single-step 在固定噪声下结果一致。
6. chunk 内跨越多个 save boundary 时保存索引正确。

### 7.3 engine 回归测试

1. Euler-Maruyama 旧配置行为不变。
2. `chunk_steps=1` 与旧逐步循环保存形状和时间轴一致。
3. `record_modes=null` 保持全部模式。
4. `record_modes=[0]` 保存映射并被 PSD analyser 正确解析。
5. 输出 dtype 与状态 dtype 一致。
6. batch parameter scan 在 chunk path 下正确拆分。
7. seed 在固定 chunk 配置下可重复。

不承诺不同 `chunk_steps` 产生 bitwise 相同的随机序列，除非 CuPy RNG 实验确认该性质。物理验收使用统计一致性和固定噪声 kernel 测试。

### 7.4 物理验收

按成本从低到高执行：

1. `D=0`，用相位线性回归而不是 PSD，短时间验证频率。
2. 测试 `omega_a=0.001, 0.01, 0.1`，覆盖扫描两端和中间。
3. 对 `dt=0.2, 0.1, 0.05` 验证 Cayley 二阶相位误差。
4. 单参数、较小 `n_traj`、缩短 `t1` 的随机 PSD 对照。
5. `save_stride=20/40` 对比 center、linewidth、R2 和 PSD baseline。
6. 通过上述验收后只运行一次完整扫描。

接受标准：

- `dt=0.1`、`D=0` 时，`omega_a=0.1` 的积分频差不高于 `5e-5`。
- CuPy fused 与 NumPy reference 的单步误差符合状态 dtype 精度。
- 随机短任务中 center 不再出现约 `0.005` 的 Euler 系统偏差。
- Euler 现有测试和结果契约无回归。
- 完整任务前，峰中心差异应主要落入 FFT 分辨率、随机统计误差和矩闭合误差范围。

## 8. 性能基准与验收

新增非 CI benchmark 脚本，分别记录：

```text
steps/s
trajectory-steps/s
kernel launches per integration step
GPU peak memory
simulation wall time
PSD wall time
```

对照组：

1. 当前 Euler + kernelized_terms。
2. Cayley generic single-step。
3. Cayley fused single-step。
4. Cayley fused chunk，`K=64/128/256`。

初始性能目标：

- fused single-step 不慢于当前 Euler 生产路径的 1.5 倍；
- fused chunk 至少比 Cayley generic single-step 快 5 倍；
- 历史轨迹显存至少降低 4 倍；
- host 端循环次数约降低 `chunk_steps` 倍；
- 最终以完整任务 wall time 为准，不把 CPU/GPU 利用率百分比本身作为验收指标。

## 9. 分阶段提交

### 阶段 A：数值正确性

- 增加 `MatrixDriftSDEModel` 可选协议。
- 增加 Cayley-Maruyama integrator 和 entry point。
- VDP 增加 `drift_matrix()`。
- 完成 NumPy 数值测试和短时物理测试。

阶段 A 合并后已经可以验证频率误差，但性能不是最终状态。

### 阶段 B：CuPy single-step

- VDP 增加 fused Cayley single-step kernel。
- 完成固定噪声 CPU/GPU 一致性测试。
- 与当前 Euler kernel 路径进行基准比较。

### 阶段 C：chunk execution

- 增加可选 ChunkIntegrator 协议。
- qphase_sde engine 增加固定步长 chunk 分支。
- VDP 增加 chunk kernel。
- 完成 save boundary、progress、seed 和 batch 回归测试。

### 阶段 D：内存与 PSD 输入

- 修复输出 dtype。
- 增加 `record_modes` 和 mode mapping。
- 验证 `save_stride=40`。
- 评估合并两批参数扫描。

### 阶段 E：生产验证与文档

- 更新中英文 integrator/engine 文档。
- 增加 `vdp_2mode_cayley.yaml` 生产候选配置。
- 运行缩短的随机任务。
- 归档 benchmark 与短时拟合结果；完整扫描因运行成本保留给最终生产验收。

## 10. 风险与回退

1. **Ito 语义风险**：扩散必须始终由左端点状态计算。测试一步均值/协方差和稳态矩，不以频率正确代替 SDE 正确性。
2. **复杂噪声归一化风险**：直接复噪声必须严格复现现有 `expand_complex_noise()` 的 `1/sqrt(2)` 约定。
3. **chunk 保存风险**：chunk 可能跨过多个 `save_stride` 边界，保存数量必须与旧公式完全一致。
4. **GPU 精度风险**：状态为 `complex64` 时，CUDA 中系数计算精度需要 benchmark；默认优先结果一致性，再决定是否减少 double 中间量。
5. **线宽分辨率风险**：Cayley 只消除积分偏差，不提高 `2*pi/T` 的频率分辨率。极窄 linewidth 仍需足够总时长。
6. **理论残差**：消除 Euler 偏差后，PSD resonance 与 `H(R0)` 本征值仍可能保留约 `1e-4` 的矩闭合/随机动力学差异。
7. **回退方式**：原 `euler_maruyama` 插件和逐步 engine 分支全部保留；配置改回旧 integrator 即可恢复现有行为。

## 11. 审核决策点

实施前需要确认以下设计选择：

1. Cayley-Maruyama 通用支持任意小规模模式数，默认上限 16、可配置至 64，并为 backend 增加 batched `solve()`。
2. model optional capability 使用 `supports_fused_step/fused_step`，由模型持有专用 CUDA 公式。
3. `chunk_steps` 属于 integrator 配置，`record_modes` 属于 engine 可选配置。
4. 按阶段 A/B 先验证数值，再进入 engine chunk 改造，并在每阶段本地提交。
5. `save_stride=40` 仅作为验证候选；现有生产配置保持不变，另增 `vdp_2mode_cayley.yaml`。
