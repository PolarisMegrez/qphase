---
description: 数据产品契约
---

# 数据产品契约

!!! info "Phase 0 契约 —— qphase 2.0"
    本页契约已经获批，并冻结供 Phase 1 在 `qphase.data` 与
    `qphase.core.task_profile` 中实现。生产 Result 序列化仍只通过 qphase 2.0 分阶段迁移修改。

QPhase 2.0 用**数据产品**取代无类型的 `trajectory + dict[str, Any]` 结果形态：数据产品
是带类型的容器，其轴、变量、不确定度与 provenance 由机器可读 schema 描述。core 定义
schema 语言与三个公开 data kind；资源包定义 quantity、provenance 与 reducer，且不得
重新定义不兼容的 Dataset 基类。

## Data kind

`DataKind` 恰好有三个取值：

- `time_series`——采样轨迹，轴如 `scan`、`trajectory`、`time`、`channel`。必须具备
  `t0`+`dt` 或显式单调 time 坐标；声明 dtype、实/复值域、channel 定义、独立实现语义
  与频率方向约定。可携带 valid length/mask、warm-up、trajectory ID 与 RNG provenance。
- `spectral`——频域产品，轴如 `scan`、`trajectory`、`frequency`、`channel`。
- `statistics`——低维结果：moments、distributions、Allan variance、first-passage 摘要、
  拟合参数。可包含结构化 table，但任意不可验证对象不得进入持久化 schema。

## 产品 schema

`ProductSchema` 可 JSON 序列化、严格 extra-forbid、具有稳定 hash。shape 在 plan 阶段
可以部分未知（`AxisSchema.size is None`），但在 materialize 前必须**闭合**。

- `AxisSchema`——`name`、`role`、可选 `size`、`coordinate`（`regular` 或 `explicit`）、
  `units`、`monotonic`。`AxisRole` 取值为 `parameter`（被扫描的参数轴——绝不是样本
  集合）、`realization`（仍保留在 payload 中的独立实现）、`coordinate`、`component` 与
  `index`。realization 轴必须被至少一个变量实际保留；parameter/grouping 坐标可以由其
  coordinate payload 或资源包定义的 segmented layout 表示。
- `SamplingBasisSchema`——已经从 payload 中归约掉的 realization 来源，例如形成平均 PSD
  的 trajectories。它具有稳定名称，并通过保留的 realization `source_axis`、固定 `count`
  或整数 `count_variable` 之一闭合，避免把已丢弃的 trajectory 维度伪装成可切片产品轴。
- `VariableSchema`——`name`、`dtype`（禁止 object dtype）、`value_domain`（`real` 或
  `complex`，与 dtype 双向校验）、引用轴的命名 `dims`、`quantity`、`units` 与
  `constraints`（例如 `nonnegative`——仅限实数值变量；或张量 `symmetry`/`layout`
  描述符——Hermitian layout 要求至少两个 component/index 维度）。由**变量**而非
  Dataset class 决定实/复值：频域产品绝不拆成互不相容的实/复 Result 类。
- `UncertaintySchema`——`target`（所描述的变量）、`kind`（`sample_std`、`sem`、
  `confidence_interval`、`covariance`、`other`）、`sampling_basis`（估计所统计的、保留
  或已归约的 realization 来源）、可选的资源包自定义
  `scope` 标识（如 `conditional`/`sampling`）、位于 `(0, 1)` 的 `confidence` 与正整数
  `count`。复值目标必须声明 `real_imag` 或 `magnitude_phase` covariance；实值目标只能
  用 `real`；没有 `custom` 逃生门。covariance 载荷是由 `data_variable` 引用的带类型
  变量——绝不是 metadata dict。
- 矩阵/张量变量使用命名维度加 symmetry/layout 描述符；moment order 是轴或变量属性，
  绝不创建新的 Dataset 类。

## 频域 quantity

冻结的最小 `SpectralQuantity` 集合为 `fourier_amplitude`、`power_spectral_density`、
`cross_spectral_density` 与 `coherence`。频域产品必须携带强制属性集：频率单位、方向、
单/双边、normalization、window 与 estimator（均不得为空字符串），以及可选的有效自由
度。PSD 变量声明 real/nonnegative；cross-spectrum 变量声明 complex 并带 Hermitian
layout。统计字段显式区分 mean、sample std、SEM 与 independent count，不再依赖键名约定。

## Moment family

core schema **没有** moment-family 字段——该领域语义归资源包所有。`qphase_sde` 定义私有、
版本化的 `SDEMomentFamilySchema` 描述符（`moment_kind`、`ordering`、固定的 `order` index
轴与显式正整数 `orders` 列表），嵌入产品的 `attributes`。仅覆盖各阶共享其余维度的矩
（例如 `moment[scan, order, channel]`）；任意混合秩矩张量明确不在首版 schema 的支持声
明之内。

## 运行时句柄与 artifact

进程内传递、session 缓存与持久化是三个分离的层次：

- `DataHandleProtocol`——进程内、可能驻留设备的 buffer，恰好支撑产品的一个**变量**：
  `variable_schema`、`device`、`dtype`、`shape`、`nbytes`、`read_only` 与 `owner`。
  唯一冻结的交换操作是 `materialize(target_device, copy_policy)`；实现绝不进行隐式的
  device-to-host 拷贝。零拷贝导出描述符属于后续设计，刻意不在冻结面内。
- `DataLeaseProtocol`——面向 consumer 的生命周期契约（`handle`、`consumer`、`scope`
  （`execution` 或 `session`）、幂等的 `release()`）。只有 owner 可以关闭或回收
  buffer；consumer 只释放 lease。pin/eviction 策略不属于冻结面。
- `RuntimeProductBacking`——产品的运行时支撑：每个 schema 变量恰好一个 handle，由
  `validate_backing` 校验（变量缺失/多余、完整 variable schema identity、dtype 与闭合轴
  shape 不符都会被拒绝）。
- `ArtifactRef`——持久化、可跨进程引用，只携带身份：artifact id、产品 schema、
  `module:attr` loader 与小写 SHA-256 content hash。无 provenance、无数组、无额外字段或
  cache state。
- `DataMaterializerProtocol`——资源包注册的 runtime handle 与 artifact 产品之间的转换
  协议。

`DataProduct` 是语义层对象，其支撑是 `RuntimeProductBacking` 或 `ArtifactRef`——两者是
不同类型，绝不可混用。`ResultProtocol.save()` 不承担跨 job 数据传递，artifact store 也
不是运行时缓存。handle/lease 的 ownership、只读规则与失败语义由 core 定义；资源包绝不
在 lease 之外把裸设备 buffer 交给另一个 job。

## 产品图与任务 profile

`ProductRequirement`/`ProductDeclaration` 描述带类型的输入与输出；`ProductGraph` 是
engine 从插件声明编译出的、经无环校验的 `ProductNode` 图（节点以 fingerprint 标识）。

`EngineTaskProfile` 使插件需求随任务条件化：

- `PluginRequirementSet`——必需/可选/禁止的插件类；三者两两不相交，按 registry 命名
  规则校验，并以排序存储，fingerprint 不依赖 YAML 顺序；
- `InputProductRequirement`/`OutputProductDeclaration`——带类型的输入选择器与输出声明；
- 可选的 profile resolver 只接收受限的 `TaskProfileResolutionContext`（规范化 job 配置
  与输入产品的 **schema**——绝不是 handle、payload、loader 或 scheduler）；规范化配置必须
  可 JSON 序列化。resolver 返回
  **完整**的 `PluginRequirementSet` 全量替代默认值，返回结果按相同不变量重新校验。

这使得 `analyze` 任务可以要求输入产品与 analyser，而不必伪装拥有 model 或
integrator——同时 model-aware analyser 仍可通过 resolver 把 `model` 显式声明为必需。
