---
description: 数据产品契约
---

# 数据产品契约

!!! warning "实验性 —— qphase 2.0"
    本页描述的契约已为 qphase/qphase_sde 2.0 升级冻结，实现位于 `qphase.data` 与
    `qphase.core.task_profile`。在任何生产 Result 序列化改动之前，本页列出的字段决议
    需经人工审核。

QPhase 2.0 用**数据产品**取代无类型的 `trajectory + dict[str, Any]` 结果形态：数据产品
是带类型的容器，其轴、变量、不确定度与 provenance 由机器可读 schema 描述。core 定义
schema 语言与三个公开 data kind；资源包定义 quantity、provenance 与 reducer，且不得
重新定义不兼容的 Dataset 基类。

## Data kind

`DataKind` 恰好有三个取值：

- `time_series`——采样轨迹，轴如 `scan`、`trajectory`、`time`、`channel`。必须具备
  `t0`+`dt` 或显式单调 time 坐标；声明 dtype、实/复值域、channel 定义、独立实现语义
  与频率方向约定。可携带 valid length/mask、warm-up、trajectory ID 与 RNG provenance。
- `spectral`——频域产品，轴如 `scan`、`trajectory`/`statistic`、`frequency`、`channel`。
- `statistics`——低维结果：moments、distributions、Allan variance、first-passage 摘要、
  拟合参数。可包含结构化 table，但任意不可验证对象不得进入持久化 schema。

## 产品 schema

`ProductSchema` 可 JSON 序列化、严格 extra-forbid、具有稳定 hash。shape 在 plan 阶段
可以部分未知（`AxisSchema.size is None`），但在 materialize 前必须**闭合**。

- `AxisSchema`——`name`、可选 `size`、`coordinate`（`regular` 或 `explicit`）、`units`、
  `monotonic`，以及标记实现轴（scan/trajectory）的 `independent` 角色，不确定度合并
  依赖该角色。
- `VariableSchema`——`name`、`dtype`（禁止 object dtype）、`value_domain`（`real` 或
  `complex`）、引用轴的命名 `dims`、`quantity`、`units` 与 `constraints`（例如
  `nonnegative`，或张量 `symmetry`/`layout` 描述符）。由**变量**而非 Dataset class 决定
  实/复值：频域产品绝不拆成互不相容的实/复 Result 类。
- `UncertaintySchema`——`target`（所描述的变量）、`kind`（`sample_std`、`sem`、
  `confidence_interval`、`covariance`、`other`）、`independent_unit`（估计所统计的实现
  轴）、`confidence` 与 `count`。复值变量的不确定度必须显式声明 covariance 表示
  （`real`、`real_imag`、`magnitude_phase`、`custom`）——不允许伪装成一个复数 "std"。
- 矩阵/张量变量使用命名维度加 symmetry/layout 描述符；moment order 是轴或变量属性，
  绝不创建新的 Dataset 类。

## 频域 quantity

冻结的最小 `SpectralQuantity` 集合为 `fourier_amplitude`、`power_spectral_density`、
`cross_spectral_density` 与 `coherence`。频域产品携带共同属性：频率单位与方向、单/双边、
normalization、window、estimator 与有效自由度。PSD 变量声明 real/nonnegative；
cross-spectrum 变量声明 complex 并带 Hermitian layout。统计字段显式区分 mean、
sample std、SEM 与 independent count，不再依赖键名约定。

## Moment family

归组相关 moment 的统计产品声明 `MomentFamilySchema`：`moment_kind`（`raw`、`central`、
`cumulant`、`factorial`）、`ordering`（`c_number`、`normal`、`symmetric`）、
`maximum_order`、张量 symmetry/layout 与 `family_id`。同一 family 的各阶 moment 保存为
一个产品，共享独立样本数与联合 covariance；只有当不同阶数来自不同采样总体、不同
estimator 或无法共享 provenance 时才拆分。

## 运行时句柄与 artifact

进程内传递、session 缓存与持久化是三个分离的层次：

- `DataHandleProtocol`——进程内、可能驻留设备的 buffer，具有 `schema`、`device`、
  `dtype`、`shape`、`nbytes`、`read_only` 与 `owner`；支持 `acquire()`/`release()`、
  `materialize()` 与 `export_interface()`。
- `DataLeaseProtocol`——引用计数生命周期；最后一个 consumer 释放后才允许回收。lease
  声明其 consumer、lifetime scope 与 pin 状态。
- `ArtifactRef`——持久化、可跨进程引用：只含 artifact id、产品 schema、loader、hash
  与 provenance。
- `DataMaterializerProtocol`——资源包注册的 runtime handle 与 artifact 产品之间的转换
  协议。

`DataProduct` 是语义层对象，可由 runtime handle 或 artifact 引用支撑。
`ResultProtocol.save()` 不承担跨 job 数据传递，artifact store 也不是运行时缓存。
handle/lease 的 ownership、只读规则、设备与 stream 同步、copy policy 与失败语义由 core
定义；资源包绝不在 lease 之外把裸设备 buffer 交给另一个 job。

## 产品图与任务 profile

`ProductRequirement`/`ProductDeclaration` 描述带类型的输入与输出；`ProductGraph` 是
engine 从插件声明编译出的、经无环校验的 `ProductNode` 图（节点以 fingerprint 标识）。

`EngineTaskProfile` 使插件需求随任务条件化：

- `PluginRequirementSet`——必需/可选/禁止的插件类；
- `InputProductRequirement`/`OutputProductDeclaration`——带类型的输入选择器与输出声明；
- profile resolver 只能检查 job 配置与输入产品的 **schema**，绝不读取底层大数组。

这使得 `analyze` 任务可以要求输入产品与 analyser，而不必伪装拥有 model 或 integrator。
