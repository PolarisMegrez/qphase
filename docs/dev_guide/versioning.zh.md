---
description: QPhase 发布版本与持久化数据兼容政策
---

# 版本与迁移政策

QPhase 发布版本使用 `x.y.z`，同时区分但协调管理 Python/API 兼容性与持久化科学数据兼容性。

## Major `x`

major 版本可以破坏 API、Workflow schema、插件契约和持久化格式。新 major 不承担读取或迁移旧 major
artifact 的义务；旧 major 的复现依靠其归档 release、环境或 branch，新 runtime 不永久保留兼容代码。

1.x 到 2.x 的迁移工具只用于工作区过渡，不属于稳定 2.x API；项目已在 2.0.0 边界完成迁移与验证，其
runtime 入口、专用 fixtures 与迁移文档已删除。

## Minor `y`

同一 major 内，公开 Python API 保持向后兼容。持久化数据可以单向演进：新 minor 必须直接读取旧 minor，
或提供经过验证的单向 migrator；旧 minor 无需读取新 minor 输出。

minor migrator 保留到下一个 major 发布，可以直接覆盖所有受支持旧 minor，也可以形成经过验证的连续迁移链。
迁移必须记录源 hash、工具/包版本和 warning，且不得原地修改源 Artifact。

## Patch `z`

patch 在公共契约层双向兼容：不得增加必填 schema 字段、改变字段含义、修改 hash 算法，或拒绝同一 minor
其他 patch 中原本合法的数据。安全修复可以拒绝依据冻结契约本来就非法的数据；真正收紧契约必须提升 minor。

## Schema 版本

包版本与 schema 版本不是同一个编号。持久化契约使用独立 ID，例如 `qphase.product/1`、
`qphase.artifact/4`、`npz/3` 与 `qphase_sde.provenance/1`。schema 修改属于 patch、minor 还是 major，取决于
兼容性影响，而不是与包版本数字对应。

迁移工具只能通过显式命令或维护工具运行。Artifact loader 不静默重写数据，普通插件发现也不导入旧 major
兼容层。

## Wave 6D 破坏性变更（qphase_sde 2.0）

Wave 6D 迁移重构了 `qphase_sde` 的 PSD/Allan 测量配置、typed 输入 requirement 名称与
product provenance 属性。所有配置模型均为 `extra="forbid"`，被删除的键会直接触发校验错误，
且不保留双键兼容。

1. **PSD/Allan `measurement` union。** `PsdAnalyzerConfig` 与 `AllanVarianceConfig`
   删除了顶层 `kind`/`modes` 键，改为严格判别的 `measurement` union。PSD 取
   `measurement: {complex_field: {channels, representation, orientation?}}` 或
   `measurement: {sampled_scalar: {channels, ...}}`；Allan 取
   `measurement: {phase_frequency: {channels, orientation, amplitude_floor, min_independent_windows}}`
   或 `measurement: {sampled_scalar: {channels, min_nonoverlap_pairs}}`。相位路径的全部旋钮
   （`orientation`、`amplitude_floor`、`min_independent_windows`）移入 `phase_frequency`
   变体；`sampled_scalar` 变体为 `extra="forbid"`，不接受 `orientation`。

   ```yaml
   # 迁移前
   analyser:
     psd: {modes: [0], kind: complex, orientation: phase_decreasing}
     allan_variance: {modes: [0], min_independent_windows: 4}

   # 迁移后
   analyser:
     psd:
       measurement:
         complex_field:
           channels: [0]
           representation: complex
           orientation: phase_decreasing
     allan_variance:
       measurement:
         phase_frequency:
           channels: [0]
           min_independent_windows: 4
   input_bindings:
     psd: {signal: input.trajectories}
     allan_variance: {signal: input.trajectories}
   ```

2. **requirement 由 `trajectories` 更名为 `signal`。** 两个分析器的 typed 输入
   requirement 现名为 `signal`，workflow 必须显式绑定：
   `input_bindings: {<analyser_label>: {signal: input.trajectories}}`。
3. **`source_analyser` 更名为 `producer_analyser`。** `assemble_typed_product`
   写出的 typed product 属性（及 provenance 条目）全局使用新键；读取
   `attributes["source_analyser"]` 的消费方必须改用 `producer_analyser`。
4. **typed product 携带逐变量 units。** `assemble_typed_product` 新增 `variable_units`
   参数，spectral analyser 在 product 边界为所有数值叶标注 units。sampled-scalar PSD
   的 `psd`、`psd_std`、`psd_sem` 均带有 `<value-units>^2/inverse_time` 形式 units
   （源 observable 无量纲时为 `1/inverse_time`）；sampled-scalar Allan 的方差、SEM
   与逐轨迹叶带有 `<value-units>^2`（无量纲为 `1`）。phase 路径的方差叶声明
   `inverse_time^2`，mean-frequency 叶声明 `inverse_time`。源 units 为 `"1"` 或 `""`
   时统一归一为上述无量纲形式。PSD product 以顶层属性暴露 `convention`，复场 product
   另暴露 `representation`。
5. **`kind: modular` 已移除。** 复场模长的 PSD 改写为 `measurement.complex_field`
   内的 `representation: magnitude`。
