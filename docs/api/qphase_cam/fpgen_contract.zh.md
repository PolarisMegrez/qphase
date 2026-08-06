# fpgen 集成契约

`qphase_cam` 使用 fpgen 提供符号动力学与精确约化。fpgen 保持为独立的本地项目；
QPhase 不得把其源码复制到 `packages/`，也不得依赖 fpgen 的内部模块。

## 支持范围

当前经过审查的契约为：fpgen `0.5.x`、model schema `2.0`、moment API
`1.0`、reduction API `1.0`。支持的状态布局是
`hermitian-declared-index-v1` 与
`hermitian-normal-anomalous-declared-index-v2`。

QPhase workspace 通过 `[tool.uv.sources]` 中的 editable 本地路径解析
fpgen。`qphase_cam` 声明 `fpgen>=0.5,<0.6`；这两个 workspace-only 包目前均
不发布到 PyPI。

## 模型接口

支持符号分岔分析的 CAM 模型实现 `cam_fpgen_dynamics()`，并返回公开类型
`fpgen.CovarianceDynamics`。返回对象必须满足：

- 状态坐标采用 QPhase 规范 Hermitian 顺序：对角元、上三角实部、上三角虚部；
- 状态索引连续，参数名称与 `model.params` 完全一致；
- 提供 state matrix，并采用受支持的布局；
- 频率参数声明为 `real`，受约束的物理参数显式声明 fpgen domain。

模型文件可以使用 `fpgen.__all__` 导出的名称构造动力学，但不得导入
`fpgen.covariance`、`fpgen.numerical` 等内部模块。

## Adapter 边界

solver、postprocessor 与测试统一使用 `FPGenDynamicsAdapter`，原始
`CovarianceDynamics` 对象为私有实现。Adapter 负责运行时与布局校验、参数顺序、
NumPy 编译函数、精确方向导数、高精度调用、线性约化搜索、符号坐标以及 provenance。

`qphase_cam.core.fpgen` 以外的代码不得访问 `_dynamics`。约化 plan 与
materialized reduction 可以越过 adapter 边界，因为 `qphase_cam.core.reduction`
需要读取其公开的数学字段。

编译后的数值接口遵守以下 shape：若状态维度为 `n`、参数数目为 `p`、状态矩阵为
`(m,m)`，则 RHS 为 `(...,n)`，Jacobian 为 `(...,n,n)`，参数 Jacobian 为
`(...,n,p)`，状态矩阵为 `(...,m,m)`。

约化搜索结果必须包含 candidates、coverage、truncation reasons 与
`manifest()`；候选可通过 `linear_reduce(candidate=...)` 建立 plan，并支持
fraction-free `materialize()`。每个候选提供稳定、可序列化的 `chart_id`
（`ret:<retained_ids>|eq:<retained_equations>`），用于区分同一 retained 变量的
不同 equation partition，并支持跨 chart 去重与追溯；搜索结果同时携带
`rejected_partitions` 明细（逐条记录被排除 partition 的原因与 retained 坐标），
`manifest()` 中含 `rejected_partition_count` 与
`materialization_skipped_oversized`（消元维数大于 3 的物化跳过）计数。
该搜索只覆盖 regular affine-elimination branches；
即使 coverage 为 exhaustive，也不能据此断言 singular branch 不存在。

## 升级流程

fpgen 公共 API 变化时，按以下顺序处理：

1. 先更新 fpgen 包版本及相应 schema/API 版本常量。
2. 针对该 fpgen revision 更新并运行
   `tests/qphase_cam/test_fpgen_contract.py`；无法解释的失败会阻止接入。
3. 运行 `uv run python tools/generate_fpgen_api_snapshot.py`，检查
   `reports/fpgen_api_snapshot.md` 中的 revision、签名和字段变化。
4. 先更新 `FPGenDynamicsAdapter`，再更新本契约文档；solver/postprocessor 仍只能
   使用 adapter。
5. 运行 `uv run pytest tests/qphase_cam` 完成验收。

API 快照只用于审计，不能替代可执行的兼容性测试，也不能为了接受新 revision 而直接
放宽版本检查。
