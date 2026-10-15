# qpsgen 集成契约

`qphase_cam` 使用 qpsgen 提供符号动力学与精确约化。qpsgen 保持为独立发布的项目；
QPhase 不得把其源码复制到 `packages/`，也不得依赖 qpsgen 的内部模块。

## 支持范围

当前经过审查的契约为：qpsgen `0.5.x`、model schema `2.0`、moment API
`1.0`、reduction API `1.1`。支持的状态布局是
`hermitian-declared-index-v1` 与
`hermitian-normal-anomalous-declared-index-v2`。

提交的 workspace 与发布后的 QPhase 包均从包索引解析其声明的
`qpsgen>=0.5,<0.6` 依赖。同时开发两个仓库时，可以临时以 editable 模式安装
qpsgen，但不得提交本地路径覆盖。

## 模型接口

支持符号分岔分析的 CAM 模型实现 `cam_qpsgen_dynamics()`，并返回公开类型
`qpsgen.CovarianceDynamics`。返回对象必须满足：

- 状态坐标采用 QPhase 规范 Hermitian 顺序：对角元、上三角实部、上三角虚部；
- 状态索引连续，参数名称与 `model.params` 完全一致；
- 提供 state matrix，并采用受支持的布局；
- 频率参数声明为 `real`，受约束的物理参数显式声明 qpsgen domain。

模型文件可以使用 `qpsgen.__all__` 导出的名称构造动力学，但不得导入
`qpsgen.covariance`、`qpsgen.numerical` 等内部模块。

## Adapter 边界

solver、postprocessor 与测试统一使用 `QPSGenDynamicsAdapter`，原始
`CovarianceDynamics` 对象为私有实现。Adapter 负责运行时与布局校验、参数顺序、
NumPy 编译函数、精确方向导数、高精度调用、线性约化搜索、符号坐标以及 provenance。

`qphase_cam.core.qpsgen` 以外的代码不得访问 `_dynamics`。约化 plan 与
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
物化错误属于诊断信息而非 partition 拒绝：`materialization_failures` 按
`chart_id` 逐条记录，`manifest()` 报告 `materialization_failure_count`，
且不计入 `rejected_reason_counts`。
该搜索只覆盖 regular affine-elimination branches；
即使 coverage 为 exhaustive，也不能据此断言 singular branch 不存在。

## 升级流程

qpsgen 公共 API 变化时，按以下顺序处理：

1. 先更新 qpsgen 包版本及相应 schema/API 版本常量。
2. 针对该 qpsgen revision 更新并运行
   `tests/qphase_cam/test_qpsgen_contract.py`；无法解释的失败会阻止接入。
3. 运行 `uv run python tools/generate_qpsgen_api_snapshot.py`，检查
   `reports/qpsgen_api_snapshot.md` 中的 revision、签名和字段变化。
4. 先更新 `QPSGenDynamicsAdapter`，再更新本契约文档；solver/postprocessor 仍只能
   使用 adapter。
5. 运行 `uv run pytest tests/qphase_cam` 完成验收。

API 快照只用于审计，不能替代可执行的兼容性测试，也不能为了接受新 revision 而直接
放宽版本检查。
