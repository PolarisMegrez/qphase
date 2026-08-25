---
description: qphase 2.x 的 artifact 与资源格式
---

# Artifact 格式

!!! warning "草案 —— 等待 Phase 1 复审批准"
    本页已按 Phase 1 审计修正（manifest 安全校验、bundle 描述符、带版本的
    存储描述符、事务化 NPZ 写入、typed product/bundle 摘要）重新冻结，目前
    处于复审中的草案状态；在复审批准之前，以 `qphase.data` 与 `qphase_sde`
    下的实现为准。本页是 [数据产品契约](data_products.md) 的格式层
    配套文档。

qphase 2.x 的 artifact 是一个**目录**：一个 `artifact_manifest.json` 加上
由已注册存储 adapter 写入的 payload 文件。参考实现 `npz/2` adapter 为每个
变量 chunk 写一个 NPZ 文件。目录自描述、可整体搬迁、可用纯 NumPy 检查——
恢复永远不需要 `allow_pickle`。

## Manifest v3

`artifact_manifest.json` 是严格校验（`extra="forbid"`）的 JSON 文档：

- `schema_version` —— 字面量 `"qphase.artifact/3"`。
- `artifact_id` —— 稳定的 artifact 标识。
- `created_at` —— 创建时间戳。
- `bundle` —— `BundleDescriptor`：`type_id`（例如
  `generic.dataset_bundle/1`、`sde.bundle/1`）；`adapter_id` 选择用于重建
  具体结果的**已注册** bundle adapter（可信注册表 id，绝不是代码路径）；
  `descriptor_schema`；经 adapter 校验的 JSON `descriptor`（SDE bundle 记
  录 scan 网格——`shape`、`dimension_order`、`axes`、`n_traj_per_point`
  及可选的 `combine`）；以及把语义角色映射到产品名的 `product_roles`。
- `products` —— 产品条目列表，每条包含：
    - `name` —— 产品名（artifact 内唯一）；
    - `product_schema` —— 完整的冻结[产品 schema](data_products.md)
      JSON（`qphase.product/1`）：轴、变量、坐标、采样基、不确定度与属性；
    - `storage` —— `adapter`（已注册 adapter id，如 `npz/2`）、
      `descriptor_schema`、adapter 私有的 `descriptor`，以及通用 `summary`
      （每个变量的 `nbytes`/`chunk_count`），列表页无需打开 adapter 即可
      读取；
    - `sha256` —— 覆盖名称、schema 与 storage 的内容哈希，**每次读取时重
      算**。
- `provenance` —— JSON 可序列化的 engine/插件元数据（经过校验）。
- `parents` —— 本 artifact 派生自的 artifact id（唯一）。
- `content_hash` —— 对规范化 bundle/product/provenance/parent 列表的
  SHA-256，每次读取时重算。

格式中不存在 `loader` 字段，也不存在任何 `module:attr` 引用：存储 adapter
与 bundle adapter 都通过进程内可信注册表解析（`register_adapter`、
`register_bundle_adapter`）。manifest 中的路径按 artifact 相对安全路径校
验；产品名、parents 与 bundle 角色目标必须唯一/一致。违例抛出分类错误：
`ArtifactNotFoundError`（同时是 `FileNotFoundError`）、
`ArtifactUnsupportedError`（未知 schema 版本）、`ArtifactCorruptError`
（解析、跨字段或哈希失败）、`ArtifactAdapterError`（未注册 adapter）与
`ArtifactChecksumError`（payload 校验失败）。

## NPZ 2.x 存储 adapter

参考 adapter（`qphase.data.npz`，adapter id `npz/2`，描述符 schema
`npz.product/2`）在其 descriptor 中为每个变量记录：

- 整个变量的 `full_shape` 与 `dtype`；
- `chunk_axis` —— 变量沿其分片的**具名**维度（未分片变量为 `null`；为满
  足字节目标可选择任意轴，不限于第一维）；
- `chunks` —— 连续、不重叠、完整覆盖的 chunk 记录：`file`（artifact 相
  对路径）、`key`、`logical_range`（沿 `chunk_axis` 的 `[start, stop)`
  区间，chunk 持有整个变量时为 `null`）、`shape`、`dtype` 与 `sha256`
  —— 哈希覆盖 dtype/shape/order/selection 头部与 C 连续 payload 字节，
  **每次读取时校验**，同时核对实际 dtype、shape 与 key 集合。

文件布局：

- 分片变量：`{stem}__{variable}__{chunk:04d}.npz`，每个 chunk 文件一个
  `"data"` key；
- 未分片变量：`{stem}__{variable}.npz`；
- `layout="single"` 把产品的所有变量作为 key 写入一个 `{stem}.npz`（不做
  外部分片）；
- 数组以**原生 dtype** 存储（包括复数与张量 payload）——绝不使用 object
  数组；元数据只存在于 manifest JSON 中。

写入是事务化的：chunk 先写入 `.staging-{token}` 暂存目录，刷盘后被重新读
取并校验（dtype/shape/hash），原子地移动到最终文件名，manifest 最后通过
原子 `os.replace` 发布。已有 manifest 永远不会被覆盖，除非
`replace=True`；替换时只有在新 manifest 发布后才删除旧 payload，因此失败
的写入不会破坏仍可读取的旧 artifact。

## 写入与读取

- `save_products(directory, products, *, provenance=None, parents=(),
  artifact_id=None, shard_target_bytes=..., bundle=None, layout="sharded",
  replace=False)` 持久化 typed dataset 并返回写出的
  `ArtifactManifestV3`。artifact-backed dataset 先被完整物化（显式加载，
  绝不隐式）；设备端 payload 以显式 `copy_policy="allow"` 拷回主机。允许
  空产品映射。未显式给出 `bundle` 时记录通用 bundle 描述符。
- `load_products(directory)` 以**惰性后端**重新打开 artifact：不读取
  payload；句柄从已校验的 manifest 暴露 shape/dtype/nbytes，
  `point_view` 等选择操作只读取触及的 chunk。`load_bundle(directory)` 进
  一步通过已注册的 bundle adapter 重建具体 bundle（通用 adapter 产出
  `GenericDataBundle`）。
- 进程内的 `DirectoryArtifactResolver` 把 artifact id 映射到目录，使
  `ArtifactRef` 后端的 dataset 能解析其存储。`save_products`/
  `load_products` 会填充该注册表；跨进程恢复必须先打开一次 artifact 目
  录才能解引用 ref。`ArtifactRef` 只携带身份——artifact id、产品名、产
  品 schema、存储 adapter id 与内容哈希；它不指名任何代码，也不指名任何
  文件系统位置。

## SDE 数据产品

`qphase_sde` 在 `engine.run()` 的每个出口返回 typed `SDEDataBundle`，并
以带 `sde.bundle/1` bundle 描述符的 v3 artifact 持久化：

- `trajectories` —— `time_series` 产品，轴为
  `(scan, trajectory, time, channel)`，带 `valid_length` 变量；设备端数
  组留在设备上（CuPy payload 以 `BackendArrayHandle` 包装，不发生拷
  贝）。
- typed 分析产品（`graph_ready=True`）—— `psd` 以带完整强制谱属性集与
  基于采样基的不确定度的 `spectral` 产品持久化；Allan 方差、矩族、矩统
  计、相干矩阵与相干载波以带声明轴、量纲与逐变量不确定度的
  `statistics` 产品持久化。
- 其余 analyser 在 Phase 2 之前仍通过带版本的 `legacy_analysis/1` 桥接
  （`graph_ready=False`）持久化：数值叶成为变量，嵌套 dict 按点号路径拍
  平，字符串/JSON 安全叶进入 `attributes["payload_meta"]`，逐点 ragged
  叶退化为记录在 `per_point_meta` 下的 meta 列表，无法桥接的键报告在
  `dropped_keys` 下。
- manifest provenance 记录 `engine`、`sde` 下的 `SDEProvenance` 记录、
  JSON 安全的任务 `meta`（以及 `meta_dropped`），以及 `qphase` 与
  `qphase_sde` 的真实安装 distribution `versions`。
- 导入 `qphase_sde.result` 即注册 `sde/1` bundle adapter，因此干净进程可
  直接凭 v3 manifest 恢复 scan bundle（shape、轴、逐点参数视图）；
  `legacy_result()` 渲染单点 1.x 视图，`point_view` 把
  `metadata["params"]` 重写为该点的 scan 参数。

## 迁移 SDE 1.x 结果

`qphase_sde.runtime.migrate` 把既有结果**单向**转换为 v3：

- `migrate_legacy_result(source, output_dir, *, adapter=None,
  shard_target_bytes=None)` —— 单个 `sde_result/1` 或
  `trajectory_set/1` 文件。
- `migrate_scan_artifact(manifest_path, output_dir, *, adapter=None)` ——
  `sde_scan/2` 逐点 artifact，按点流式转换：每个 shard 读两遍（结构遍、
  chunk 遍），每个点沿 scan 轴为每个变量贡献一个 chunk，峰值内存保持在
  一个 shard 加一个输出 chunk 以内。

保证：源文件计算 SHA-256（记录在输出 manifest 的 provenance 中）且**绝
不被修改**；输出目录必须为空且与源不相交；未知 object payload 在 npy 头
部层级被拒绝，除非 `adapter` 把它们映射为可桥接的 mapping。迁移
provenance 同时记录转换环境的真实 distribution 版本。两个函数都返回
`MigrationReport`（含 `MigrationWarning` 条目），对无法识别的输入抛出
`LegacyFormatError`。

## Service 与 GUI 访问

列表访问永远不物化 payload，也永远不注册 artifact location：

- `SchedulerService.describe_products(path, *, session_dir)` 仅依据
  manifest 加上对 manifest 引用 payload 文件的 `stat` 构建
  `ArtifactProductCatalog`：artifact id、loader（adapter id）、内容哈
  希、总大小（只统计被引用文件——目录中的杂散文件不计入）、一个
  `BundleSummary`（type/adapter id、描述符 schema、产品角色、解包的
  scan shape/combine/axes 与 `n_traj_per_point`），以及每个产品一个
  `ProductSummary`：kind、轴（含 regular 坐标的 `start`/`step`）、带约
  束的变量、坐标、采样基、不确定度、设备、带分类 `missing_reason` 的
  `materializable`（adapter 未注册或 payload 文件缺失）、逻辑 `nbytes`
  与磁盘物理 `physical_nbytes`、`chunk_count`、`sha256`、
  `schema_version`/`schema_fingerprint`、存储 adapter 与描述符 schema、
  属性。
- GUI 以 `GET /sessions/{session_id}/jobs/{job_name}/products` 暴露该目
  录并返回其 JSON；缺失或非 artifact 目录返回 404，不支持或损坏的
  artifact 返回 422。
