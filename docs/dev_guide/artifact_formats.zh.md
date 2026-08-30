---
description: qphase 2.x 的 artifact 与资源格式
---

# Artifact 格式

!!! note "当前 v4 格式"
    本页描述已批准的 v4 artifact 与 NPZ 3.x 契约。实际行为以
    `qphase.data` 与 `qphase_sde` 下的实现为准。本页是[数据产品契约](data_products.md)
    的格式层配套文档。

qphase 2.x 的 artifact 是一个**目录**：一个 `artifact_manifest.json` 加上
由已注册存储 adapter 写入的 payload 文件。参考实现 `npz/3` adapter 为每个
变量 chunk 写一个 NPZ 文件。目录自描述、可整体搬迁、可用纯 NumPy 检查——
恢复永远不需要 `allow_pickle`。

## Manifest v4

`artifact_manifest.json` 是严格校验（`extra="forbid"`）的 JSON 文档：

- `schema_version` —— 字面量 `"qphase.artifact/4"`。
- `artifact_id` —— 稳定的 artifact 标识。
- `created_at` —— 创建时间戳。
- `bundle` —— `BundleDescriptor`：`type_id`（例如
  `generic.dataset_bundle/1`、`sde.bundle/1`）；`adapter_id` 选择用于重建
  具体结果的**已注册** bundle adapter（可信注册表 id，绝不是代码路径）；
  `descriptor_schema`；经 adapter 校验的 JSON `descriptor`（SDE bundle 记
  录 scan 网格——`shape`、`dimension_order`、`axes`、`n_traj_per_point`
  及可选的 `combine`）；以及把稳定语义角色映射到 job-local 产品名的
  `product_roles`。不具备跨 workflow 稳定含义的 label 不写入 roles。
- `products` —— 产品条目列表，每条包含：
    - `name` —— 产品名（artifact 内唯一）；
    - `product_schema` —— 完整的冻结[产品 schema](data_products.md)
      JSON（`qphase.product/1`）：轴、变量、坐标、采样基、不确定度与属性；
    - `storage` —— `adapter`（已注册 adapter id，如 `npz/3`）、
      `descriptor_schema`、adapter 私有的 `descriptor`，以及通用 `summary`
      （每个变量的 `nbytes`/`chunk_count`），列表页无需打开 adapter 即可
      读取；
- `provenance` —— JSON 可序列化的 engine/插件元数据（经过校验）。
- `parents` —— 本 artifact 派生自的 artifact id（唯一）。

格式中不存在 `loader` 字段，也不存在任何 `module:attr` 引用：存储 adapter
与 bundle adapter 都通过进程内可信注册表解析（`register_adapter`、
`register_bundle_adapter`）。manifest 中的路径按 artifact 相对安全路径校
验；产品名、parents 与 bundle 角色目标必须唯一/一致。违例抛出分类错误：
`ArtifactNotFoundError`（同时是 `FileNotFoundError`）、
`ArtifactUnsupportedError`（未知 schema 版本）、`ArtifactCorruptError`
（解析或结构失败）与 `ArtifactAdapterError`（未注册 adapter）。已注册的
storage/bundle adapter
会在 manifest 读取阶段完成 descriptor 的 metadata-only 严格校验。未知 bundle
adapter 仍可作为 generic bundle 列出；已知 adapter 拥有的畸形 descriptor
属于损坏，而不是单纯不可用。manifest metadata 使用严格 JSON，拒绝 `NaN` 与
无穷大；typed 数值 payload 数组仍可保存它们。对于已注册 storage adapter，
manifest 校验还会聚合不同产品的 payload ownership：同一产品的一个文件可包含
多个 key，但同一 payload 文件不得由两个产品共享。

## NPZ 3.x 存储 adapter

参考 adapter（`qphase.data.npz`，adapter id `npz/3`，描述符 schema
`npz.product/3`）在其 descriptor 中为每个变量记录：

- 整个变量的 `full_shape` 与 `dtype`；
- `chunk_axis` —— 变量沿其分片的**具名**维度（未分片变量为 `null`；为满
  足字节目标可选择任意轴，不限于第一维）；
- `chunks` —— 连续、不重叠、完整覆盖的 chunk 记录：`file`（artifact 相
  对路径）、`key`、`logical_range`（沿 `chunk_axis` 的 `[start, stop)`
  区间，chunk 持有整个变量时为 `null`）、`shape`、`dtype`。普通读取只核对
  实际 dtype、shape 与该 payload 文件由 descriptor 声明的精确 key 集合；额外
  key 视为损坏。

文件布局：

- 分片变量：`{stem}__{variable}__{chunk:04d}.npz`，每个 chunk 文件一个
  `"data"` key；
- 未分片变量：`{stem}__{variable}.npz`；
- `layout="single"` 把产品的所有变量作为 key 写入一个 `{stem}.npz`（不做
  外部分片）；
- 数组以**原生 dtype** 存储（包括复数与张量 payload）——绝不使用 object
  数组；元数据只存在于 manifest JSON 中。

写入是事务化的：chunk 先写入 `.staging-{token}` 暂存目录，发布前校验
descriptor，随后原子地移动到最终文件名，manifest 最后通过原子
`os.replace` 发布。普通读写不计算 payload 哈希。已有 manifest 永远不会被覆盖，除非
`replace=True`；替换时只有在新 manifest 发布后才删除旧 payload，因此失败
的写入不会破坏仍可读取的旧 artifact。首次发布时，如果目标 payload 路径已经
存在且不属于经过校验的旧 manifest，也会拒绝覆盖。

## 写入与读取

当前 scheduler 把每个 Job 目录作为有且仅有一个主 bundle artifact 的根目录。
Job 日志、配置快照和导出的 CSV 不属于 artifact payload，除非 manifest 显式引用
它们。未来的多 artifact 布局可以在 Job 下建立独立 artifact 根目录，但必须保持
本页定义的 manifest 契约。

- `save_products(directory, products, *, provenance=None, parents=(),
  artifact_id=None, shard_target_bytes=..., bundle=None, layout="sharded",
  replace=False)` 持久化 typed dataset 并返回写出的
  `ArtifactManifest`。artifact-backed dataset 先被完整物化（显式加载，
  绝不隐式）；设备端 payload 以显式 `copy_policy="allow"` 拷回主机。允许
  空产品映射；所有持久化产品 schema 必须闭合。未显式给出 `bundle` 时记录通用 bundle 描述符。
- `load_products(directory)` 以**惰性后端**重新打开 artifact：不读取
  payload；句柄从已校验的 manifest 暴露 shape/dtype/nbytes，
  `point_view` 等选择操作只读取触及的 chunk。`load_bundle(directory)` 进
  一步通过已注册的 bundle adapter 重建具体 bundle（通用 adapter 产出
  `GenericDataBundle`）。
- 进程内的 `DirectoryArtifactResolver` 把 artifact id 映射到目录，使
  `ArtifactRef` 后端的 dataset 能解析其存储。`save_products`/
  `load_products` 会填充该注册表；跨进程恢复必须先打开一次 artifact 目
  录才能解引用 ref。`ArtifactRef` 只携带身份——artifact id、产品名、产
  品 schema 与存储 adapter id；它不指名任何代码，也不指名任何
  文件系统位置。

## 附件

无法装入 typed product 的辅助文件（逐点 ragged 分析记录、配置快照等）
可以作为**附件**随 artifact 目录保存。它们在 manifest 的自由 provenance
中以 `"attachments"` 键声明，条目为 `{"name", "path", "media_type"}`：

```json
"provenance": {
  "attachments": [
    {"name": "analysis_sidecar", "path": "analysis_sidecar.json",
     "media_type": "application/json"}
  ]
}
```

附件不是 payload：不携带 product schema，列表接口不报告其大小，
artifact 目录中未登记的文件也无法通过 artifact 接口读取。已声明附件通
过 `read_artifact_attachment(directory, name)` 读取：路径按与 payload
相同的 artifact 相对规则校验（禁止逃逸）；`application/json` 附件返回
解析后的对象，其他 media type 返回原始 `bytes`。读取未声明的名称会抛
出 `ArtifactNotFoundError`。

## SDE 数据产品

`qphase_sde` 在 `engine.run()` 的每个出口返回 typed `SDEDataBundle`，并
以带 `sde.bundle/1` bundle 描述符的 v4 artifact 持久化：

- `trajectories` —— `time_series` 产品，轴为
  `(scan, trajectory, time, channel)`，带 `valid_length` 变量；设备端数
  组留在设备上（CuPy payload 以 `BackendArrayHandle` 包装，不发生拷
  贝）。
- typed 分析产品（`graph_ready=True`）—— `psd` 以带完整强制谱属性集与
  基于采样基的不确定度的 `spectral` 产品持久化；Allan 方差、矩族、矩统
  计、相干矩阵与相干载波以带声明轴、量纲与逐变量不确定度的
  `statistics` 产品持久化。
- 每个 graph-ready scan 产品都携带 typed `(scan,)` 扁平扫描参数坐标。所有点
  共用的 frequency、tau、lag、channel 等采样坐标折叠为 dimension coordinate；
  逐点变化的坐标保留为显式 auxiliary coordinate。
- `sde/1` bundle adapter 会把 scan descriptor 与全部产品 schema 交叉校验：
  shape extent 必须是严格正整数；所有带 `scan` 轴的产品必须等于 bundle 的扁平
  scan size；稳定的 `trajectories`/`primary_spectrum` role 必须分别指向兼容的
  time-series/spectral 产品。
- 其余 analyser 在 Phase 2 之前仍通过带版本的 `legacy_analysis/1` 桥接
  （`graph_ready=False`）持久化：数值叶成为变量，嵌套 dict 按点号路径拍
  平，字符串/JSON 安全叶进入 `attributes["payload_meta"]`，逐点 ragged
  叶退化为记录在 `per_point_meta` 下的 meta 列表，无法桥接的键报告在
  `dropped_keys` 下。
- manifest provenance 记录 `engine`、`sde` 下带版本的
  `qphase_sde.provenance/1`、JSON 安全的任务 `meta`（以及
  `meta_dropped`），以及 `qphase` 与 `qphase_sde` 的真实安装 distribution
  `versions`。其中 `dt` 是 SDE 积分步长，不是 core 通用 provenance 字段，
  也不是保存时序的 sample interval；其他资源包定义自己的数值 provenance schema。
- 导入 `qphase_sde.result` 即注册 `sde/1` bundle adapter，因此干净进程可
  直接凭 v4 manifest 恢复 scan bundle（shape、轴、逐点参数视图）；
  `legacy_result()` 渲染单点 1.x 视图，`point_view` 把
  `metadata["params"]` 重写为该点的 scan 参数。
- SDE bundle roles 只记录稳定含义：保留轨迹时的 `trajectories`，以及恰有一个
  频谱产品时的 `primary_spectrum`。其余产品按 kind、quantity 和 fields 选择。
- peak candidates/paths 当前不声明为 graph-ready 公共产品。分组 ragged schema
  与 producer 是 ProductGraph executor 之前必须完成的 Global Phase 5A。在此
  之前，旧 PSD peak metadata 保存为带显式 `source_product` 与
  `payload_field` 路由的 `legacy_peaks/1` bridge；`legacy_result()` 可还原原始
  `analysis["psd"]["peaks"]` 字段，但不会把它声明为 graph-ready peak 产品。

## Service 与 GUI 访问

列表访问永远不物化 payload，也永远不注册 artifact location：
- `SchedulerService.list_artifacts(session_dir)` 对每个 v4 manifest-backed
  目录返回一个条目并使用真实 UUID `artifact_id`。Artifact 目录之外的普通
  文件不具有 artifact 身份，只使用项目相对 `file_ref`。
- GUI 使用 `GET /sessions/{session_id}/artifacts/{artifact_id}` 返回 typed
  产品目录，使用 `GET /sessions/{session_id}/files/{file_ref}` 读取普通文件。

- `SchedulerService.describe_products(path, *, session_dir)` 仅依据
  manifest 加上对 manifest 引用 payload 文件的 `stat` 构建
  `ArtifactProductCatalog`：artifact id、loader（adapter id）、总大小（只
  统计被引用文件——目录中的杂散文件不计入）、一个
  `BundleSummary`（type/adapter id、描述符 schema、产品角色、解包的
  scan shape/combine/axes 与 `n_traj_per_point`），以及每个产品一个
  `ProductSummary`：kind、轴（含 regular 坐标的 `start`/`step`）、带约
  束的变量、坐标、采样基、不确定度、设备、带分类 `missing_reason` 的
  `materializable`（adapter 未注册或 payload 文件缺失）、逻辑 `nbytes`
  与磁盘物理 `physical_nbytes`、`chunk_count`、
  `schema_version`/`schema_fingerprint`、存储 adapter 与描述符 schema、
  属性。
- GUI 以 `GET /sessions/{session_id}/jobs/{job_name}/products` 暴露该目
  录并返回其 JSON；缺失或非 artifact 目录返回 404，不支持或损坏的
  artifact 返回 422。
