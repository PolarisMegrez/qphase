---
description: qphase 2.x 工件与资源格式
---

# 工件格式

!!! info "qphase 2.0 —— Phase 1 实现"
    本页记录 Phase 1 在 `qphase.data`（工件存储 + NPZ 2.x 适配器）与
    `qphase_sde`（类型化数据包 + 1.x 迁移工具）中落地的磁盘格式，
    是 [数据产品契约](data_products.md) 的格式层伴生文档。

qphase 2.x 工件是一个**目录**：一个 `artifact_manifest.json`，加上每个变量
分块各一个 NPZ 文件。工件目录自描述、可整体搬迁，并且可以用纯 NumPy 检查
——恢复数据绝不需要 `allow_pickle`。

## Manifest v3

`artifact_manifest.json` 是严格校验（`extra="forbid"`）的 JSON 文档：

- `schema_version` —— 字面值 `"qphase.artifact/3"`。
- `artifact_id` —— 稳定的工件标识。
- `created_at` —— 创建时间戳。
- `loader` —— `module:attr` 语法的公开恢复入口；默认为
  `qphase.data.npz:load_product_backing`。加载器绝不使用 `allow_pickle`。
- `products` —— 产品条目列表，每条包含：
    - `name` —— 产品名（工件内唯一）；
    - `product_schema` —— 冻结的[产品模式](data_products.md) JSON；
    - `storage` —— `adapter` 标识与 `variables`：变量名到其**分块记录**
      的映射；
    - `sha256` —— 基于该产品各分块哈希的内容哈希。
- `provenance` —— 可 JSON 序列化的引擎/插件元数据（经过校验）。
- `parents` —— 本工件派生自的父工件 id。
- `content_hash` —— 对产品/父级规范化列表的 SHA-256。

每条分块记录包含 `file`（相对工件目录的路径，因此工件可整体搬迁）、
`key`（恒为 `"data"`）、`shape`、`dtype`、`sha256`（对 C 连续负载字节的
哈希，**每次读取都会校验**），以及 `axis0_range` —— 分片变量沿第 0 维的
`[start, stop)` 区间；当分块持有整个变量时为 `null`。

## NPZ 2.x 布局

参考存储适配器（`qphase.data.npz`）的写入规则：

- 未分片变量写作 `{stem}__{variable}.npz`；
- 分片变量写作 `{stem}__{variable}__{chunk:04d}.npz`，沿第 0 维切分；
- 每个分块只有一个 `"data"` 键，保持**原生 dtype**（包括复数与张量负载）
  —— 绝不使用 object 数组；
- 元数据只存在于 manifest JSON 中，绝不写入 NPZ 文件。

## 写入与读取

- `save_products(directory, products, *, provenance=None, parents=(),
  artifact_id=None, shard_target_bytes=...)` 持久化类型化数据集并返回写出的
  `ArtifactManifestV3`。artifact 支撑的数据集会先被显式完整物化（绝不是隐式
  加载）；设备端负载以显式 `copy_policy="allow"` 拷贝回主机。允许空的产品
  映射（例如只产出标量的运行）。
- `load_products(directory)` 以**惰性支撑**的数据集重新打开工件：不读取
  负载；句柄从 manifest 直接给出 shape/dtype/nbytes；`point_view` 等选择
  操作只读取触及的分块（取单点不会整体拼接）。
- `ArtifactManifestV3.read(directory)` 读取并校验 manifest；
  `manifest.product_ref(name)` 构造单个产品的持久 `ArtifactRef`。
- 进程内注册表把产品级工件 id（`"{artifact_id}:{product}"`）映射到工件
  目录，使 `ArtifactRef` 支撑的数据集能解析存储位置。
  `save_products`/`load_products` 会填充该注册表；跨进程恢复必须先打开
  一次工件目录，然后才能解引用 ref。

## SDE 数据产品

`qphase_sde` 的每个 `engine.run()` 出口都返回类型化的 `SDEDataBundle`。
bundle 暴露 `products`（产品名到数据集的映射）、`provenance` 与
`require(...)`，同时实现 `ResultProtocol` 与 `DatasetResultProtocol`：

- `trajectories` —— `time_series` 产品，轴为
  `(scan, trajectory, time, channel)`，并带 `valid_length` 变量；
  设备端数组留在设备上（CuPy 负载包装为 `BackendArrayHandle`，不拷贝）。
- 分析产品 —— 通过版本化的 `legacy_analysis/1` 桥持久化：数值叶子成为
  变量；嵌套字典按点号路径展平；字符串/JSON 安全叶子进入
  `attributes["payload_meta"]`；扫描下长度不齐的叶子降级为逐点 meta 列表，
  记入 `per_point_meta`；无法桥接的键列入 `dropped_keys`。
- `legacy_result()` 给出单点的 1.x 视图；`point_view` 会把
  `metadata["params"]` 重写为该点的扫描参数（旧 `SDEScanResult` 语义）。

## 迁移 SDE 1.x 结果

`qphase_sde.runtime.migrate` 把既有结果**单向**转换为 v3：

- `migrate_legacy_result(source, output_dir, *, adapter=None,
  shard_target_bytes=None)` —— 转换单个 `sde_result/1` 或
  `trajectory_set/1` 文件。
- `migrate_scan_artifact(manifest_path, output_dir, *, adapter=None)` ——
  转换 `sde_scan/2` 逐点工件，按点流式处理：每个 shard 读取两遍（结构遍、
  分块遍），每个点沿扫描轴为每个变量贡献一个分块，峰值内存不超过一个
  shard 加一个输出分块。

保证：源文件会被 SHA-256 哈希（记入输出 manifest 的 provenance）且**绝不
被修改**；输出目录必须为空且与源不相交；未知的 object 负载在 npy 头部
级别即被拒绝，除非用 `adapter` 将其映射为桥兼容结构。两个函数都返回
`MigrationReport`（含 `MigrationWarning` 条目），对无法识别的输入抛出
`LegacyFormatError`。

## Service 与 GUI 访问

列表查询绝不物化负载：

- `SchedulerService.describe_products(path, *, session_dir)` 返回
  `ArtifactProductCatalog`：工件 id、loader、内容哈希、总大小，以及每个
  产品一条 `ProductSummary`（kind、轴——含 regular 坐标的
  `start`/`step`——变量、设备、`nbytes`、`chunk_count`、`sha256`、
  属性），全部来自 manifest 与模式。
- GUI 以 `GET /sessions/{session_id}/jobs/{job_name}/products` 暴露同一
  目录的 JSON；目录缺失或不是 v3 工件时返回 404。
