---
description: 结果与复现
---

# 结果与复现

每次 Workflow Execution 都会在当前 Project 下创建一个不可变的 **Session**。
Session 保存完整 Workflow、解析后的 Job 配置、事件、日志和类型化 Artifact。

## Session 目录

Session 根目录只由 `qphase.toml` 决定。标准布局如下：

```text
runs/YYYY/MM/<session-id>/
  session_manifest.json
  workflow_snapshot.yaml
  events.jsonl
  qphase.log
  simulate/                       # 逻辑 Job 名称
    config_snapshot.json
    artifact_manifest.json
    00_<product>.npz
  fit/
    config_snapshot.json
    artifact_manifest.json
    fit_results.csv
```

参数扫描始终是一个 Job 和一个逻辑 dataset，不会为每个参数点建立 Session 或
scheduler Job。

## 复现记录

`session_manifest.json` 记录 `project_id`、`workflow_id`、规范化的
`workflow_hash`、Session 状态和各 Job 状态。`workflow_snapshot.yaml` 是 Session
启动时捕获的完整 `qphase.workflow/2` 文档。

每个 Job 的 `config_snapshot.json` 记录合并后的 Project defaults、解析后的插件
配置、环境信息和输入输出关系。它是审计记录，不是独立 Workflow，不能直接传给
`qphase run`。

使用同一个 Project 和内容未变化的 Workflow 恢复中断 Session：

```powershell
qphase run <workflow-id> --resume-from runs/2026/08/<session-id>
```

Project ID、Workflow ID 或 Workflow 内容哈希不匹配时，QPhase 会拒绝恢复。若要
修改实验，应编辑或复制 Workflow 文档并启动新的 Execution。

## Artifact Manifest

每个已保存逻辑结果都有 `artifact_manifest.json`（schema `qphase.artifact/4`），记录 product schema、bundle 描述符、provenance、物理 payload 文件和已注册的存储适配器 id。物理布局由
`system.scan_runtime.storage_layout` 控制：

- `single`：每个 product 一个 payload 文件。
- `sharded`：同一 Job 目录内数量有限的 chunk 文件。
- `per_point`：遗留别名，解析为按字节目标分块。
- `auto`：按配置阈值选择 `single` 或 `sharded`。

内置默认值为 512 MiB 自动阈值和 128 MiB 目标 shard。`qphase.data.load_bundle` 恢复相同的逻辑 bundle，不受物理布局影响。

## SDE Artifact

SDE engine 返回 `SDEDataBundle`，并持久化为 Artifact v4 目录：经过校验的
`artifact_manifest.json` 加上 `npz/3` payload 文件。`trajectories` product 保存
`(scan, trajectory, time, channel)` 轴上的复振幅以及逐轨迹的 `valid_length`，
仅在启用轨迹保留时存在。分析器载荷成为各自的命名 product——PSD 对应
`spectral` product，Allan variance、coherence 与矩统计对应 `statistics`
product，其余分析器经版本化 bridge product 保存。

当 `engine.sde.keep_traj` 为 false 时，分析器 product 仍然保留，原始轨迹在
分析后释放。最新字段定义见 [SDE 输出参考](../api/qphase_sde/output.zh.md)。

## 跨 Job 后处理

后处理应表示为同一 Workflow 中的下游 Job。例如 `analyser.lorentz_fitter` 在
`engine.sde.mode: analyze` 下消费 PSD dataset，并生成 `fit_results.csv` 和
`psd_merged.csv`。这些文件属于下游 Job 的 Artifact，而不是独立 run。
