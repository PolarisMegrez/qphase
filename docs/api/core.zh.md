---
description: 核心 API 参考
---

# 核心 API 参考

## Project 与 Workflow

`ProjectContext.discover()` 按显式路径、`QPHASE_PROJECT` 或向上搜索解析当前
`qphase.project/2` manifest，并暴露 Workflow catalog、defaults、项目插件目录和
Session 根目录。

`WorkflowCatalog` 递归列出严格的 `qphase.workflow/2` 文档，并按稳定 Workflow ID
解析。`load_workflow()` 校验文档并返回包含逻辑 `JobConfig` 节点的
`WorkflowSpec`。

## Scheduler

```python
Scheduler(
    system_config: SystemConfig | None = None,
    project: ProjectContext | None = None,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
    on_job_dir: Callable[[Path], None] | None = None,
    cancellation: CancellationController | None = None,
)
```

`Scheduler.run(workflow, dry_run=False, resume_from=None)` 校验逻辑 Job 图，解析
engine 和插件，将 scan grid 交给资源包 engine，持久化 Artifact，并返回
`list[JobResult]`。

每次新 Execution 创建一个 Session，并保存 Workflow 快照和内容哈希。恢复要求
Project ID、Workflow ID 和 Workflow hash 全部一致。

## 配置

- `WorkflowSpec`：版本化 Workflow 身份、标题、Collection、Tag 和 Jobs。
- `JobConfig`：一个 engine、插件配置、结构化输入、输出、ScanSpec 和保存意图。
- `SystemConfig`：只包含机器策略，包括默认保存、scan 存储与 checkpoint、资源
  提示、进度和日志。
- `ProjectManifest`：可迁移的 Project 身份和项目相对路径。

Project 路径不得写入 `SystemConfig`。

## Scan 与 Execution

`ScanSpec` 和 `ScanAxisSpec` 校验显式 `values`、`linspace` 和 `logspace` axes。
`ParameterGrid` 是传给 engine 的编译后表示；scan 始终是一个逻辑 Job。

`ExecutionContext` 携带 grid、资源快照、进度 reporter、取消令牌、ArtifactStore
和 CheckpointStore。engine 应通过 context 报告自然工作单位。

## Registry

`RegistryCenter` 按 namespace/name 发现、校验并创建插件。已安装资源包使用 entry
point，Project 本地插件从 `qphase.toml` 声明的路径发现。

主要方法包括 `register()`、`register_lazy()`、`create()`、`list()` 和
`get_plugin_schema()`。
