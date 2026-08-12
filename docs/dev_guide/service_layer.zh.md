---
description: 服务层
---

# 服务层

服务层是 CLI、本地 GUI、notebook 和自动化共同使用的结构化 Python API：

```text
client -> qphase.service -> qphase.core -> resource engine -> plugins
```

client 负责展示，service 负责应用用例，core 负责执行契约。任何 client 都不应调用
另一个 client，也不应复制 scheduler 规则。

## 服务

- `ConfigService`：读取 Project defaults、预览合并后的 Job 配置、校验插件，并
  独立访问 SystemConfig。
- `RegistryService`：发现插件并暴露 schema 与 manifest。
- `SchedulerService`：加载 Workflow、建立低副作用 Execution plan、调用 core
  Scheduler，并检查 Session Artifact。
- `ExecutionManager`：异步排队 Execution、输出事件、取消、在 Job 边界暂停，以及
  修改尚未开始的 Job。
- `ProjectService`：列出 Session 历史、识别中断 Session、管理别名与回收站，并
  通过 revision check 编辑 Workflow 文档。

`qphase.service.models` 中的返回值是可序列化 DTO，并统一使用 Project、Workflow、
Job、Execution、Session 和 Artifact 术语。

## 规划边界

`SchedulerService.build_plan()` 校验 Job、插件需求、scan、依赖和预期 Artifact，
但不创建 Session，也不导入只在运行时需要的 engine 状态。参数点只做摘要，不会
枚举为 Job。

## 配置归属

- `qphase.toml`：Project 身份和可迁移的项目相对路径。
- Project defaults：可复现的插件默认配置。
- Workflow：科学意图和逻辑 Job 图。
- 插件 schema：插件专属校验和默认值。
- Engine manifest：必需与可选插件 namespace。
- SystemConfig：用户/机器运行策略，不包含 Project 路径。

service 方法应保持精简。CLI 与 GUI 需要同一规则时，应在 core 或 service 中实现
一次，并向两个 client 返回结构化数据。
