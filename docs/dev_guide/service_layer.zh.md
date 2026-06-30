---
description: 服务层
---

# 服务层

服务层是 QPhase 面向 Python 客户端的应用 API。它位于用户界面和 core primitives 之间，让 CLI、GUI、notebook 以及未来的本地 API 复用同一套编排逻辑。

```text
client UI -> qphase.service -> qphase.core -> resource packages -> plugins
```

## 设计规则

CLI 与 GUI 是平级客户端。CLI 负责命令行参数解析和终端输出；GUI 负责可视化交互、配置编辑、执行计划、进度和结果浏览。二者都不应包装对方。

服务层返回结构化对象，而不是打印 Rich 表格或 Typer 消息。这样同一个 API 可以被测试、GUI 和自动化脚本复用。

服务层应优先包裹现有 core 行为，再引入新行为。如果规则已经存在于 `qphase.core.scheduler`、`qphase.core.config_loader` 或 `qphase.core.registry`，service 应委托它，或抽出共享 helper，而不是复制规则。

## 当前 Facade

`ConfigService` 负责加载 system/global/job 配置、规范化 raw job 字典、预览合并后的配置，并通过 registry 校验 job 中的 plugin 配置块。

`RegistryService` 负责发现插件、返回 catalog、暴露插件 JSON schema、校验插件配置、报告可扫描参数，并读取 engine manifest。

`SchedulerService` 负责列出和加载 job、构建结构化执行计划、通过 core scheduler 执行 job、提供 dry-run plan、读取 session manifest，并汇总 artifacts。

## 结构化模型

服务层返回模型位于 `qphase.service.models`。初始模型包括 plugin catalog、config validation issue、merged config preview、execution plan、execution plan job/edge、artifact summary 和 run handle。

这些模型有意保持可序列化。GUI 或本地 HTTP API 应能把它们转成 JSON，而不依赖 scheduler 内部对象或 plugin 实例。

## 执行计划

`SchedulerService.build_plan()` 是 dry-run、GUI preview 和未来机器可读 CLI 输出共享的计划接口。它校验 job、展开参数扫描、报告依赖边，并返回 validation issues，同时不创建真实 run session。

plan 应尽量保持无副作用。创建 run 目录、写 manifest、实例化 engine 属于执行阶段，而不是预览阶段。

## 配置所有权

System config 拥有运行时路径和 core 行为。Global config 拥有用户或机器默认值，例如 backend 偏好。Job config 拥有一次 workflow 的意图。Plugin schema 拥有 plugin-specific 校验契约。Engine manifest 拥有所需和可选 plugin namespace。

Global defaults 可以补全运行必需选择，但 optional workflow step 不应只因为 global 默认存在就自动运行。某个 namespace 是否参与 job，应由 engine manifest 和显式 job 配置决定。

## 实现建议

在多个客户端共享用例真正出现之前，service 方法应保持薄封装。优先返回小 DTO，而不是暴露 scheduler 私有状态。如果 service 多处需要 core 私有方法，应先抽出带测试的 core helper，再扩展 service API。
