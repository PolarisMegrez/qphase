---
description: QPhase 项目与执行概念
---

# 核心概念

QPhase 在 CLI、GUI、服务 API 与存储格式中统一使用以下术语。

| 术语 | 契约 |
| --- | --- |
| **Project（项目）** | 由 `qphase.toml` 声明的可迁移研究边界，拥有 Workflow、项目默认配置、本地插件目录和 Session 存储。 |
| **Workflow（工作流）** | 带稳定 `id`、元数据和逻辑 Job 图的版本化 YAML 文档。 |
| **Job（任务节点）** | Workflow 中的一个逻辑节点。参数扫描仍然是一个 Job，并产生一个逻辑数据集。 |
| **Execution（执行）** | 一次排队中或运行中的 Workflow 执行尝试。重试会创建新的 Execution。 |
| **Session（会话记录）** | 一次 Execution 的持久化记录，包含 manifest、配置快照、日志、事件和 Artifact。 |
| **Artifact（制品）** | Job 产生的有类型输出，由 artifact manifest 描述。 |
| **Collection（集合）** | 存储在 Workflow 中、可版本控制的分组元数据。目录可以与之对应，但目录不是身份。 |
| **Tag（标签）** | 用于检索和筛选的可迁移多对多 Workflow 元数据。 |
| **Archive（归档视图）** | 收藏、别名、备注、虚拟目录等用户本地组织信息，不属于可复现契约。 |

## 身份与位置

Project 和 Workflow 的 ID 是稳定身份，文件路径只是当前位置。移动 Project
不会改变 `project_id`；在 `configs/workflows/` 内移动 Workflow 也不会改变其
`id`。因此 CLI 与 GUI 使用 Workflow ID 和 Session ID，而不把路径当作身份。

`qphase.toml` 是 Project 路径的唯一来源。`SystemConfig` 只保存机器策略、硬件与
资源提示。这样，同一个可迁移 Project 可以在工作站或服务器上运行，而不会携带原
用户的绝对路径。

## 可复现边界

Workflow 元数据、Collection、Tag、插件默认值和本地插件代码属于 Project，应当
进入版本控制。GUI Archive 是用户本地信息，只能引用 `(project_id, workflow_id)`
或 `(project_id, session_id)`，不得成为执行或复现实验的必要条件。

CLI 始终是完整的自动化接口。GUI 只是同一服务层的可视化客户端，不引入第二套执行
模型。
