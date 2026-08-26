---
description: QPhase 发布版本与持久化数据兼容政策
---

# 版本与迁移政策

QPhase 发布版本使用 `x.y.z`，同时区分但协调管理 Python/API 兼容性与持久化科学数据兼容性。

## Major `x`

major 版本可以破坏 API、Workflow schema、插件契约和持久化格式。新 major 不承担读取或迁移旧 major
artifact 的义务；旧 major 的复现依靠其归档 release、环境或 branch，新 runtime 不永久保留兼容代码。

当前1.x到2.x迁移工具只用于工作区过渡，不属于稳定2.x API。Global Phase 4 完成项目迁移与验证后，删除其
runtime 入口、专用 fixtures 与迁移文档。

## Minor `y`

同一 major 内，公开 Python API 保持向后兼容。持久化数据可以单向演进：新 minor 必须直接读取旧 minor，
或提供经过验证的单向 migrator；旧 minor 无需读取新 minor 输出。

minor migrator 保留到下一个 major 发布，可以直接覆盖所有受支持旧 minor，也可以形成经过验证的连续迁移链。
迁移必须记录源 hash、工具/包版本和 warning，且不得原地修改源 Artifact。

## Patch `z`

patch 在公共契约层双向兼容：不得增加必填 schema 字段、改变字段含义、修改 hash 算法，或拒绝同一 minor
其他 patch 中原本合法的数据。安全修复可以拒绝依据冻结契约本来就非法的数据；真正收紧契约必须提升 minor。

## Schema 版本

包版本与 schema 版本不是同一个编号。持久化契约使用独立 ID，例如 `qphase.product/1`、
`qphase.artifact/3`、`npz/2` 与 `qphase_sde.provenance/1`。schema 修改属于 patch、minor 还是 major，取决于
兼容性影响，而不是与包版本数字对应。

迁移工具只能通过显式命令或维护工具运行。Artifact loader 不静默重写数据，普通插件发现也不导入旧 major
兼容层。
