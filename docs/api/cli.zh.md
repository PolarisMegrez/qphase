---
description: QPhase 2 命令行接口
---

# CLI 参考

CLI 是 QPhase 最优先且完整的自动化接口。需要 Project 的命令可使用全局选项：

```bash
qphase --project <project-root-or-qphase.toml> <command>
```

未指定时，QPhase 依次检查 `QPHASE_PROJECT` 并向上查找 `qphase.toml`。

## Project

```bash
qphase project init [PATH] [--name TEXT] [--force]
qphase project show
qphase project tag [--add TAG] [--remove TAG] [--private]
qphase project alias [TEXT] [--clear]
qphase project note [TEXT] [--clear]
qphase project reindex
qphase project migrate --dry-run
```

`init` 写入 `qphase.toml`，建立 Workflow、插件和 Session 目录，并生成 Project
插件默认值。`show` 显示已解析的 Project ID 与各路径。`tag`/`alias`/`note`
为 Project 本身添加注解。`reindex` 重建对象 catalog read model 并列出
location issue。`migrate --dry-run` 零写入地预览 Phase 4 历史迁移。

## Workflow 目录

```bash
qphase workflow list [--collection NAME] [--tag TAG] [--query TEXT] [--json]
qphase workflow show WORKFLOW_ID
qphase workflow path WORKFLOW_ID
qphase workflow tag WORKFLOW_ID@REVISION [--add TAG] [--remove TAG] [--private]
```

目录会递归扫描 Project Workflow 根目录，并可按 Collection、Tag 或 ID/标题/路径
文本筛选。正常调用使用稳定 Workflow ID，而不是文件名；重复 ID 会被拒绝。
`workflow tag` 为 workflow 的某一个内容版本添加注解。

## Catalog 与 Tags

Session、Artifact、Occurrence、Job 和 Execution 通过对象 catalog 列出与注解。
所有 `list` 命令接受同一组过滤器：`--tag`、`--tag-any`、`--tag-without`、
`--tag-descendant`、`--tag-namespace`、`--facet k=v`、`--range k=low..high`、
`--lifecycle`、`--retention`、`--direct`、`--limit` 与 `--offset`。此外还有按
对象种类校验的派生 facet 过滤：`--plugin`（job）、`--quantity`（artifact）、
`--model`/`--engine`/`--has-model`（session）。

```bash
qphase session list|tag|lifecycle|retention ...
qphase artifact list|tag|lifecycle|retention ...
qphase occurrence list [--session ID] [--artifact ID]
qphase occurrence tag SESSION_ID ARTIFACT_ID [--add TAG] [--job NAME] [--private]
qphase occurrence retention SESSION_ID ARTIFACT_ID [VALUE] [--clear] [--job NAME]
qphase job list ...
qphase job tag WORKFLOW_ID@REVISION:JOB_NAME [--add TAG] [--remove TAG] [--private]
qphase execution tag EXECUTION_ID [--add TAG] [--remove TAG] [--private]
```

tag 命令默认写入共享 annotation 文档；`--private` 把改动留在用户私有库。
`execution tag` 编辑冻结的 submission tag，仅当 execution 仍在排队时允许。

```bash
qphase tag policy show
qphase tag policy validate
qphase tag promote KIND OBJECT_ID TAG
qphase view save NAME --kind KIND [--tag TAG] [--lifecycle X] [--retention Y]
qphase view list
qphase view delete NAME
```

`tag promote` 把私有 tag 提升到共享 annotation 层。Saved view 是用户私有的
命名 catalog 过滤器。对象模型、tag policy、继承与 shadow 规则见
[Tags 与 Catalog](../user_guide/tags.md) 指南。

## 执行

```bash
qphase run WORKFLOW [OPTIONS]
```

`WORKFLOW` 是稳定 ID 或相对于 Workflow 根目录的 YAML 路径。

| 选项 | 用途 |
| --- | --- |
| `--plan` | 校验并显示逻辑 Job、扫描摘要和依赖边，不创建 Session。 |
| `--dry-run` | 与 plan 相同的预检行为。 |
| `--resume-from PATH` | 从兼容的中断 Session 恢复。 |
| `--verbose`, `-v` | 显示详细终端诊断。 |
| `--log-file PATH` | 额外写入指定日志文件。 |
| `--log-json` | 使用 JSON 文件日志。 |
| `--suppress-warnings` | 抑制捕获的 warning。 |
| `--json` | 输出机器可读的计划或完成摘要。 |

成功执行会在 Project Session 根目录下创建一个 Session 并打印其路径。CLI 正常
输出保持简洁，完整诊断写入 Session 日志。

## 插件

```bash
qphase list [--category NAME] [--tree] [--parent PATH]
qphase show model.vdp_2mode [backend.numpy ...]
qphase template model.vdp_2mode [OPTIONS]
```

`list --tree` 会显示声明过的子插件类和实现。本地插件从当前 Project manifest
指定的路径加载。

## 配置

```bash
qphase config show [--system]
qphase config set KEY VALUE [--system]
qphase config reset [--system] [--force]
qphase config schema PLUGIN_PATH
qphase config options PARENT/SLOT
```

不带 `--system` 时操作 Project 插件默认值（标准布局为
`configs/defaults.yaml`）；带 `--system` 时操作 `~/.qphase/config.yaml` 中的
稀疏用户机器策略。

## GUI

```bash
qphase gui [--host 127.0.0.1] [--port 8000] [--reload]
```

本地 API 没有远程认证，因此只接受 loopback 地址；服务器使用 SSH tunnel。GUI
与 CLI 使用相同的 Project、Workflow、Execution 和 Session 服务。
