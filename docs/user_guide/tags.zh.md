---
description: Tags、Annotation 与项目对象 Catalog
---

# Tags 与 Catalog

Tags 是 QPhase 的组织层：为 Project、Workflow、Execution、Session、
Artifact 和 Artifact Occurrence 打标签，便于事后检索、筛选与分组。Tag 从不
改变一次运行的计算结果——它们是关于对象的元数据，存放在不可变的运行记录
之外。**Catalog** 是让这些标签可查询的搜索索引。

## 对象模型

共有七种对象可以携带 tag，各自有稳定 identity：

| 种类 | Identity | 含义 |
| --- | --- | --- |
| Project | `project_id` | 以 `qphase.toml` 为根的整个项目。 |
| Workflow revision | `workflow_id@revision` | Workflow 文档的一个内容版本。 |
| Job | `workflow_id@revision:job_name` | 某 Workflow 版本中的一个逻辑 Job。 |
| Execution | `execution_id` | 一次已提交的 Workflow 运行。 |
| Session | `session_id` | Session 根目录下一次持久化的运行目录。 |
| Artifact | `artifact_id` | 一个不可变的结果包（identity，不是路径）。 |
| Occurrence | `artifact_id:session_id:job_name` | 某 Artifact 在某 Session 某 Job 中的一次产出。 |

Artifact 是 identity,Occurrence 是这个 identity 被产出的一处位置。同一个
Artifact 可以出现在多个 Session 中——Artifact 上的 annotation 跟随
identity,Occurrence 上的 annotation 只属于该产出上下文。

由于 `:` 用于拼接上述 identity,job 名与 artifact id 绝不能包含它;新写入
会在校验时直接拒绝。

## Tag 语法与 Namespace

Tag 形如 `namespace:value` 或 `namespace:path/to/value`，全部小写。
Namespace 匹配 `[a-z][a-z0-9_]*`；每段路径匹配 `[a-z0-9][a-z0-9_.-]*`。
写入时会做规范化，因此 `Task:Scan` 与 `task:scan` 是同一个 tag。

层次路径是一等公民：用 descendant 匹配查询 `purpose:paper` 时也会命中
`purpose:paper/fig3`。

## Tag Policy

可选的项目 tag policy 位于 `configs/tags.yaml`(schema
`qphase.tag-policy/1`):

```yaml
schema: qphase.tag-policy/1
namespaces:
  stage:
    cardinality: one        # 每个对象至多一个 stage:* 取值
    values: [q1, q2]        # 封闭取值表
  task:
    open: true              # 任意取值均可
  model:
    aliases: {vdp: vdp_2mode}
    objects: [workflow, session]   # 限制可作用的对象种类
    inherit: false           # 不沿对象层级向下传递
reserved_namespaces: [system]
retention_inherits_to_occurrences: true
```

各 namespace 规则：

- `cardinality`:`many`（默认）或 `one`。在 `one` 的 namespace 中，更近的
  assignment 会 **shadow** 更远的（见下）。
- `values`：封闭取值表。省略并设置 `open: true` 表示自由取值。
- `aliases`：把别名拼写映射到规范取值。
- `objects`：限制该 namespace 可挂在哪些对象种类上。
- `inherit`：为 `false` 时，tag 不会从 workflow 传到 job、session 或
  occurrence。
- `reserved_namespaces`：处处拒绝（为未来的内建用途保留）。
- `retention_inherits_to_occurrences`:session 的 retention 是否默认传给它的
  occurrence（默认 `true`)。

没有 policy 文件时没有任何治理：tag 只做语法规范化。Policy 有一个基于内容
hash 的 **revision**；每个 tag assignment 都会冻结验证它时的 revision，因此
之后修改 policy 不会改写历史 provenance。每个 assignment 还会冻结治理它的
最小 namespace 规则(`inherit`、`cardinality`、`objects`):历史 assignment
的 effective tag 解析按写入时的规则进行;在规则冻结之前写入的 assignment
（或没有治理规则的，例如私有 tag）则回退到读取时的当前 policy。用
`qphase tag policy show` 查看 policy，用 `qphase tag policy validate` 校验。

## 四种 Tag Scope

Tag 从四个 scope 进入系统，按由远到近排列：

1. **Declared tag**(workflow/job）：写在 workflow YAML 或 job 定义中。
   Workflow 运行时，解析后的 declared tag 会冻结进 session 的
   `workflow_snapshot.yaml` / `tag_snapshot.yaml`——之后再改 workflow 文件
   不会改写历史。在 catalog 中，workflow revision 的 declared tag 按来源
   优先级裁决：当前 workflow 文件在场时胜出（按当前 policy
   canonicalize）；仅存在于历史 session 中的 revision 保留其冻结 snapshot
   的 canonical 声明；若同一 revision 的多个冻结 snapshot 互相矛盾，则
   回退为不带 policy provenance 的原始语法 tag。
2. **Submission tag**(execution)：提交时给定并连同每个 tag 的最小
   namespace 规则一起冻结在 execution 记录上。每次正式运行都拥有一个
   execution 记录——排队运行（`qphase execution` / GUI）与直接 `qphase
   run` 皆如此——session manifest 通过 `execution_id` 回链到它。每个
   submission tag 携带从 execution id 确定性派生的稳定 assignment id，在
   execution 记录、session manifest 与 catalog 三处一致。
   只有 execution 仍在排队时才能编辑（`qphase execution tag`)；替换排队中
   的 tag 会使旧 assignment id 作废。
3. **共享 annotation**：写进项目内的 annotation 文档
   (`session_annotations.json`、`artifact_annotations.json`、
   `.qphase/project_annotations.json`)。所有打开该项目的人都可见，应当纳入
   版本控制。
4. **用户私有 tag**：存放在 `~/.qphase/gui/` 下的用户级数据库中，绝不写入
   项目。任何 tag 命令加 `--private` 即可。私有 tag 只对你自己参与查询，也
   可以用 `qphase tag promote` 提升为共享 annotation。

## 继承与 Shadow

Tag 沿层级向下传递：project → workflow revision → job，以及
workflow/execution → session → artifact occurrence。标记为
`inherit: false` 的 namespace 只留在声明它的对象上。一条历史 assignment
是否继承、是否 shadow，由冻结在它上面的 namespace 规则决定（见 *Tag
Policy*)，绝不会因之后的 policy 修改而被重新裁决。

在同一对象上，`cardinality: one` 的 namespace 中更近的 scope 胜出：session
annotation 会 shadow 同 namespace 的 workflow declared 取值，私有 tag 则
shadow 两者。被 shadow 的 tag 在 API 中仍带 provenance 可见，但默认列表不
显示，也不参与查询匹配。

`lifecycle` 从不继承——它描述对象本身。policy 允许时 `retention` 从
session 传给 occurrence,occurrence 也可以在本地覆盖。设置 session 的
retention 时会把 policy 的继承开关一并冻结，之后的 policy 修改不会重新
裁决历史；在此契约之前写入的 session 回退当前 policy，由 Phase 4 迁移回填。

## Lifecycle 与 Retention

Lifecycle 和 retention 是类型化字段，不是 tag:

- **Lifecycle**(`active`、`reference`、`superseded`、`archived`）标记对象
  所处阶段。`archived` 表示冷存储：对象仍可查询，但视为退出活跃使用。
- **Retention**(`transient`、`preserve`、`evidence`、`pinned`）声明数据
  需要保留的级别。`evidence` 与 `pinned` 会进入内建的 *paper-evidence*
  虚拟目录。

```bash
qphase session lifecycle <session-id> reference
qphase session retention <session-id> evidence
qphase artifact lifecycle <artifact-id> archived
qphase occurrence retention <session-id> <artifact-id> pinned [--job NAME]
```

## CLI 查询

所有 list 命令接受同一组过滤器：

```bash
qphase session list --tag task:scan --tag-without task:wip \
    --tag-any method:cam --tag-any method:fpgen \
    --tag-descendant purpose:paper --tag-namespace model \
    --facet status=completed --range start_time=2026-08-01.. \
    --lifecycle active --retention evidence \
    --direct --limit 50 --offset 0
```

- `--tag` 要求具有该 effective tag（可重复，AND)。`--tag-any` 命中任一重复
  值即可。`--tag-without` 排除。
- `--tag-descendant` 匹配该 tag 或其任意下级路径；`--tag-namespace` 匹配
  整个 namespace。
- `--facet k=v` 按对象 facet 过滤；`--range k=low..high` 按区间过滤（两端
  均可留空）。
- `--direct` 忽略继承来的 tag，只匹配直接 assignment。

list 命令还把派生 facet 快捷过滤暴露为按对象种类校验的旗标：job 的
`--plugin`（命中任一声明的 plugin)、artifact 的 `--quantity`（命中任一产出
quantity)，以及 session 的 `--model`/`--engine`/`--has-model`（经由该 session
workflow revision 的 job 解析）。同样的过滤器也存在于 `CatalogQuery` 与 GUI
catalog 路由的 HTTP query 参数上；用在错误的对象种类上会被拒绝。

各对象命令组：

```bash
qphase project tag --add task:paper [--private]
qphase project alias "paper project" [--clear]
qphase project note "results for the paper" [--clear]
qphase workflow list [--collection NAME] [--query TEXT] [--json]
qphase workflow tag <workflow_id@revision> --add task:reviewed [--private]
qphase job list / qphase job tag <workflow_id@revision:job_name> --add ...
qphase execution tag <execution-id> --add ... [--private]   # 共享层仅限排队中
qphase session list|tag|lifecycle|retention ...
qphase artifact list|tag|lifecycle|retention ...
qphase occurrence list [--session ID] [--artifact ID]
qphase occurrence tag <session-id> <artifact-id> --add ... [--job NAME]
```

Occurrence annotation 按产出 job 键控；当一个 artifact 在同一 session 的多个
job 中出现时，必须用 `--job` 消歧。

把私有 tag 提升到共享层：

```bash
qphase tag promote <kind> <object-id> <tag>
```

## 虚拟目录与 Saved View

内建虚拟目录按语义分组 session:`by-model`、`paper-evidence`、
`diagnostics`、`superseded`、`cold-storage`。`by-model` 列出 workflow
revision 声明了任意 model plugin 的 session；要筛选某个具体 model，改用
`model` 查询过滤器。Saved view 是用户私有的命名过滤器：

```bash
qphase view save review --kind session --tag task:scan --lifecycle active
qphase view list
qphase view delete review
```

## Reindex 与 Location Issue

Catalog 是位于 `.qphase/object_catalog.sqlite` 的 SQLite read model，从
磁盘真值重建。一个轻量 fingerprint 探针会在 manifest 或 workflow 文件变化时
自动重建；`qphase project reindex` 强制重建并打印各类对象计数。无法索引的
artifact 位置会报告为 location issue:`unsupported`（未知 manifest
schema)、`corrupt`(manifest 不可读）或 `conflict`（同一 artifact
identity 的两处 occurrence facet 不一致）。`qphase project reindex` 会列出
这些 issue，绝不会静默丢弃。

## 迁移边界

1.x 到 2.x 的 major 迁移已完成：迁移命令与兼容层已在 2.0.0 边界移除。
复现旧 major 的数据需使用其归档发布、环境或分支——见
[版本与迁移策略](../dev_guide/versioning.zh.md)。
