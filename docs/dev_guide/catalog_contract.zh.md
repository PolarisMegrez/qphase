---
description: Catalog、identity 与 annotation 开发契约
---

# Catalog 契约

本页固定 tag/annotation/catalog 子系统对外保证的契约。改动这些区域的代码
必须维持这些契约；修改契约本身需要提升 schema 或 read-model 版本。

## Identity 规则

- 路径是**位置**，绝不是 identity。Session 和 Artifact 可以移动或复制，其
  id 不变。
- **Artifact** 的 identity 是 `qphase.artifact/4` manifest 中不可变的
  `artifact_id`。Catalog 绝不制造 id:manifest 不可读或不受支持的位置不会
  产生任何 artifact 行，而是作为 `corrupt`/`unsupported` location issue
  报告。
- **Occurrence** 的 identity 是三元组 `artifact_id:session_id:job_name`。
  `session_annotations.json` 内的 occurrence annotation 键为
  `job_name:artifact_id`，因此同一 session 中同一 artifact 的两个
  occurrence 永不碰撞。旧式的裸 artifact 键属于迁移输入，由
  `qphase project migrate --dry-run` 标出。
- **Workflow revision** 的 identity 是 `workflow_id@revision`，其中
  `revision` 是 workflow 文档的内容 hash;**job** 的 identity 扩展为
  `workflow_id@revision:job_name`。revision 由 `configs/workflows` 文件、
  session snapshot 和 execution 记录确定性地重建——同样的内容总是得到同样
  的 revision id。
- **Execution** 的 identity 是持久化 `qphase.execution/1` 记录中的
  `execution_id`。

## Provenance 冻结

- Workflow/job 的 declared tag 按 session 冻结进 `workflow_snapshot.yaml`
  和 `tag_snapshot.yaml`（规范化 tag、稳定 assignment id、验证它们的
  policy revision)。之后修改 workflow 文件或 policy 都不会改写历史
  session 的 provenance。
- Submission tag 连同 `tag_policy_revision` 冻结在 execution 记录上；只有
  execution 仍在排队时才可修改。
- 每个 annotation `TagAssignment` 在写入时冻结 `policy_revision`。
  Assignment 不可变：编辑一个 tag 是删除旧 assignment 并新增一个，因此任何
  effective tag 都能引用稳定的 `assignment_id`。
- Declared tag 没有 annotation 文档；其 assignment id 由声明的 identity 经
  `sha256` 确定性导出。

## Catalog 是可重建的 Read Model

- Catalog(`<project>/.qphase/object_catalog.sqlite`,read-model schema
  `qphase.catalog/3`）是磁盘真值的纯函数：manifest、snapshot、execution
  记录、workflow 文件、tag policy 和 annotation 文档。任何时候都可以删除并
  用 `qphase project reindex` 重建。
- 读取前用轻量 fingerprint(manifest/记录计数加上 policy 与 workflow 的
  mtime）探测；不匹配则先重建再服务。
- 数据库损坏、schema 不匹配或属于其他项目时，从磁盘真值重建，而不是返回空
  结果。`meta` 表把数据库绑定到 `project_id`，因此复制来的项目不会读到别的
  项目的 catalog。
- 同一项目的并发重建由按路径的锁串行化；每次重建原子地删除并重建数据库
  文件。

## Annotation 文档与锁

- 共享 annotation 有三类文档：`session_annotations.json`
  (`qphase.session-annotations/1`，同时携带 per-occurrence annotation)、
  `artifact_annotations.json`(`qphase.artifact-annotations/1`）和
  `.qphase/project_annotations.json`(`qphase.project-annotations/1`，承载
  project 自身以及 workflow revision、job、execution 的 annotation)。
- 写入做乐观并发检查（`revision` 必须匹配）,read-check-write 循环由同级
  `<name>.lock` 文件中的跨进程锁串行化（阻塞式 `msvcrt`/`fcntl` 锁，持有
  者崩溃时由 OS 释放)。Revision 冲突以 `RuntimeError` 抛出；调用方重新加载
  后重试。
- Annotation 写入绝不触碰 artifact manifest 或 payload。
- 用户私有状态（私有 tag、私有 alias/note、saved view、项目位置）存放在
  `~/.qphase/gui/<project_id>.sqlite`，读取时叠加。私有状态绝不进入共享
  文档或 catalog。

## Artifact 与 Occurrence 的分离

- Artifact 变更（`tag_artifact`、lifecycle）要求恰好一个已索引位置；多个
  位置抛出 `ArtifactAmbiguousError`，绝不静默取第一个。
- Occurrence 变更定位到一个产出上下文；当一个 artifact 在同一 session 的
  多个 job 中出现时，必须给出 job 名。
- 同一 artifact 的后续 occurrence 若 identity facet 与首个已索引 occurrence
  不一致，记为 `conflict` location issue;artifact 行保持首个所见。

## 明确不做

按 Phase 3 审计，本子系统明确不做：

- 引入 Campaign/Study 对象；
- 建立第二个 catalog 或 archive 目录树；
- 把私有 annotation 写入共享真值；
- 实现通用 ORM、event sourcing 或自动资源调度；
- 恢复 artifact payload hash;
- 把 Phase 6 的 worker/GUI renderer 工作提前到本阶段；
- 为损坏文件制造 artifact id，或静默返回空 catalog。
