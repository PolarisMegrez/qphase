---
description: 本地 GUI 与长时间 Execution 管理
---

# GUI 与本地 API

QPhase Workbench 是 CLI 同一服务层的可视化客户端，用于管理 Workflow、排队或
运行中的 Execution、Session 历史、日志与 Artifact。科学算法和插件内部并行仍由
资源包 engine 负责。

## 启动

```bash
pip install "qphase[gui]"
qphase gui
```

浏览器打开 `http://127.0.0.1:8000`。本地 API 没有认证，因此拒绝公开监听地址；
远程工作站或服务器应使用 SSH tunnel。

## 长时间 Execution

提交 Workflow 会创建异步 Execution。当前 Workbench 同时运行一个 Execution，
并维护有界 FIFO 队列。关闭浏览器不会停止服务端 worker。

- Execution 页面显示当前状态、活动 Job/engine stage、scan 进度和插件；
- `events.jsonl` 保存采样后的进度与控制事件；
- Session 日志按 `SystemConfig` 保存完整诊断；
- `session.lock` 提供 owner 心跳，失去 owner 的 `running` Session 会显示为
  `interrupted`，但不会篡改原 manifest。

取消是协作式的，QPhase 不会强行终止正在运行的 GPU kernel。

## 暂停与修订

暂停只在逻辑 Job 边界生效。Execution 排队或暂停时，可以用保持同名的完整配置
替换尚未开始的 Job；QPhase 会重新校验 Job 图和插件配置，并记录修订事件。

因此，长时间 `SDE -> analysis` Workflow 可以在 SDE 完成后、分析开始前修改下游
Job。但这不能恢复上游已经丢弃的数据，例如 `keep_traj: false` 后无法追加依赖原始
轨迹的 analyser。

## Workflow 与 Session

Workflow catalog 支持文本搜索以及可点击的 Collection/Tag 筛选，大型 Project 不必
浏览一个扁平长列表。

Workflow 文档使用内容 revision，过期 GUI 写入会冲突而不是覆盖 IDE 修改。Session
展示状态、别名、备注、逻辑 Job、事件、日志和 Artifact。删除非运行 Session 时先
移动到 Project 本地回收站，永久清理需要显式操作。

未来 Archive 可以提供基于 Project/Workflow/Session ID 的虚拟目录、收藏与私人备注，
但它始终是用户本地元数据，不影响执行与复现。

主要 API 为 `/workflows`、`/plans`、`/executions`、`/sessions` 与
`/workflow-docs`。GUI 当前不提供多用户认证、公开网络服务、多 Execution 资源调度、
插件热加载、SDE 时间步 checkpoint、在线 FFT 或 Archive 虚拟目录。
