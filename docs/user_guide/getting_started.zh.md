---
description: 使用 QPhase Project 与 Workflow
---

# 快速开始

## 安装

QPhase 需要 Python 3.11 或更高版本。在本 monorepo 中执行：

```bash
git clone https://github.com/PolarisMegrez/qphase.git
cd qphase
uv sync
```

## 创建 Project

```bash
qphase project init my-research --name "My Research"
cd my-research
qphase project show
```

初始化后包含：

```text
qphase.toml                    # Project 身份与相对路径
configs/defaults.yaml         # 项目范围的插件默认值
configs/workflows/            # 可版本控制的 Workflow 文档
models/                       # 本地插件目录
runs/                         # Session 记录（通常不进入版本控制）
```

在 Project 目录外执行时使用 `qphase --project <path> ...`。否则 QPhase 会从
当前目录向上查找 `qphase.toml`。

## 创建 Workflow

新建 `configs/workflows/examples/test_run.yaml`：

```yaml
schema: qphase.workflow/2
id: test_run
title: First SDE run
description: Small CPU example for installation verification
collection: examples
tags: [quickstart, sde]

jobs:
  - name: simulate
    save: true
    engine:
      sde:
        t0: 0.0
        t1: 10.0
        dt: 0.01
        n_traj: 16
        seed: 42
        ic: [["1.0+0.0j", "0.0+0.0j"]]
    backend:
      numpy: {float_dtype: float64}
    integrator:
      euler_maruyama: {}
    model:
      vdp_2mode:
        omega_a: 1.0
        omega_b: 1.0
        gamma_a: 0.1
        gamma_b: 0.1
        Gamma: 1.0
        g: 0.5
```

整个文档称为 Workflow，`simulate` 是其中的逻辑 Job。一个 Workflow 可以包含
多个通过 `input` 或 `depends_on` 连接的 Job。

## 检查与运行

```bash
qphase workflow list
qphase workflow show test_run
qphase run test_run --plan
qphase run test_run
```

`--plan` 只校验并显示逻辑 Job 图，不创建 Session。正常执行时，CLI 显示简洁的
进度，最终输出 Session 路径；完整诊断日志按 `SystemConfig` 写入 Session。

## 查看结果

```text
runs/YYYY/MM/<session-id>/
  session_manifest.json
  events.jsonl
  <job-name>/
    config_snapshot.json
    artifact_manifest.json
    qphase.log
    ...
```

`qphase gui` 可用于可视化浏览 Workflow 与 Session。CLI 仍然是脚本、agent 和
远程服务器场景中的权威接口。

继续阅读[核心概念](concepts.zh.md)、[Workflow 配置](configuration.zh.md)与
[结果和可复现性](output.zh.md)。
