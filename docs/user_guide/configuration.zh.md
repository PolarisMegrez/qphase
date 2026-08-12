---
description: Project、Workflow、Job 与系统配置
---

# Workflow 配置

QPhase 将可迁移的研究意图与机器策略分开：

1. 插件 schema 默认值由已安装插件定义；
2. Project 插件默认值位于 `qphase.toml` 的 `paths.defaults`，通常为
   `configs/defaults.yaml`；
3. 版本化 Workflow 文档定义元数据与逻辑 Job；
4. `job.system` 可以为单个 Job 覆盖同一套 `SystemConfig` schema。

后者优先级更高。Project 路径只允许出现在 `qphase.toml` 中。

## Project Manifest

```toml
schema = "qphase.project/2"
project_id = "my-research"
name = "My Research"

[paths]
workflows = "configs/workflows"
defaults = "configs/defaults.yaml"
plugins = ["models"]
sessions = "runs"
```

所有路径都是相对于 Project 根目录的可迁移路径，绝对路径和 `..` 会被拒绝。
Project 发现顺序为 `--project`、`QPHASE_PROJECT`、从当前目录向上查找。

## Workflow 文档

```yaml
schema: qphase.workflow/2
id: vdp_cam
title: VDP CAM scan
description: Optional human-readable purpose
collection: vdp_2mode
tags: [cam, multistability]

jobs:
  - name: solve
    save: true
    engine:
      cam: {}
    backend:
      numpy: {float_dtype: float64}
    model:
      vdp_2mode:
        omega_a: 0.0
        omega_b: 0.0
        gamma_a: 2.0
        gamma_b: 0.5
        Gamma: 0.0001
        g: 0.5
    cam_solver:
      multistability: {n_guesses: 50, guess_bounds: auto}
```

`schema`、`id`、`title` 与非空 `jobs` 必填。Workflow ID 在 Project 内必须
唯一，移动文件时不得改变。QPhase 2 明确拒绝没有 `qphase.workflow/2` 包装的旧
顶层 Job 列表。

Collection 与 Tag 是可版本控制的元数据。目录可以按 Collection 分类，但目录不是
Workflow 身份。

## 逻辑 Job

| 字段 | 含义 |
| --- | --- |
| `name` | Workflow 内唯一的 Job 名称。 |
| `engine` | 恰好一个 engine 及其配置。 |
| 插件命名空间 | `backend`、`model`、`integrator`、`analyser` 等插件配置。 |
| `params` | 可选 engine 参数。 |
| `scan` | 可选显式 `ScanSpec`。 |
| `input` | 可选结构化上游数据输入。 |
| `depends_on` | 对其他 Job 的显式控制依赖。 |
| `save` | `true`、`false` 或 Artifact 基础名称。 |
| `system` | 可选的单 Job `SystemConfig` 覆盖。 |

项目范围数值默认值写入 `configs/defaults.yaml`。仅存在默认值不会激活可选插件；
Workflow 必须显式选择对应命名空间。

## 参数扫描

插件配置中的列表始终是字面值。扫描必须使用 `ScanSpec`：

```yaml
scan:
  combine: cartesian
  axes:
    omega_a:
      target: model.vdp_2mode.omega_a
      logspace: {start: -3, stop: -1, num: 31}
    gamma_b:
      target: model.vdp_2mode.gamma_b
      linspace: {start: 0.2, stop: 1.1, num: 101}
```

每个轴只能使用 `values`、`linspace` 或 `logspace` 之一。`cartesian` 按 YAML
声明顺序生成维度；`zipped` 要求各轴等长并生成一个 `point` 维度。

扫描仍然是一个逻辑 Job。engine 接收 `ParameterGrid` 后自行选择逐点、tile、融合或
GPU 策略；core 不为每个参数点创建 Job 或 Session。

## 数据流

```yaml
input:
  from: simulate
  mode: dataset
```

`dataset` 一次传入完整上游结果；`map` 在一个下游 Job 内惰性产生 point/group
view，并可使用 `select` 与 `group_by`。字符串 `input` 与 `aggregate_input` 已删除。

## SystemConfig

`SystemConfig` 是与 Project 无关的机器策略，按以下顺序合并：包内默认值、可选站点
策略、`~/.qphase/config.yaml` 稀疏用户覆盖、`QPHASE_SYSTEM_CONFIG`、显式加载
路径。

它只包含结果自动保存、扫描 Artifact 布局、chunk checkpoint、资源提示、进度与
日志策略，不包含 Workflow、插件或 Project 路径。`qphase config show --system`
显示机器策略；`qphase config show` 显示 Project 插件默认值。

checkpoint 只覆盖已完成的 scan chunk，不覆盖 SDE 内部时间步。
