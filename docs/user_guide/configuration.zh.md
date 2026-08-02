---
description: 任务配置
---

# 任务配置

QPhase 使用经过校验的 YAML 配置描述可复现的逻辑任务。一个 job 选择一个
engine，配置其插件，并可选声明参数扫描或读取上游结果。

## 配置层级

配置按以下顺序合并，后者优先级更高：

1. 核心与插件 schema 默认值。
2. `configs/global.yaml` 中的项目默认值。
3. job YAML。
4. `job.system`，使用相同的 `SystemConfig` schema 覆盖当前 job 的运行策略。

框架运行策略只属于 `SystemConfig`，不会重复设计为 job 顶层快捷字段。

## Job 结构

```yaml
name: vdp_cam
save: true

engine:
  cam: {}

plugins:
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

| 字段 | 含义 |
| --- | --- |
| `name` | 唯一逻辑任务名称。 |
| `engine` | 恰好一个 engine 配置。 |
| `plugins` | engine 所需的命名空间插件配置。 |
| `params` | 可选的 engine 专用参数。 |
| `scan` | 可选的显式 `ScanSpec`。 |
| `input` | 可选的结构化上游输入。 |
| `save` | `true`、`false` 或输出基础名称。 |
| `system` | 可选的 `SystemConfig` 同 schema 覆盖。 |

部分现有资源包仍兼容 job 顶层插件命名空间。新资源包应优先使用显式
`plugins` 映射。

## 参数扫描

扫描必须显式声明。插件配置中的列表始终是插件本身的普通值，不再被解释为扫描。
已知标量模型参数若使用旧的列表扫描语法，配置加载器会给出迁移错误。

每条轴包含显示名称、插件目标路径和一种数值生成方式：

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

| 生成方式 | 含义 |
| --- | --- |
| `values: [...]` | 显式值列表。 |
| `linspace: {start, stop, num, endpoint}` | 线性间隔；`endpoint` 默认 `true`。 |
| `logspace: {start, stop, num, endpoint, base}` | `base` 的指数；`base` 默认 `10`。 |

`combine: cartesian` 按 YAML 轴声明顺序产生结果维度，上例 shape 为
`(31, 101)`。`combine: zipped` 要求所有轴长度相等，并只产生一个 `point`
维度。

扫描不会产生 scheduler 子任务。engine 收到一个运行时 `ParameterGrid`，自行选择
逐点、tile、融合或 GPU 执行策略。session manifest 和输出树中仍然只有一个逻辑
job 条目与一个目录。

## 上游输入

完整数据集输入写为：

```yaml
input:
  from: vdp_2mode_cayley_sim
  mode: dataset
```

`mode: dataset` 只调用一次下游 engine，并传入完整结果。`mode: map` 在同一个逻辑
下游 job 内惰性处理选中的 point/group view：

```yaml
input:
  from: source_scan
  mode: map
  select: {omega_a: [0.01, 0.1]}
  group_by: [gamma_b]
```

`select` 按命名轴值筛选；`group_by` 将指定轴上的 view 组合为
`AggregateResult`。map 不建立参数点目录。字符串形式的 `input` 和旧
`aggregate_input` 字段会产生明确迁移错误。

## SystemConfig

内置默认值位于 `qphase.core/system.yaml`。用户级覆盖可位于
`~/.qphase/config.yaml`；读取配置不会自动创建该文件。通过
`qphase config set --system` 写入时，只持久化相对内置/机器默认值发生变化的字段。
`QPHASE_SYSTEM_CONFIG` 或显式加载路径可提供更高优先级的文件。解析顺序为：包内
默认值、可选机器策略、用户覆盖、环境变量指定文件、显式加载路径。

```yaml
auto_save_results: true

reporting:
  progress:
    refresh_interval: 0.5
    non_tty_milestone_percent: 10.0
    eta_warmup_seconds: 2.0
    eta_min_samples: 3
    eta_smoothing: 0.25
  logging:
    session_file: true
    filename: qphase.log
    file_level: DEBUG
    console_level: WARNING
    format: text
    capture_warnings: true

scan_runtime:
  storage_layout: auto
  auto_shard_threshold_mib: 512
  shard_target_mib: 128
  checkpoint:
    enabled: false
    interval_chunks: 1
    keep_on_success: false
  resources:
    cpu_worker_limit: null
    memory_limit_mib: null
    gpu_device: null
    gpu_memory_fraction: null
```

`storage_layout` 可取 `auto`、`single`、`sharded` 或 `per_point`。`auto` 默认在
dataset 超过 512 MiB 时分片。资源字段由 core 采集并通过 `ExecutionContext`
传给 engine；每个逻辑 job 还会单独采样 CPU、主机内存和可选 backend device 的
动态事实。scheduler 不据此进行多 job 资源调度。

checkpoint 只覆盖已完成的 scan chunk，不覆盖 SDE 单条轨迹内部的时间步。
resume 会校验配置、插件、backend 与 dtype，只有兼容的 checkpoint 才会被接受。
