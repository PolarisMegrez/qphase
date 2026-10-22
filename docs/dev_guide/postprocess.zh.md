---
description: 后处理架构
---

# 后处理架构

后处理不再是一个独立的 core 命令，也不是一个独立的资源包。它被表达为一个普通的 scheduler job：使用 SDE engine 的 `mode: analyze` 加上 `analyser.lorentz_fitter` 插件。

## 设计原则

- **Core** (`qphase`) 在 `qphase.core.aggregation` 中提供通用的内存聚合容器，
  并提供 typed product 持久化层。
- **`qphase_sde`** 提供 SDE 专属的跨 job analyser：`lorentz_fitter` 处理 PSD 峰，
  `allan_scaling` 处理白 FM 窗口和微扰幂律；其结果以 typed product 形式持久化。
- **`qphase postprocess` CLI 命令已被移除**，请改用 `qphase run <workflow.yaml>`。

## 工作流示例

```yaml
- name: sim
  save: true
  scan:
    axes:
      omega_a:
        target: model.kerr_2mode.omega_a
        values: [0.9, 1.1]
  engine:
    sde:
      t1: 1.0
      dt: 0.01
      n_traj: 2
  model:
    kerr_2mode:
      omega_a: 0.9
      omega_b: 1.0
      chi: 0.01
      gamma_a: 0.1
      gamma_b: 0.1
      g: 0.1
  analyser:
    psd:
      measurement:
        complex_field: {channels: [0], representation: complex}
  input_bindings:
    psd:
      signal: input.trajectories

- name: fit
  input:
    from: sim
    mode: dataset
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: omega_a
      mode: 0
```

scheduler 会：

1. 编译 `sim` scan，并由 SDE engine 将其作为一个逻辑 dataset 执行。
2. 将完整 dataset 一次传给 `fit` job。
3. 在 `analyze` 模式下运行 `analyser.lorentz_fitter`，其 typed product 会持久化到 `fit` job 的 run 目录。

## 输出文件

analyser 不再写 CSV/NPZ/PKL 旁路文件：所有结果都以一个 typed product 的形式持久化到 job 目录的 artifact manifest 中。
可用 `qphase.data.load_products(job_dir)` 加载。

| Product | 生成者 | 内容 |
| :--- | :--- | :--- |
| `lorentz_fitter` | `lorentz_fitter` | 每个扫描值一行，包含拟合参数、由协方差得到的标准差，以及合并后的 PSD/SEM 曲线。 |
| `allan_scaling` | `allan_scaling` | 逐点白 FM 窗口、Allan 强度、不确定度、频率与门控，以及共同 tau/epsilon 窗口与 bootstrap 幂律拟合。 |

## 边界

- 单个 result 内的分析（单 job PSD、寻峰、distribution）属于 `analyser` 插件。
- Dataset view、内存聚合容器与 typed product 持久化属于 core。
- SDE 专属的曲线拟合与 payload 提取属于 `lorentz_fitter`、`allan_scaling` 等下游
  `qphase_sde` analyser 插件。
