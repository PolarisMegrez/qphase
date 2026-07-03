---
description: 后处理架构
---

# 后处理架构

SDE 后处理当前属于 `qphase_sde`，不属于 core。它依赖 SDE result metadata、PSD analyser payload、distribution payload 和 scan parameters，因此 core 只应提供 scheduling 与 registry 机制。

## 当前兼容入口

旧 import path 仍然可用：

```python
from qphase_sde.postprocess import postprocess_run, export_postprocess_bundle
```

core CLI 命令 `qphase postprocess` 也保留为兼容 facade。它在运行时导入 `qphase_sde.postprocess`，并保持现有 CSV 输出行为。

## 拆分后的 Category

新代码应优先使用 category modules：

| Category | Module | 职责 |
| :--- | :--- | :--- |
| Aggregator | `qphase_sde.aggregators.scan_psd` | 加载已保存的 SDE `.npz` 文件，并提取对齐后的 PSD traces。 |
| Fitter | `qphase_sde.fitters.lorentzian` | 从单条 PSD trace 拟合 Lorentzian 峰参数。 |
| Exporter | `qphase_sde.exporters.csv_bundle` | 写出 `fit_results.csv`、`psd_merged.csv` 和可选 distribution bundles。 |

这些模块刻意保持小而稳定，先包裹已经验证过的行为。旧 facade 可以继续编排它们，同时调用方逐步迁移到显式 category。

## Scheduler Workflow Engine

`qphase_sde` 现在暴露 package-level workflow engine：

```text
engine.sde_postprocess = qphase_sde.workflows.postprocess.engine:SDEPostprocessEngine
```

该 engine 包裹现有 postprocess use case，使后处理可以表示为 scheduler job。第一版基于 `run_dir` 处理已保存的 run directory 或 `.npz` 文件；后续版本可以加入更完整的上游 scheduler input 内存聚合。

示例 job 形态：

```yaml
name: postprocess
engine:
  sde_postprocess:
    run_dir: runs/example_session
    scan_param: epsilon
    mode: 0
    overwrite: true
```

## 边界

单个 result 内的分析仍属于 analyser plugins。跨 result 合并属于 aggregators。曲线拟合属于 fitters。文件布局和导出格式属于 exporters。workflow engine 负责把这些能力组合成一个面向用户的 postprocess job。
