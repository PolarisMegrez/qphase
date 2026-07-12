---
layout: default
title: qphase_sde
parent: API 参考
nav_order: 2
---

# `qphase_sde` API 参考

`qphase_sde` 包是 QPhase 的随机微分方程（SDE）资源包，提供：

*   `engine.sde` — SDE 仿真引擎。
*   积分器插件（`euler_maruyama`、`milstein`、`srk`）。
*   SDE 模型实现（如 `kerr_3pa`、`kerr_3mode`、`vdp_level3`）。
*   分析器插件（`psd`、`dist`、`pdist`、`lorentz_fitter`）。

本节专门记录 `qphase_sde` 的行为。核心框架（调度器、注册表、插件加载、结果协议）请参考 [Core API](../core.md)。

## 目录

*   [引擎](./engine.zh.md) — `EngineConfig`、`SDEResult`、`save_stride` 与分析模式。
*   [积分器](./integrators.zh.md) — `Integrator` 协议与 `GenericSRK`。
*   [模型](./models.zh.md) — 内置模型与 `SDEModel` 协议。
*   [分析器](./analyzers.zh.md) — PSD、分布与 Lorentz 拟合分析器。
*   [输出格式](./output.zh.md) — `.npz`、`.csv` 与合并 bundle 的字段说明。
