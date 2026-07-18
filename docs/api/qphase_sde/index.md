---
layout: default
title: qphase_sde
parent: API Reference
nav_order: 2
---

# `qphase_sde` API Reference

The `qphase_sde` package is the stochastic-differential-equation (SDE) resource
package for QPhase. It provides:

*   `engine.sde` — the SDE simulation engine.
*   Integrator plugins (`euler_maruyama`, `milstein`, `srk`).
*   SDE model implementations (`vdp_2mode`, `kerr_2mode`, `kerr_3mode`).
*   Analyzer plugins (`psd`, `dist`, `pdist`, `lorentz_fitter`).

This section documents `qphase_sde`-specific behavior. For the core framework
(scheduler, registry, plugin loading, result protocols), see the
[Core API](../core.md).

## Sections

*   [Engine](./engine.md) — `EngineConfig`, `SDEResult`, `save_stride`, and
    analyze mode.
*   [Integrators](./integrators.md) — `Integrator` protocol and `GenericSRK`.
*   [Models](./models.md) — built-in models and the `SDEModel` protocol.
*   [Analyzers](./analyzers.md) — PSD, distribution, and Lorentzian-fitting
    analyzers.
*   [Output Schemas](./output.md) — `.npz`, `.csv`, and merged bundle formats.
