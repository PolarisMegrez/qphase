# QPhaseSDE Protocols and Implementer Guide

This document summarizes the minimal cross-domain protocols in `QPhaseSDE.core.protocols` and the domain-level extensions under each package's `protocols.py`.

## Core minimal protocols

- BackendBase: minimal backend capabilities (array creation, linalg, RNG). No third-party imports in core.
- RNGBase: opaque RNG with `seed()` and optional `spawn()`.
- StateBase: minimal state container with `data_view()`, `view()`, `copy()`, `to_backend()`.
- TrajectorySetBase: minimal time-series container.
- Serializable/Snapshotable: lightweight metadata contracts for IO/snapshots.

Semantics: view returns aliasing views when possible; copy returns independent storage; `to_backend()` may copy and should document when it does.

## Domain protocols

- backends/protocols.py: `ExtendedBackend` with optional helpers like `stack`, `to_device`, `complex_view`, `real_imag_split`, and `capabilities()`.
- integrators/protocols.py: `Integrator.step(y, t, dt, model, noise, backend)`; optional `reset()`, `supports_*` feature flags.
- noise_models/protocols.py: `NoiseModel` consuming `NoiseSpecLike` and producing per-step increments.
- visualizers/protocols.py: `Renderer` with `validate(spec)` and `render(ax_or_buffer, data, spec, style)` returning standard metadata.
- states/protocols.py: `ExtendedState` for higher-level operations like `slice()`, `view_complex_as_real()`, `persist()`.

## Visualizers architecture

- Spec layer (Pydantic) validates spec inputs early.
- Renderer layer implements drawing against a provided axis/buffer.
- Service layer selects renderer via registry, merges styles, renders and saves figures, and returns metadata. Supports function-style and class-style renderers.

Current built-ins and conventions:
- Phase portraits: kinds `re_im` and `abs_abs` via `visualizer:phase_portrait`
- Power spectral density (PSD): kinds `complex` and `modular` via `visualizer:psd`
	- Convention `symmetric`/`unitary`: unitary FFT and angular frequency axis ω
	- Convention `pragmatic`: engineering FFT and frequency axis f
	- Axis scales are controlled via style keys `x_scale`/`y_scale` = `linear` or `log`

## Registration and plugins

- Single central registry (`core/registry.py`) with domain aliases (e.g., `visualizers/register.py`).
- Lazy imports supported; rich metadata recorded (registered_at, builder_type, delayed_import, module_path).
- Plugins can register via entry points or dynamic import paths (future work).

Examples:
```python
from QPhaseSDE.core.registry import registry

# Get a function-style renderer and call it
render_pp = registry.create("visualizer:phase_portrait")
meta = render_pp(ax, data, {"kind":"re_im", "modes":[0]}, {"linewidth":0.8})

# Use the service to handle validation, slicing, and saving
from QPhaseSDE.visualizers.service import render_from_spec
render_from_spec({"kind":"psd", "modes":[0,1]}, data, t0=0.0, dt=1e-3, outdir=Path("out"), style_overrides={"x_scale":"log", "y_scale":"log"})
```

## Testing

- Contract tests validate protocol adherence (view/copy semantics, spec validation, registry behaviors).
 - Visualizer tests should include: spec validation failures, style overrides (x/y scale), and sided PSD rendering continuity.
