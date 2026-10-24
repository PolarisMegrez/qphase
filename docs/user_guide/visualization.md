---
description: Rendering typed products with qphase_viz
---

# Visualization

`qphase_viz` turns a **typed result bundle into figures**. It is a translation
layer, not an analysis tool: a plotter selects named typed products through the
`DatasetResultProtocol` contract, decides which array becomes which axis, line or
colour, and renders it. It never runs an analyser, never reads files by path and
does not depend on `qphase_sde` or `qphase_cam`.

The same plotter and configuration layer is the intended backend for the Phase 11
GUI preview (not available yet). Publication figures remain ordinary scripts
built on top of it, which is also how a highly customised layout is achieved: use
the plotter for the data-to-graphics conventions, then override the layout.

## Running the engine

A visualization job is a normal Workflow v2 job that consumes an upstream dataset:

```yaml
schema: qphase.workflow/2
id: typed_viz
title: Typed visualization
jobs:
  - name: plot_phase
    input: {from: my_simulation, mode: dataset}
    engine:
      viz: {output_dir: figures, format: png}
    visualizer:
      phase_plane:
        plots: [{channel_x: 0, mode: hist2d}]
      power_spectrum:
        plots: [{channels: [0], scale: log}]
```

`output_dir` and `format` are the only engine-wide output settings; every plot
spec lives under its own `visualizer.<plugin>` key. From Python the same objects
are built directly:

```python
from qphase_viz import VizEngine, VizEngineConfig
from qphase_viz.plotters.phase import PhasePlanePlotter

engine = VizEngine(
    config=VizEngineConfig(output_dir="figures", format="png"),
    plugins={"visualizer": PhasePlanePlotter(plots=[{"channel_x": 0}])},
)
result = engine.run(bundle)  # bundle implements DatasetResultProtocol
```

`result.data` is the list of written paths and `result.metadata` only counts
them. There is no analysis payload and no second serialization: a run persists
its figures under `output_dir`, and `result.save(path)` is an explicit no-op kept
because the core `ResultProtocol` requires the method. Whether figures should
become Artifact attachments instead is a later review item — nothing above
depends on it.

## Styles

Figures render with a named style:

| Style | Typography |
| --- | --- |
| `publication` (default) | STIX text and math, 12 pt body, 14.4 pt axis labels |
| `sm` | the same families, one size step down (10.5 pt body) |
| `sans` | Arial with sans math, 18 pt labels / 15 pt ticks |

All three share the axis geometry (1 pt spines, no top/right spine, inward
ticks, frameless legend, 300 dpi export).

Set the style and text export once on the engine, or per figure:

```yaml
engine:
  viz:
    output_dir: figures
    format: svg
    style: publication        # or sm / sans
    editable: true
    style_overrides: {axes.grid: false}
visualizer:
  time_series:
    plots:
      - channels: [0]
        style: sm                        # this figure only
        style_overrides: {axes.labelsize: 12}
```

Precedence for every figure is **single plot > engine > `publication` default**,
and `style_overrides` is applied last, so an explicit rc value wins over the
style table and over the editable switch.

Styling happens inside a local Matplotlib rc context: rendering never changes the
process-wide `rcParams`, so plotter calls do not leak into other figures.

## Editable text

With `editable: true` (the default) SVG keeps real text (`svg.fonttype: none`)
and PDF embeds TrueType (`pdf.fonttype: 42`), so labels stay editable in
Inkscape or Illustrator. `editable: false` writes outlines instead. PNG output is
unaffected either way.

Fonts, sizes, labels and annotations are configuration, not code: `style`
selects the size system, `style_overrides` adjusts individual rc values, and
`BasePlotterConfig` fields cover `title`, `xlabel`, `ylabel`, `xlim`, `ylim`,
`grid`, `figsize`, `dpi`, `filename`, `constrained_layout` and `annotations`.

```yaml
annotations:
  - text: "$\\gamma/\\kappa = 0.1$"
    x: 0.02
    y: 0.95
    coords: axes fraction   # or "data"
    fontsize: 9
    color: "#111827"
```

## Plotters

| Plotter | Draws | Typed input |
| --- | --- | --- |
| `time_series` | amplitude vs time | `time_series` / `field_amplitude` product |
| `phase_plane` | Re–Im or channel–channel plane | the same trajectory product |
| `power_spectrum` | one PSD curve | `spectral` / `power_spectral_density` (`axis`, `psd`) |
| `parameter_evolution` | a metric vs a scan coordinate | typed peak or trajectory product |
| `spectral_scan` | one panel per mode across a scan | the PSD product with its scan axis |
| `heatmap` | a 2-D scalar grid | any product with `x`, `y`, `value[row, column]` |
| `distribution` | per-realization sample densities | a product of samples per realization |

The single-point plotters (`time_series`, `phase_plane`, `power_spectrum`)
refuse a multi-point scan: map the input over points or select one point
explicitly instead of relying on an implicit average.

### PSD across a scan

`spectral_scan` draws one panel per mode, all sharing the frequency axis, with
one curve per selected scan point and a legend built from the scan coordinate.
The scan coordinate is always named:

```yaml
visualizer:
  spectral_scan:
    plots:
      - scan_parameter: omega        # coordinate stored on the PSD product
        channels: [0, 1, 2]          # one panel each
        scan_indices: [0, 10, 20]    # or scan_coordinates, see below
        log_scale: true
        label_template: "{name}={value:.6g}"
```

Selecting by value needs an explicit bound: `scan_coordinates: [0.12, 0.24]`
**together with** `scan_tolerance: 0.01`. There is no default tolerance, because a
nearest match without one would silently draw a different scan point than the
one requested; a value further than the tolerance from every stored point is an
error. A curve keeps the colour of its scan position, so drawing a subset never
repaints the remaining curves.

### Scalar heatmap

`heatmap` renders a `pcolormesh` plus a colorbar. `x` is the 1-D row coordinate,
`y` the 1-D column coordinate and `value[row, column]` the grid, so the declared
orientation and the rendered figure agree:

```yaml
visualizer:
  heatmap:
    plots:
      - x_variable: gamma_b
        y_variable: gamma_c
        value_variable: local_information
        valid_variable: physical       # optional boolean mask, False = blank
        cmap: flow                     # "scan", "flow" or any Matplotlib map
        norm: linear                   # or log (needs strictly positive data)
        colorbar_label: Information
        contours:
          - variable: continuation_allowed
            levels: [0.5]
        series:
          - x: [0.8]
            y: [0.957]
            marker: "*"
            linestyle: null
```

Non-finite cells and masked cells are left blank, never drawn as zero. A grid
whose shape does not match the coordinates, no finite valid value, or a log norm
over non-positive data are errors, and an ambiguous product selection asks for
`product`.

### Sample distributions

`distribution` draws the distribution of samples that were already computed
upstream. `value_variable` holds `value[realization]` or
`value[scan, realization]`; selecting several scan points draws one curve per
point and never pools them.

```yaml
visualizer:
  distribution:
    plots:
      - value_variable: estimate
        scan_parameter: omega
        scan_indices: [0, 10]        # or scan_coordinates + scan_tolerance
        center: reference_mean       # first series' mean becomes zero
        scale: -1                    # reference - value
        bin_width: 0.0001            # or "bins: 40"
        labels: [reference, perturbed]
        show_means: true
        show_fwhm: true
        shift_label: signal shift    # needs show_means and two series
        width_label: noise width     # needs show_fwhm
```

`center` selects the origin and `scale` the sign: `center: none` draws
`scale * value`, `center: reference_mean` draws `scale * (value - reference)`
with the first selected series' mean as the reference. Densities are normalised
as `counts / (N * bin_width)` with `N` the full finite sample count, so a window
clipped by `xlim` holds less than unit area while the curve keeps the complete
distribution's height. The half-maximum width needs the peak fully inside the
range and raises otherwise rather than extrapolating from the array edge.
