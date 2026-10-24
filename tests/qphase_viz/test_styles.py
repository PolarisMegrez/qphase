"""Style system tests: named rc tables, editable output, annotations, scoping."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pytest
from qphase_viz import VizEngine, VizEngineConfig
from qphase_viz.plotters.evolution import TimeSeriesPlotter
from qphase_viz.plotters.parameter import ParameterEvolutionPlotter
from qphase_viz.plotters.phase import PhasePlanePlotter
from qphase_viz.plotters.spectrum import PowerSpectrumPlotter
from qphase_viz.styles import (
    DEFAULT_STYLE,
    PALETTE,
    STYLE_NAMES,
    RenderOptions,
    allan_scan_colormap,
    cam_flow_colormap,
    rc_params,
)

from tests.qphase_viz.typed_fixtures import (
    VizTestBundle,
    peak_scan_bundle,
    spectrum_bundle,
    trajectory_bundle,
)

#: Values the two STIX styles share: axis geometry and export resolution.
_SHARED = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
}

_PUBLICATION = {
    **_SHARED,
    "font.size": 12.0,
    "axes.labelsize": 14.4,
    "axes.titlesize": 11.2,
    "legend.fontsize": 12.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
}

_SM = {
    **_SHARED,
    "font.size": 10.5,
    "axes.labelsize": 13.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
}

_SANS = {
    **_SHARED,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "axes.labelsize": 18.0,
    "xtick.labelsize": 15.0,
    "ytick.labelsize": 15.0,
}


def _plotter_for(name: str):
    """Return ``(plotter, bundle)`` rendering one annotated figure."""
    annotation = {"text": f"note-{name}", "x": 0.05, "y": 0.95}
    if name == "time_series":
        return (
            TimeSeriesPlotter(
                plots=[{"channels": [0], "annotations": [annotation]}]
            ),
            trajectory_bundle(),
        )
    if name == "phase_plane":
        return (
            PhasePlanePlotter(
                plots=[{"channel_x": 0, "annotations": [annotation]}]
            ),
            trajectory_bundle(),
        )
    if name == "power_spectrum":
        bundle = VizTestBundle({"psd": spectrum_bundle().products["psd"]})
        return (
            PowerSpectrumPlotter(
                plots=[{"channels": [0], "annotations": [annotation]}]
            ),
            bundle,
        )
    if name == "parameter_evolution":
        return (
            ParameterEvolutionPlotter(
                plots=[
                    {
                        "parameter": "omega",
                        "metric": "psd_peak_freq",
                        "annotations": [annotation],
                    }
                ]
            ),
            peak_scan_bundle(),
        )
    raise AssertionError(name)


@pytest.mark.parametrize(
    ("style", "expected"),
    [("publication", _PUBLICATION), ("sm", _SM), ("sans", _SANS)],
)
def test_style_rc_values_are_exact(style, expected):
    """Each named style reproduces the publication/SM/sans values verbatim."""
    params = rc_params(style)
    for key, value in expected.items():
        assert params[key] == value, key


def test_only_the_declared_styles_exist():
    assert STYLE_NAMES == ("publication", "sm", "sans")
    assert DEFAULT_STYLE == "publication"
    with pytest.raises(ValueError, match="unknown figure style"):
        rc_params("nonexistent")  # type: ignore[arg-type]


def test_rc_params_returns_a_copy():
    params = rc_params("publication")
    params["font.size"] = 99.0
    assert rc_params("publication")["font.size"] == 12.0


def test_default_options_are_publication_and_editable():
    options = RenderOptions()
    assert options.style == "publication"
    assert options.editable is True
    params = options.rc_params()
    assert params["font.family"] == "STIXGeneral"
    assert params["axes.labelsize"] == 14.4
    assert params["svg.fonttype"] == "none"
    assert params["pdf.fonttype"] == 42


def test_editable_text_can_be_disabled_and_overridden():
    outlined = RenderOptions(editable=False).rc_params()
    assert outlined["svg.fonttype"] == "path"
    assert outlined["pdf.fonttype"] == 3

    overridden = RenderOptions(overrides={"pdf.fonttype": 3}).rc_params()
    assert overridden["pdf.fonttype"] == 3
    assert overridden["svg.fonttype"] == "none"


def test_merge_priority_is_figure_over_engine_over_default():
    engine = RenderOptions(
        style="sm", editable=False, overrides={"axes.labelsize": 20.0}
    )

    # Nothing set on the figure: the engine's choice survives.
    inherited = engine.merged(style=None, editable=None, overrides=None)
    assert inherited.style == "sm"
    assert inherited.editable is False
    assert inherited.rc_params()["axes.labelsize"] == 20.0

    # The figure wins on every axis, and its overrides layer on top.
    figure = engine.merged(
        style="publication",
        editable=True,
        overrides={"axes.labelsize": 30.0},
    )
    params = figure.rc_params()
    assert figure.style == "publication"
    assert params["axes.labelsize"] == 30.0
    assert params["font.family"] == "STIXGeneral"
    assert params["svg.fonttype"] == "none"


def test_engine_style_reaches_plotter_and_spec_wins(tmp_path):
    engine = VizEngine(
        config=VizEngineConfig(
            output_dir=str(tmp_path), format="svg", style="sm"
        ),
        plugins={
            "visualizer": TimeSeriesPlotter(
                plots=[
                    {"channels": [0], "filename": "engine_default"},
                    {
                        "channels": [0],
                        "filename": "figure_override",
                        "style": "publication",
                    },
                ]
            )
        },
    )

    result = engine.run(trajectory_bundle())

    sizes = {path.stem: path.read_text(encoding="utf-8") for path in result.data}
    assert "font-size: 13.5px" in sizes["engine_default"]
    assert "font-size: 14.4px" in sizes["figure_override"]


def test_spec_style_overrides_beat_engine_overrides(tmp_path):
    engine = VizEngine(
        config=VizEngineConfig(
            output_dir=str(tmp_path),
            format="svg",
            style_overrides={"axes.labelsize": 20.0},
        ),
        plugins={
            "visualizer": TimeSeriesPlotter(
                plots=[
                    {
                        "channels": [0],
                        "xlabel": "Labelled",
                        "style_overrides": {"axes.labelsize": 30.0},
                    }
                ]
            )
        },
    )

    results = engine.run(trajectory_bundle())
    (path,) = results.data
    svg = path.read_text(encoding="utf-8")
    assert "font-size: 30px" in svg
    assert "font-size: 20px" not in svg


def test_direct_plotter_call_renders_publication_by_default(tmp_path):
    plotter = TimeSeriesPlotter(
        plots=[{"channels": [0], "xlabel": "Labelled"}]
    )

    (path,) = plotter.plot(trajectory_bundle(), tmp_path, "svg")

    svg = path.read_text(encoding="utf-8")
    assert "font-family: 'STIXGeneral'" in svg
    assert "font-size: 14.4px" in svg
    assert ">Labelled<" in svg


def test_editable_svg_keeps_text_and_outlined_svg_does_not(tmp_path):
    plotter = TimeSeriesPlotter(
        plots=[{"channels": [0], "xlabel": "Labelled"}]
    )
    outlined_plotter = TimeSeriesPlotter(
        plots=[{"channels": [0], "xlabel": "Labelled", "filename": "outlined"}]
    )

    (editable,) = plotter.plot(trajectory_bundle(), tmp_path, "svg")
    (outlined,) = outlined_plotter.plot(
        trajectory_bundle(), tmp_path, "svg", RenderOptions(editable=False)
    )

    editable_text = editable.read_text(encoding="utf-8")
    assert "<text" in editable_text
    assert ">Labelled<" in editable_text
    assert outlined != editable
    assert "<text" not in outlined.read_text(encoding="utf-8")


def test_editable_pdf_uses_true_type_fonts(tmp_path):
    plotter = TimeSeriesPlotter(
        plots=[{"channels": [0], "xlabel": "Labelled"}]
    )

    (path,) = plotter.plot(trajectory_bundle(), tmp_path, "pdf")

    assert path.suffix == ".pdf"
    payload = path.read_bytes()
    # fonttype 42 embeds a TrueType program; fonttype 3 (outlined) embeds none.
    assert b"/FontFile2" in payload
    assert b"/Type0" in payload

    outlined_plotter = TimeSeriesPlotter(
        plots=[{"channels": [0], "filename": "outlined"}]
    )
    (outlined,) = outlined_plotter.plot(
        trajectory_bundle(), tmp_path, "pdf", RenderOptions(editable=False)
    )
    assert b"/FontFile2" not in outlined.read_bytes()


@pytest.mark.parametrize(
    "plotter_name",
    ["time_series", "phase_plane", "power_spectrum", "parameter_evolution"],
)
def test_annotations_reach_every_plotter(tmp_path, plotter_name):
    plotter, bundle = _plotter_for(plotter_name)

    (path,) = plotter.plot(bundle, tmp_path, "svg")

    svg = path.read_text(encoding="utf-8")
    assert f">note-{plotter_name}<" in svg


def test_annotation_attributes_are_applied(tmp_path):
    plotter = TimeSeriesPlotter(
        plots=[
            {
                "channels": [0],
                "annotations": [
                    {
                        "text": "sized",
                        "x": 0.5,
                        "y": 0.5,
                        "coords": "data",
                        "fontsize": 6.0,
                        "color": PALETTE["sde"],
                        "ha": "center",
                        "va": "center",
                        "rotation": 15.0,
                    }
                ],
            }
        ]
    )

    (path,) = plotter.plot(trajectory_bundle(), tmp_path, "svg")
    svg = path.read_text(encoding="utf-8")
    assert ">sized<" in svg
    assert "font-size: 6px" in svg
    assert PALETTE["sde"].lower() in svg.lower()
    assert "rotate(-15)" in svg


def test_rcparams_are_not_leaked_by_plotter_or_engine(tmp_path):
    before = dict(mpl.rcParams)

    plotter = TimeSeriesPlotter(
        plots=[{"channels": [0], "style": "sans", "editable": False}]
    )
    plotter.plot(trajectory_bundle(), tmp_path, "png")
    assert dict(mpl.rcParams) == before

    engine = VizEngine(
        config=VizEngineConfig(
            output_dir=str(tmp_path / "engine"),
            format="png",
            style="sm",
            style_overrides={"axes.labelsize": 22.0},
        ),
        plugins={"visualizer": plotter},
    )
    engine.run(trajectory_bundle())
    assert dict(mpl.rcParams) == before


def test_palette_and_colormaps_are_the_single_source():
    assert PALETTE == {
        "ink": "#111827",
        "cam": "#0072B2",
        "sde": "#D55E00",
        "accent": "#E69F00",
        "green": "#009E73",
        "magenta": "#CC79A7",
        "purple": "#6F3CC3",
        "proxy": "#66717A",
        "noncritical": "#7A8288",
        "white_fm": "#DCEEFF",
        "flow_zero": "#FAFAF7",
    }
    scan = allan_scan_colormap()
    assert scan.name == "qphase_scan"
    # Trimmed plasma: the first stop is not the raw plasma endpoint.
    assert tuple(scan(0.0)) != tuple(mpl.colormaps["plasma"](0.0))
    flow = cam_flow_colormap()
    assert flow.name == "qphase_cam_flow"
    assert flow(0.0) == mpl.colors.to_rgba(PALETTE["cam"])
    assert flow(1.0) == mpl.colors.to_rgba(PALETTE["sde"])


def test_annotation_defaults_inherit_the_style(tmp_path):
    plotter = TimeSeriesPlotter(
        plots=[{"channels": [0], "annotations": [{"text": "inherit", "x": 0, "y": 1}]}]
    )

    (path,) = plotter.plot(trajectory_bundle(), tmp_path, "svg")

    svg = path.read_text(encoding="utf-8")
    assert ">inherit<" in svg
    # No explicit size or colour: the annotation is drawn at font.size.
    assert svg.count("font-size: 12px") >= 1


def test_exported_public_api_is_stable():
    import qphase_viz

    for name in (
        "AnnotationSpec",
        "DEFAULT_STYLE",
        "PALETTE",
        "RenderOptions",
        "STYLE_NAMES",
        "StyleName",
        "allan_scan_colormap",
        "cam_flow_colormap",
        "rc_params",
    ):
        assert name in qphase_viz.__all__
        assert getattr(qphase_viz, name) is not None


def test_exported_filenames_keep_their_semantics(tmp_path):
    plotter = TimeSeriesPlotter(
        plots=[
            {"channels": [0], "filename": "chosen"},
            {"channels": [0], "transform": "imag"},
        ]
    )

    files = plotter.plot(trajectory_bundle(), tmp_path, "png")

    assert [path.name for path in files] == ["chosen.png", "time_series_imag.png"]
    assert all(path.parent == Path(tmp_path) for path in files)


def test_figure_is_closed_when_rendering_raises(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import qphase_viz.plotters.evolution as evolution

    def explode(*args, **kwargs):
        raise RuntimeError("rendering failed")

    monkeypatch.setattr(evolution, "draw_annotations", explode)
    plotter = TimeSeriesPlotter(plots=[{"channels": [0]}])

    with pytest.raises(RuntimeError, match="rendering failed"):
        plotter.plot(trajectory_bundle(), tmp_path, "png")

    assert plt.get_fignums() == []
    assert not list(tmp_path.glob("*.png"))
