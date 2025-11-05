from __future__ import annotations

"""Pydantic schemas for the triad configuration (model/profile/run).

Key highlights:
- model.ic accepts either a flat complex-string list or a nested list; it is
    normalized to a nested list and validated for complex parsing.
- run.viz.phase is a list of per-figure specifications with required kind/modes
    and an optional t_range.
- profile.viz.phase carries per-kind matplotlib kwargs to style the plots.
"""

from typing import Dict, List, Literal, Optional, Any, Union
from pydantic import BaseModel, Field, model_validator
try:
    from pydantic import ConfigDict  # pydantic v2
except Exception:
    ConfigDict = None  # type: ignore
from QPhaseSDE.core.errors import QPSConfigError


class NoiseConfig(BaseModel):
    """Noise configuration for the model layer (A-class)."""
    kind: Literal["independent", "correlated"]
    covariance: Optional[List[List[float]]] = None

    @model_validator(mode="after")
    def _check_cov(self):
        if self.kind == "correlated" and self.covariance is None:
            raise QPSConfigError("[510] noise.covariance is required for kind='correlated'")
        return self


class ModelConfig(BaseModel):
    """Model layer (A-class) with module/function and parameters."""
    # A-class: no defaults; all required
    module: str
    function: str
    params: Dict
    # Use strings for complex initial conditions; accept either
    # - flat list: ["7.0+0.0j", "0.0-7.0j"]
    # - nested list: [["7.0+0.0j", "0.0-7.0j"], ["7.0+0.0j", "0.0-7.0j"]]
    # We'll normalize to nested list in a validator and verify parseability to complex.
    ic: List[List[str]]
    noise: NoiseConfig
    comment: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_ic(cls, values: Dict[str, Any]):
        ic = values.get("ic")
        if ic is None:
            return values
        # If flat list, wrap it
        if isinstance(ic, list) and (not ic or not isinstance(ic[0], list)):
            values["ic"] = [ic]
        return values

    @model_validator(mode="after")
    def _validate_ic_complex(self):
        # Check all entries are parseable to complex
        for vec in self.ic:
            for s in vec:
                try:
                    complex(s)
                except Exception as e:
                    raise QPSConfigError(f"[511] ic contains non-complex-parsable value: {s!r}") from e
        # Ensure non-empty
        if not self.ic or not self.ic[0]:
            raise QPSConfigError("[512] ic must contain at least one vector of complex values")
        return self


class SaveConfig(BaseModel):
    """Save parameters: root directory and optional save decimation."""
    root: str
    save_every: Optional[int] = None
    # Required save toggles
    save_timeseries: bool
    save_psd_complex: bool
    save_psd_modular: bool


class ProfileVizPhaseConfig(BaseModel):
    # Matplotlib kwargs per kind
    re_im: Optional[Dict] = None
    abs_abs: Optional[Dict] = None


class ProfileConfig(BaseModel):
    # B-class: backend/integrator selection and IO + style knobs
    backend: Literal["numpy", "numba", "torch", "cupy"]
    solver: Literal["euler", "milstein"]
    save: SaveConfig
    # Style: matplotlib kwargs and PSD conventions moved here (centralized)
    visualizer: Optional[Dict[str, Dict]] = None  # e.g., { phase_portrait: { Re_Im: {...}, abs_abs: {...} }, psd: { convention: 'symmetric', x_scale: 'log', y_scale: 'log', xlim: [...] } }
    # Back-compat alias
    viz: Optional[Dict[str, Dict]] = None

    @model_validator(mode="after")
    def _migrate_viz(self):
        if self.visualizer is None and self.viz is not None:
            # migrate old 'viz' to 'visualizer'
            self.visualizer = self.viz
        return self


class TimeConfig(BaseModel):
    t0: float = 0.0
    dt: float  # required (no default)
    steps: int  # required (no default)


class TrajConfig(BaseModel):
    # C-class trajectories
    n_traj: int  # required
    # New seed strategy (no legacy seed)
    seed_file: Optional[str] = None
    master_seed: Optional[int] = None
    # RNG stream strategy for noise sampling
    rng_stream: Optional[Literal["per_trajectory", "batched"]] = None

    # Forbid unknown fields (e.g., legacy 'seed')
    if ConfigDict is not None:
        model_config = ConfigDict(extra='forbid')  # type: ignore
    else:
        class Config:
            extra = 'forbid'


class VizPhaseConfig(BaseModel):
    # One spec per figure
    # Accept Re_Im (preferred) or re_im; and abs_abs (case sensitive by default, accept Abs_Abs as special-case)
    kind: Literal["re_im", "Re_Im", "abs_abs", "Abs_Abs"]
    modes: List[int]
    t_range: Optional[List[float]] = None
    # New: subsampling factor for plotting only (does not affect saved data)
    plot_every: Optional[int] = None

    @model_validator(mode="after")
    def _check_modes(self):
        # normalize kind for internal use
        if self.kind in ("Re_Im", "re_im"):
            self.kind = "re_im"
        elif self.kind in ("Abs_Abs", "abs_abs"):
            self.kind = "abs_abs"
        else:
            raise QPSConfigError("[513] Unsupported kind for phase portrait")

        if self.kind == "re_im" and len(self.modes) != 1:
            raise QPSConfigError("[514] re_im requires exactly one mode index in modes")
        if self.kind == "abs_abs" and len(self.modes) != 2:
            raise QPSConfigError("[515] abs_abs requires exactly two mode indices in modes")
        if self.t_range is not None:
            if len(self.t_range) != 2 or float(self.t_range[1]) <= float(self.t_range[0]):
                raise QPSConfigError("[516] t_range must be [t_start, t_end] with t_end > t_start")
        if self.plot_every is not None:
            try:
                pe = int(self.plot_every)
            except Exception:
                raise QPSConfigError("[520] plot_every must be an integer >= 1")
            if pe < 1:
                raise QPSConfigError("[520] plot_every must be an integer >= 1")
        return self


class VizPSDConfig(BaseModel):
    kind: Literal["complex", "modular"]
    modes: List[int]
    # convention now belongs to profile.visualizer.psd; removed here
    xlim: Optional[List[float]] = None
    t_range: Optional[List[float]] = None
    # placeholders for future features
    welch: Optional[dict] = None
    zeropad: Optional[int] = None

    @model_validator(mode="after")
    def _check(self):
        if not self.modes:
            raise QPSConfigError("[517] psd.modes must contain at least one index")
        if self.t_range is not None:
            if len(self.t_range) != 2 or float(self.t_range[1]) <= float(self.t_range[0]):
                raise QPSConfigError("[518] t_range must be [t_start, t_end] with t_end > t_start")
        # validate xlim size if provided
        if self.xlim is not None:
            if len(self.xlim) != 2:
                raise QPSConfigError("[519] xlim must be [xmin, xmax]")
        return self


class RunVizConfig(BaseModel):
    # Preferred: visualizer.phase_portrait
    phase_portrait: Optional[List[VizPhaseConfig]] = None
    # Back-compat: viz.phase
    phase: Optional[List[VizPhaseConfig]] = None
    # PSD figures
    psd: Optional[List[VizPSDConfig]] = None


class RunConfig(BaseModel):
    # C-class (runtime numeric knobs only): time + trajectories
    time: TimeConfig
    trajectories: TrajConfig

    @model_validator(mode="after")
    def _migrate_viz(self):
        return self


class TriadConfig(BaseModel):
    """Legacy triad config (model/profile/run). Used for migration to jobs-based config."""
    model: ModelConfig
    profile: ProfileConfig
    run: RunConfig

# New jobs-based schema

class JobSpec(BaseModel):
    # Unified figure spec list: either phase-portrait or PSD entries
    kind: Literal["re_im", "Re_Im", "abs_abs", "Abs_Abs", "complex", "modular"]
    modes: List[int]
    t_range: Optional[List[float]] = None
    # Note: xlim now belongs to profile.visualizer.psd (style)

class JobConfig(BaseModel):
    # Required model section
    module: str
    function: str
    params: Dict
    ic: List[List[str]]
    noise: NoiseConfig
    # Optional metadata and figure requests
    name: Optional[str] = None
    combinator: Optional[Literal["cartesian", "zipped"]] = None
    visualizer: Optional[List[JobSpec]] = None

class RootConfig(BaseModel):
    profile: ProfileConfig
    run: RunConfig
    jobs: List[JobConfig]
