from __future__ import annotations

"""Pydantic schemas for the triad configuration (model/profile/run).

Key highlights:
- model.ic accepts either a flat complex-string list or a nested list; it is
    normalized to a nested list and validated for complex parsing.
- run.viz.phase is a list of per-figure specifications with required kind/modes
    and an optional t_range.
- profile.viz.phase carries per-kind matplotlib kwargs to style the plots.
"""

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field, model_validator
try:
    from pydantic import ConfigDict  # pydantic v2
except Exception:
    ConfigDict = None  # type: ignore
from QPhaseSDE.core.errors import ConfigError


class NoiseConfig(BaseModel):
    """Noise configuration for the model layer (A-class)."""
    kind: Literal["independent", "correlated"]
    covariance: Optional[List[List[float]]] = None

    @model_validator(mode="after")
    def _check_cov(self):
        if self.kind == "correlated" and self.covariance is None:
            raise ConfigError("[510] noise.covariance is required for kind='correlated'")
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
                    raise ConfigError(f"[511] ic contains non-complex-parsable value: {s!r}") from e
        # Ensure non-empty
        if not self.ic or not self.ic[0]:
            raise ConfigError("[512] ic must contain at least one vector of complex values")
        return self


class SaveConfig(BaseModel):
    """Save parameters: root directory and optional save decimation."""
    root: str = Field(default="runs")
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
    # B-class: defaults allowed for backend and solver
    backend: Literal["numpy", "numba"] = Field(default="numpy")
    solver: Literal["euler", "milstein"] = Field(default="euler")
    save: SaveConfig
    # Preferred key: visualization, with nested phase_portrait kinds
    visualization: Optional[Dict[str, Dict]] = None  # e.g., { phase_portrait: { Re_Im: {...}, abs_abs: {...} } }
    # Back-compat alias
    viz: Optional[Dict[str, Dict]] = None

    @model_validator(mode="after")
    def _migrate_viz(self):
        if self.visualization is None and self.viz is not None:
            # migrate old 'viz' to 'visualization'
            self.visualization = self.viz
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

    @model_validator(mode="after")
    def _check_modes(self):
        # normalize kind for internal use
        if self.kind in ("Re_Im", "re_im"):
            self.kind = "re_im"
        elif self.kind in ("Abs_Abs", "abs_abs"):
            self.kind = "abs_abs"
        else:
            raise ConfigError("[513] Unsupported kind for phase portrait")

        if self.kind == "re_im" and len(self.modes) != 1:
            raise ConfigError("[514] re_im requires exactly one mode index in modes")
        if self.kind == "abs_abs" and len(self.modes) != 2:
            raise ConfigError("[515] abs_abs requires exactly two mode indices in modes")
        if self.t_range is not None:
            if len(self.t_range) != 2 or float(self.t_range[1]) <= float(self.t_range[0]):
                raise ConfigError("[516] t_range must be [t_start, t_end] with t_end > t_start")
        return self


class VizPSDConfig(BaseModel):
    kind: Literal["complex", "modular"]
    modes: List[int]
    # convention now belongs to profile.visualization.psd; removed here
    xlim: Optional[List[float]] = None
    t_range: Optional[List[float]] = None
    # placeholders for future features
    welch: Optional[dict] = None
    zeropad: Optional[int] = None

    @model_validator(mode="after")
    def _check(self):
        if not self.modes:
            raise ConfigError("[517] psd.modes must contain at least one index")
        if self.t_range is not None:
            if len(self.t_range) != 2 or float(self.t_range[1]) <= float(self.t_range[0]):
                raise ConfigError("[518] t_range must be [t_start, t_end] with t_end > t_start")
        # validate xlim size if provided
        if self.xlim is not None:
            if len(self.xlim) != 2:
                raise ConfigError("[519] xlim must be [xmin, xmax]")
        return self


class RunVizConfig(BaseModel):
    # Preferred: visualization.phase_portrait
    phase_portrait: Optional[List[VizPhaseConfig]] = None
    # Back-compat: viz.phase
    phase: Optional[List[VizPhaseConfig]] = None
    # PSD figures
    psd: Optional[List[VizPSDConfig]] = None


class RunConfig(BaseModel):
    # C-class: dt/steps/n_traj/seed required per instruction.
    time: TimeConfig
    trajectories: TrajConfig
    visualization: Optional[RunVizConfig] = None
    # Back-compat alias
    viz: Optional[RunVizConfig] = None

    @model_validator(mode="after")
    def _migrate_viz(self):
        if self.visualization is None and self.viz is not None:
            self.visualization = self.viz
        # migrate inner key from phase to phase_portrait if needed
        if self.visualization is not None:
            rv = self.visualization
            if getattr(rv, 'phase_portrait', None) is None and getattr(rv, 'phase', None) is not None:
                rv.phase_portrait = rv.phase
        return self


class TriadConfig(BaseModel):
    """Root triad container with model, profile, and run sections."""
    model: ModelConfig
    profile: ProfileConfig
    run: RunConfig
