"""qphase_sde: Entry Point
----------------------

Entry point for qphase_sde resource package.
"""

from typing import Any

from .core.engine import Engine, EngineConfig
from .core.errors import QPSError
from .core.result import SDEResult


def main(
    config: dict[str, Any],
    plugins: dict[str, Any],
    data: Any | None = None,
) -> SDEResult:
    """Execute SDE simulation package.

    Parameters
    ----------
    config : dict[str, Any]
        Job parameters (time, n_traj, etc.).
    plugins : dict[str, Any]
        Instantiated plugins (backend, integrator, model, noise_model).
    data : Any | None
        Input data (unused for SDE currently).

    Returns
    -------
    SDEResult
        Simulation result.

    """
    # 1. Extract Core Components from Plugins
    backend = plugins.get("backend")
    integrator = plugins.get("integrator")
    model = plugins.get("model")
    # noise_model plugin is deprecated/removed; engine handles noise internally
    # noise_model = plugins.get("noise_model")

    if model is None:
        # Fallback: Check if model is defined in config (legacy/simple mode)
        # But for v0.2, we prefer plugins.
        raise QPSError("SDE simulation requires a 'model' plugin.")

    # 2. Initialize Engine
    # Engine handles dependency injection of backend/integrator
    engine = Engine(backend=backend, integrator=integrator)

    # 3. Prepare Simulation Parameters
    # Extract time config
    time_cfg = config.get("time")
    if not time_cfg:
        raise QPSError("Missing 'time' configuration in job params.")

    # Extract initial conditions
    ic = config.get("ic")
    if ic is None:
        raise QPSError("Missing 'ic' (initial conditions) in job params.")

    n_traj = config.get("n_traj", 1)

    # Extract other run options
    seed = config.get("seed")
    # save_every = config.get("save_every", 1)  # Currently unused

    # 4. Run Simulation
    try:
        trajectory = engine.run_sde(
            model=model,
            ic=ic,
            time=time_cfg,
            n_traj=n_traj,
            # noise_spec=noise_model, # Deprecated
            seed=seed,
        )
    except Exception as e:
        raise RuntimeError(f"SDE Simulation failed: {e}") from e

    # 5. Wrap and Return Result
    result = SDEResult(
        trajectory=trajectory,
        meta={
            "config": config,
            "model_name": getattr(model, "name", str(model)),
            "n_traj": n_traj,
        },
        kind="trajectory",
    )

    return result


main.config_schema = EngineConfig  # type: ignore[attr-defined]
