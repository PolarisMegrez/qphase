"""Bifurcation detection and optional critical-point refinement."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from scipy.optimize import brentq

from qphase_cam.core.jacobian import JacobianResolver

from .base import CAMPostprocessor, CAMPostprocessorConfig


class BifurcationConfig(CAMPostprocessorConfig):
    refine: bool = True
    tolerance: float = Field(1e-10, gt=0.0)
    max_iterations: int = Field(100, ge=1)


class Bifurcation(CAMPostprocessor):
    name: ClassVar[str] = "bifurcation"
    description: ClassVar[str] = "Locate Jacobian stability crossings"
    config_schema: ClassVar[type[BifurcationConfig]] = BifurcationConfig

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        if len(result.axes) != 1:
            raise ValueError("bifurcation requires exactly one continuation axis")
        if not result.meta.get("continuation"):
            raise ValueError(
                "bifurcation refinement requires continuation output; "
                "unordered solution slots are not branches"
            )
        parameter, values = next(iter(result.axes.items()))
        eigenvalues = result.postprocess.get("jacobian_eigenvalues")
        if eigenvalues is None:
            resolver = JacobianResolver()
            spectra = []
            for index, state in enumerate(result.states[:, 0]):
                params = dict(result.params)
                params[parameter] = float(values[index])
                jacobian = resolver.resolve(
                    model, backend.asarray(state), params, backend
                )
                spectra.append(np.linalg.eigvals(np.asarray(jacobian)))
            eigenvalues = np.asarray(spectra)[:, None, :]
        critical = np.max(np.real(eigenvalues[:, 0]), axis=-1)
        brackets = []
        refined = []
        for index in range(len(values) - 1):
            if np.isfinite(critical[index : index + 2]).all() and (
                np.sign(critical[index]) != np.sign(critical[index + 1])
            ):
                bracket = (float(values[index]), float(values[index + 1]))
                brackets.append(bracket)
                if self.config.refine:
                    refined.append(
                        self._refine(
                            model,
                            result,
                            backend,
                            parameter,
                            bracket,
                            index,
                        )
                    )
        return {
            "bifurcation_brackets": np.asarray(brackets),
            "bifurcation_values": np.asarray(refined),
        }

    def _refine(
        self,
        model: Any,
        result: Any,
        backend: Any,
        parameter: str,
        bracket: tuple[float, float],
        index: int,
    ) -> float:
        from qphase_cam.solver.common import solve_single_state

        guess = np.asarray(result.states[index, 0])
        resolver = JacobianResolver()

        def critical(value: float) -> float:
            params = dict(result.params)
            params[parameter] = value
            solution = solve_single_state(
                model,
                params,
                guess,
                method="cholesky",
                tolerance=self.config.tolerance,
            )
            if not solution.success:
                raise ValueError("steady-state refinement failed")
            jacobian = resolver.resolve(
                model, backend.asarray(solution.state), params, backend
            )
            return float(np.max(np.real(np.linalg.eigvals(np.asarray(jacobian)))))

        return float(
            brentq(
                critical,
                bracket[0],
                bracket[1],
                xtol=self.config.tolerance,
                maxiter=self.config.max_iterations,
            )
        )
