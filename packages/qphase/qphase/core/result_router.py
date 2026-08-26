"""Input resolution and logical-result routing for the core scheduler."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import JobConfig
from .errors import ErrorCode, QPhaseConfigError, QPhaseIOError
from .execution import ExecutionContext
from .protocols import ResultProtocol

log = logging.getLogger("qphase")

__all__ = ["ResultRouter"]


class ResultRouter:
    """Route runtime results and resolve logical job inputs.

    The router owns data-flow policy. It does not choose storage layouts,
    instantiate plugins, or interpret resource-package results.
    """

    def __init__(self, system_config: Any) -> None:
        self.system_config = system_config

    def resolve_input(
        self,
        job: JobConfig,
        job_results: dict[str, ResultProtocol],
        *,
        source: str | None = None,
        manifest: Mapping[str, Any] | None = None,
        session_dir: Path | None = None,
    ) -> ResultProtocol | None:
        """Resolve an input from memory, the current Session, or a directory."""
        if job.input is None:
            return None
        input_source = source or job.input.from_

        if input_source in job_results:
            return job_results[input_source]

        if manifest is not None and input_source in manifest["jobs"]:
            job_entry = manifest["jobs"][input_source]
            if job_entry.get("status") == "completed" and session_dir is not None:
                output_rel_path = job_entry.get("output_dir")
                if output_rel_path:
                    job_dir = session_dir / output_rel_path
                    from .result_loader import load_result

                    log.info("Loading result for '%s' from disk...", input_source)
                    try:
                        result = load_result(input_source, job_dir)
                    except Exception as exc:
                        raise QPhaseIOError(
                            f"failed to load completed upstream result "
                            f"'{input_source}' from '{job_dir}': {exc}",
                            code=ErrorCode.ARTIFACT_IO,
                            context={"source": input_source, "path": str(job_dir)},
                        ) from exc
                    job_results[input_source] = result
                    return result

        input_path = Path(input_source)
        if input_path.exists():
            if input_path.is_dir():
                log.info(
                    "Job '%s' input '%s' is a directory; passing it to the "
                    "resource engine.",
                    job.name,
                    input_source,
                )
                from .aggregation import DirectoryInputResult

                return DirectoryInputResult(
                    path=input_path,
                    meta={"input_kind": "directory", "path": str(input_path)},
                )
            raise QPhaseConfigError(
                f"Job '{job.name}' specifies file input '{input_source}', "
                "but file loading is not currently supported.",
                code=ErrorCode.INPUT,
            )

        raise QPhaseConfigError(
            f"Job '{job.name}' input '{input_source}' not found. "
            "Expected a previous job name or a valid directory path.",
            code=ErrorCode.INPUT,
            hint="Run the upstream job first, or fix the 'input.from' reference.",
        )

    def route_output(
        self,
        job: JobConfig,
        output_result: ResultProtocol,
        job_results: dict[str, ResultProtocol],
        job_dir: Path,
        *,
        context: ExecutionContext | None = None,
    ) -> None:
        """Expose one result to downstream jobs and optionally persist it."""
        output_alias = job.output or job.name
        job_results[job.name] = output_result
        if job.output:
            job_results[job.output] = output_result

        should_save = False
        save_name = output_alias
        if job.save is not None:
            if isinstance(job.save, bool):
                should_save = job.save
            elif isinstance(job.save, str):
                should_save = True
                save_name = job.save
        else:
            should_save = self.system_config.auto_save_results

        if not should_save:
            return

        save_path = job_dir / save_name
        try:
            if context is not None:
                context.artifacts.save_result(output_result, save_name)
            else:
                output_result.save(save_path)
            log.debug("Job '%s' result saved to %s", job.name, save_path)
        except Exception as exc:
            raise QPhaseIOError(
                f"Failed to save job '{job.name}' output to '{save_path}': {exc}",
                code=ErrorCode.ARTIFACT_IO,
                hint="Check disk space and write permissions for the Job directory.",
            ) from exc
