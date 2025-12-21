"""qphase: Configuration Snapshot
---------------------------------------------------------
Manages the creation and storage of configuration snapshots to ensure reproducibility.
It captures the complete state of a job execution, including the merged configuration,
system settings, plugin versions, and random seeds, serializing this metadata into
the run directory for future reference.

Public API
----------
``ConfigSnapshot`` : Pydantic model for serializable configuration snapshots
``SnapshotManager`` : Manager for creating, saving, loading, and comparing snapshots

Notes
-----
- Snapshots are saved as JSON in run directories for reproducibility
- Includes job config, system config, plugin versions, and timestamps

"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import JobConfig
from .system_config import SystemConfig


class ConfigSnapshot(BaseModel):
    """Configuration snapshot for reproducibility.

    This captures all necessary information to reproduce a job execution,
    including job configuration, system configuration, and metadata.
    """

    # Snapshot metadata
    snapshot_version: str = "1.0"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Job configuration
    job_name: str
    job_config: dict[str, Any]
    job_index: int

    # System configuration
    system_config: dict[str, Any] | None = None

    # Plugin configurations (validated)
    plugin_configs: dict[str, Any] = Field(default_factory=dict)

    # Engine configuration
    engine_config: dict[str, Any] = Field(default_factory=dict)

    # Job dependencies
    input_job: str | None = None
    output_job: str | None = None

    # Execution metadata
    run_id: str | None = None
    run_dir: Path | None = None

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        json_encoders = {
            Path: str,  # Convert Path objects to strings for JSON serialization
        }


class SnapshotManager:
    """Manager for configuration snapshots.

    This class handles the creation, saving, and loading of configuration snapshots
    to enable result reproducibility and debugging.
    """

    def __init__(self, snapshot_dir: Path):
        """Initialize snapshot manager.

        Parameters
        ----------
        snapshot_dir : Path
            Directory where snapshots should be stored

        """
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        job: JobConfig,
        job_index: int,
        system_config: SystemConfig | None,
        validated_plugins: dict[str, Any],
        engine_config: dict[str, Any],
        run_id: str | None = None,
        run_dir: Path | None = None,
        input_job: str | None = None,
        output_job: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConfigSnapshot:
        """Create a configuration snapshot.

        Parameters
        ----------
        job : JobConfig
            Job configuration
        job_index : int
            Index of the job in the job list
        system_config : SystemConfig | None
            System configuration
        validated_plugins : dict[str, Any]
            Validated plugin configurations
        engine_config : dict[str, Any]
            Engine configuration
        run_id : str | None, optional
            Run ID for this execution
        run_dir : Path | None, optional
            Run directory path
        input_job : str | None, optional
            Input job name
        output_job : str | None, optional
            Output job name
        metadata : dict[str, Any] | None, optional
            Additional metadata

        Returns
        -------
        ConfigSnapshot
            Created snapshot

        """
        # Convert job config to dict
        job_config_dict = job.model_dump(exclude_unset=True)

        # Convert system config to dict
        system_config_dict = None
        if system_config is not None:
            system_config_dict = system_config.model_dump()

        # Create snapshot
        snapshot = ConfigSnapshot(
            job_name=job.name,
            job_config=job_config_dict,
            job_index=job_index,
            system_config=system_config_dict,
            plugin_configs=validated_plugins,
            engine_config=engine_config,
            run_id=run_id,
            run_dir=run_dir,
            input_job=input_job,
            output_job=output_job,
            metadata=metadata or {},
        )

        return snapshot

    def save_snapshot(self, snapshot: ConfigSnapshot, run_dir: Path) -> Path:
        """Save snapshot to disk.

        Parameters
        ----------
        snapshot : ConfigSnapshot
            Snapshot to save
        run_dir : Path
            Run directory for this job execution

        Returns
        -------
        Path
            Path where snapshot was saved

        """
        # Ensure run_dir exists
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save snapshot as JSON
        snapshot_path = run_dir / "config_snapshot.json"

        # Use custom JSON encoder for Path objects
        def json_encoder(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(
                snapshot.model_dump(),
                f,
                indent=2,
                default=json_encoder,
                ensure_ascii=False,
            )

        return snapshot_path

    def load_snapshot(self, snapshot_path: Path) -> ConfigSnapshot:
        """Load snapshot from disk.

        Parameters
        ----------
        snapshot_path : Path
            Path to snapshot file

        Returns
        -------
        ConfigSnapshot
            Loaded snapshot

        Raises
        ------
        FileNotFoundError
            If snapshot file doesn't exist

        """
        with open(snapshot_path, encoding="utf-8") as f:
            snapshot_dict = json.load(f)

        return ConfigSnapshot(**snapshot_dict)

    def list_snapshots(self, job_name: str | None = None) -> list[ConfigSnapshot]:
        """List all snapshots.

        Parameters
        ----------
        job_name : str | None, optional
            Filter by job name

        Returns
        -------
        list[ConfigSnapshot]
            List of matching snapshots

        """
        snapshots = []

        # Search all snapshot files
        for snapshot_file in self.snapshot_dir.glob("**/config_snapshot.json"):
            try:
                snapshot = self.load_snapshot(snapshot_file)
                if job_name is None or snapshot.job_name == job_name:
                    snapshots.append(snapshot)
            except Exception:
                # Skip invalid snapshot files
                continue

        # Sort by creation time (newest first)
        snapshots.sort(key=lambda s: s.created_at, reverse=True)

        return snapshots

    def get_latest_snapshot(self, job_name: str) -> ConfigSnapshot | None:
        """Get the most recent snapshot for a job.

        Parameters
        ----------
        job_name : str
            Job name

        Returns
        -------
        ConfigSnapshot | None
            Latest snapshot or None if not found

        """
        snapshots = self.list_snapshots(job_name)
        return snapshots[0] if snapshots else None

    def compare_snapshots(
        self, snap1: ConfigSnapshot, snap2: ConfigSnapshot
    ) -> dict[str, Any]:
        """Compare two snapshots.

        Parameters
        ----------
        snap1 : ConfigSnapshot
            First snapshot
        snap2 : ConfigSnapshot
            Second snapshot

        Returns
        -------
        dict[str, Any]
            Comparison results

        """
        comparison = {
            "job_name": {
                "snap1": snap1.job_name,
                "snap2": snap2.job_name,
                "identical": snap1.job_name == snap2.job_name,
            },
            "system_config": {
                "identical": snap1.system_config == snap2.system_config,
            },
            "plugin_configs": {
                "identical": snap1.plugin_configs == snap2.plugin_configs,
            },
            "engine_config": {
                "identical": snap1.engine_config == snap2.engine_config,
            },
            "created_at": {
                "snap1": snap1.created_at,
                "snap2": snap2.created_at,
            },
        }

        return comparison

    def export_snapshot(
        self,
        snapshot: ConfigSnapshot,
        export_path: Path,
        include_result: bool = False,
    ) -> Path:
        """Export snapshot to a standalone file.

        Parameters
        ----------
        snapshot : ConfigSnapshot
            Snapshot to export
        export_path : Path
            Where to save exported snapshot
        include_result : bool, optional
            If True, also save the result data (if available)

        Returns
        -------
        Path
            Path where snapshot was exported

        """
        export_data = snapshot.model_dump()

        # Optionally include result data
        if include_result and snapshot.run_dir is not None:
            result_path = Path(snapshot.run_dir) / "result.json"
            if result_path.exists():
                with open(result_path, encoding="utf-8") as f:
                    result_data = json.load(f)
                export_data["result_data"] = result_data

        # Save to file
        def json_encoder(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(
                export_data, f, indent=2, default=json_encoder, ensure_ascii=False
            )

        return export_path
