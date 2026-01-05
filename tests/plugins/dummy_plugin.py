from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DummyConfig(BaseModel):
    param: float = Field(1.0, json_schema_extra={"scanable": True})
    description: str = "A dummy plugin"


class DummyResult(BaseModel):
    data: Any
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> Any:
        return self.metadata.get("label")

    def save(self, path: str | Path) -> None:
        """Save the dummy result to output directory."""
        import json

        output_path = Path(path)
        output_file = output_path / "result.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        result_data = {
            "data": self.data,
            "metadata": self.metadata,
        }

        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)


class DummyPlugin:
    config_schema = DummyConfig

    def __init__(self, config: DummyConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def run(self, data=None):
        return DummyResult(
            data={"status": "ok", "param": self.config.param}, metadata={"dummy": True}
        )
