from pydantic import BaseModel, Field
from qphase.core.protocols import ResultBase


class DummyConfig(BaseModel):
    param: float = Field(1.0, json_schema_extra={"scanable": True})
    description: str = "A dummy plugin"


class DummyPlugin:
    config_schema = DummyConfig

    def __init__(self, config: DummyConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def run(self, data=None):
        return ResultBase(
            data={"status": "ok", "param": self.config.param}, metadata={"dummy": True}
        )
