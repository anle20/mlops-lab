"""Training configuration: a YAML file validated by pydantic.

Why: parameters live in one reviewable place instead of being scattered through
the code, and a typo or a nonsense value fails fast with a clear message.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    c: float = Field(default=1.0, gt=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(default=0.2, gt=0, lt=1)
    seed: int = 42


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output_dir: Path = Path("artifacts")


def load_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return TrainConfig.model_validate(raw)
