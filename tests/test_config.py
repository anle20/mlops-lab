from pathlib import Path

import pytest
from pydantic import ValidationError

from mlops_lab.config import TrainConfig, load_config


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_repo_config_is_valid():
    cfg = load_config("configs/train.yaml")
    assert cfg.model.c > 0


def test_missing_fields_fall_back_to_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "model:\n  c: 0.5\n"))
    assert cfg.model.c == 0.5
    assert cfg.data == TrainConfig().data


def test_empty_file_gives_defaults(tmp_path):
    assert load_config(write(tmp_path, "")) == TrainConfig()


@pytest.mark.parametrize(
    "text",
    [
        "data:\n  test_size: 5\n",  # out of range
        "model:\n  c: -1\n",  # must be positive
        "model:\n  cc: 1\n",  # typo in a key
    ],
)
def test_invalid_config_is_rejected(tmp_path, text):
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, text))
