from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    raw_dir: Path
    processed_dir: Path


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg: dict) -> Paths:
    p = cfg["paths"]
    return Paths(
        data_dir=Path(p["data_dir"]),
        raw_dir=Path(p["raw_dir"]),
        processed_dir=Path(p["processed_dir"]),
    )
