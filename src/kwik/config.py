"""Run configuration, loaded from a YAML file in configs/."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class DatasetSpec:
    name: str          # "triviaqa" | "nq_open" | "popqa" | "local"
    n: int             # questions to sample
    role: str = "train"  # "train" = pooled into train/val/test splits; "transfer" = held out entirely
    path: str | None = None  # only for name == "local" (a .jsonl with question, answers)


@dataclass
class Config:
    run_name: str
    model_id: str
    datasets: list[DatasetSpec]
    out_dir: str = "runs"
    seed: int = 0
    batch_size: int = 8
    max_new_tokens: int = 24
    layer_stride: int = 2          # save every k-th layer (plus the last) to keep files small
    dtype: str = "auto"            # "auto" | "float32" | "bfloat16" | "float16"
    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"
    load_in_4bit: bool = False     # Colab GPU only (needs bitsandbytes)
    split: dict = field(default_factory=lambda: {"train": 0.6, "val": 0.2, "test": 0.2})
    sampling_k: int = 5            # samples for the agreement baseline
    sampling_temperature: float = 1.0
    run_baselines: bool = True

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    raw["datasets"] = [DatasetSpec(**d) for d in raw["datasets"]]
    return Config(**raw)
