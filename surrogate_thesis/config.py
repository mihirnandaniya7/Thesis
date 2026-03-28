from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _construct_dataclass(cls: type, data: dict[str, Any]) -> Any:
    valid_fields = {item.name for item in fields(cls)}
    payload = {key: value for key, value in data.items() if key in valid_fields}
    return cls(**payload)


@dataclass(slots=True)
class SimulatorConfig:
    start: str = "2025-01-01"
    days: int = 120
    time_resolution_minutes: int = 15
    internal_substeps: int = 24
    base_load_kw: float = 1.8
    daily_amplitude_kw: float = 0.9
    morning_peak_kw: float = 0.45
    evening_peak_kw: float = 0.85
    weekday_multiplier: float = 1.08
    weekend_multiplier: float = 0.93
    seasonal_amplitude: float = 0.12
    noise_std_kw: float = 0.08
    ar_coefficient: float = 0.72
    peak_event_probability: float = 0.05
    peak_event_scale_kw: float = 0.7
    include_stage2_microgrid: bool = False
    pv_peak_kw: float = 2.5
    battery_capacity_kwh: float = 5.0
    battery_power_kw: float = 1.8


@dataclass(slots=True)
class DatasetConfig:
    lookback: int = 16
    horizon: int = 1
    feature_columns: list[str] = field(
        default_factory=lambda: ["load_kw", "hour_sin", "hour_cos", "is_weekend"]
    )
    target_column: str = "load_kw"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass(slots=True)
class ModelConfig:
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 3
    transformer_dim_feedforward: int = 128
    dropout: float = 0.1


@dataclass(slots=True)
class TrainingConfig:
    device: str = "cpu"
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 6


@dataclass(slots=True)
class EvaluationConfig:
    latency_samples: int = 256
    prediction_plot_points: int = 288


@dataclass(slots=True)
class DecoratorConfig:
    enabled: bool = True
    candidate_model_names: list[str] = field(default_factory=lambda: ["lstm", "transformer"])
    rolling_window: int = 24
    warmup_steps: int = 32
    threshold_multipliers: list[float] = field(
        default_factory=lambda: [0.75, 1.0, 1.25, 1.5]
    )
    preferred_threshold_multiplier: float = 1.0


@dataclass(slots=True)
class ExperimentConfig:
    run_name: str = "baseline_v1"
    seed: int = 42
    model_names: list[str] = field(
        default_factory=lambda: ["persistence", "linear_regression", "lstm", "transformer"]
    )
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    decorator: DecoratorConfig = field(default_factory=DecoratorConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        config = _construct_dataclass(cls, data)
        if isinstance(data.get("simulator"), dict):
            config.simulator = _construct_dataclass(SimulatorConfig, data["simulator"])
        if isinstance(data.get("dataset"), dict):
            config.dataset = _construct_dataclass(DatasetConfig, data["dataset"])
        if isinstance(data.get("model"), dict):
            config.model = _construct_dataclass(ModelConfig, data["model"])
        if isinstance(data.get("training"), dict):
            config.training = _construct_dataclass(TrainingConfig, data["training"])
        if isinstance(data.get("evaluation"), dict):
            config.evaluation = _construct_dataclass(EvaluationConfig, data["evaluation"])
        if isinstance(data.get("decorator"), dict):
            config.decorator = _construct_dataclass(DecoratorConfig, data["decorator"])
        return config

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
