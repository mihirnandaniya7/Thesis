from __future__ import annotations

import copy
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from surrogate_thesis.config import ExperimentConfig
from surrogate_thesis.data.dataset import DatasetSplit
from surrogate_thesis.models import (
    LSTMRegressor,
    LinearRegressionBaseline,
    PersistenceBaseline,
    TransformerRegressor,
)


@dataclass(slots=True)
class ModelArtifacts:
    name: str
    model: Any
    checkpoint_path: Path
    history: dict[str, list[float]] = field(default_factory=dict)
    training_seconds: float = 0.0


def fit_model(
    model_name: str,
    train_split: DatasetSplit,
    val_split: DatasetSplit,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> ModelArtifacts:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_dim = train_split.X.shape[-1]
    horizon = train_split.y.shape[-1]

    if model_name == "persistence":
        target_index = config.dataset.feature_columns.index(config.dataset.target_column)
        model = PersistenceBaseline(
            target_feature_index=target_index,
            horizon=horizon,
        ).fit(train_split.X, train_split.y)
        checkpoint = output_path / f"{model_name}.pkl"
        checkpoint.write_bytes(pickle.dumps(model))
        return ModelArtifacts(name=model_name, model=model, checkpoint_path=checkpoint)

    if model_name == "linear_regression":
        model = LinearRegressionBaseline().fit(train_split.X, train_split.y)
        checkpoint = output_path / f"{model_name}.pkl"
        checkpoint.write_bytes(pickle.dumps(model))
        return ModelArtifacts(name=model_name, model=model, checkpoint_path=checkpoint)

    model = _build_torch_model(
        model_name=model_name,
        input_dim=input_dim,
        horizon=horizon,
        config=config,
    )
    checkpoint = output_path / f"{model_name}.pt"
    history_path = output_path / f"{model_name}_history.json"
    history, training_seconds = _train_torch_model(
        model=model,
        train_split=train_split,
        val_split=val_split,
        checkpoint_path=checkpoint,
        config=config,
    )
    history_path.write_text(json.dumps(history, indent=2))
    return ModelArtifacts(
        name=model_name,
        model=model,
        checkpoint_path=checkpoint,
        history=history,
        training_seconds=training_seconds,
    )


def predict_model(
    artifacts: ModelArtifacts,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    if hasattr(artifacts.model, "predict"):
        return np.asarray(artifacts.model.predict(X), dtype=np.float32)

    model = artifacts.model
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    tensor_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (features,) in loader:
            predictions = model(features.to(torch_device)).cpu().numpy()
            outputs.append(predictions)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _build_torch_model(
    model_name: str,
    input_dim: int,
    horizon: int,
    config: ExperimentConfig,
) -> nn.Module:
    model_config = config.model
    if model_name == "lstm":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_size=model_config.lstm_hidden_size,
            num_layers=model_config.lstm_num_layers,
            horizon=horizon,
            dropout=model_config.dropout,
        )
    if model_name == "transformer":
        return TransformerRegressor(
            input_dim=input_dim,
            d_model=model_config.transformer_d_model,
            nhead=model_config.transformer_nhead,
            num_layers=model_config.transformer_num_layers,
            dim_feedforward=model_config.transformer_dim_feedforward,
            horizon=horizon,
            dropout=model_config.dropout,
        )
    raise ValueError(f"Unknown torch model: {model_name}")


def _train_torch_model(
    model: nn.Module,
    train_split: DatasetSplit,
    val_split: DatasetSplit,
    checkpoint_path: Path,
    config: ExperimentConfig,
) -> tuple[dict[str, list[float]], float]:
    device = torch.device(config.training.device)
    model.to(device)

    train_dataset = TensorDataset(
        torch.tensor(train_split.X, dtype=torch.float32),
        torch.tensor(train_split.y, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_split.X, dtype=torch.float32),
        torch.tensor(val_split.y, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    start = perf_counter()

    for _ in range(config.training.max_epochs):
        model.train()
        train_loss_total = 0.0
        train_examples = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_size = len(features)
            train_loss_total += loss.item() * batch_size
            train_examples += batch_size

        model.eval()
        val_loss_total = 0.0
        val_examples = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                predictions = model(features)
                loss = criterion(predictions, targets)
                batch_size = len(features)
                val_loss_total += loss.item() * batch_size
                val_examples += batch_size

        epoch_train_loss = train_loss_total / max(train_examples, 1)
        epoch_val_loss = val_loss_total / max(val_examples, 1)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        if epoch_val_loss < best_val_loss - 1e-8:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.patience:
                break

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_path)
    return history, perf_counter() - start
