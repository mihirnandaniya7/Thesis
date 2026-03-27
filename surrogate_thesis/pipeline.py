from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from surrogate_thesis.config import ExperimentConfig
from surrogate_thesis.controller import HybridController
from surrogate_thesis.data import prepare_dataset, save_dataset_artifacts
from surrogate_thesis.evaluation import evaluate_model
from surrogate_thesis.evaluation.plots import (
    plot_error_by_hour,
    plot_prediction_overview,
    plot_residual_overview,
    plot_runtime_comparison,
)
from surrogate_thesis.simulation import ReferenceSimulator
from surrogate_thesis.training import fit_model


def run_experiment(config: ExperimentConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    resolved_output = _resolve_output_dir(config.run_name, output_dir)
    data_dir = resolved_output / "data"
    models_dir = resolved_output / "models"
    plots_dir = resolved_output / "plots"
    for path in (resolved_output, data_dir, models_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    _set_global_seeds(config.seed)
    config.save(resolved_output / "resolved_config.json")

    simulator = ReferenceSimulator()
    raw_frame = simulator.generate_series(config.simulator, seed=config.seed)
    dataset = prepare_dataset(raw_frame, config.dataset)
    save_dataset_artifacts(dataset, data_dir)

    evaluation_results = []
    prediction_columns: dict[str, np.ndarray] = {}
    model_artifacts = {}

    for model_name in config.model_names:
        artifacts = fit_model(
            model_name=model_name,
            train_split=dataset.train,
            val_split=dataset.val,
            config=config,
            output_dir=models_dir,
        )
        model_artifacts[model_name] = artifacts
        result = evaluate_model(
            reference_simulator=simulator,
            artifacts=artifacts,
            test_split=dataset.test,
            normalization=dataset.normalization,
            config=config,
        )
        evaluation_results.append(result)
        prediction_columns[model_name] = result.predictions

    metrics_frame = pd.DataFrame([result.to_record() for result in evaluation_results]).sort_values(
        by="MAE"
    )
    metrics_frame.to_csv(resolved_output / "metrics.csv", index=False)
    predictions_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dataset.test.target_timestamps),
            "y_true": evaluation_results[0].y_true[:, 0],
            **{name: values[:, 0] for name, values in prediction_columns.items()},
        }
    )
    predictions_frame.to_csv(resolved_output / "test_predictions.csv", index=False)

    plot_prediction_overview(
        timestamps=dataset.test.target_timestamps,
        y_true=evaluation_results[0].y_true,
        predictions_by_model=prediction_columns,
        output_path=plots_dir / "prediction_overview.png",
        points=config.evaluation.prediction_plot_points,
    )
    plot_residual_overview(
        y_true=evaluation_results[0].y_true,
        predictions_by_model=prediction_columns,
        output_path=plots_dir / "residuals.png",
    )
    plot_runtime_comparison(
        metrics_frame=metrics_frame,
        output_path=plots_dir / "runtime_comparison.png",
    )
    plot_error_by_hour(
        hours=dataset.test.target_hours,
        y_true=evaluation_results[0].y_true,
        predictions_by_model=prediction_columns,
        output_path=plots_dir / "error_by_hour.png",
    )

    best_model_row = metrics_frame.iloc[0]
    controller = HybridController(threshold=float(best_model_row["MAE"]) * 1.1)
    summary = {
        "run_name": config.run_name,
        "output_dir": str(resolved_output),
        "best_model": str(best_model_row["model_name"]),
        "controller_demo_decision": controller.decide(float(best_model_row["MAE"])),
        "metrics": metrics_frame.to_dict(orient="records"),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (resolved_output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_output_dir(run_name: str, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / f"{timestamp}_{run_name}"
