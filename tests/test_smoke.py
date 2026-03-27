from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from surrogate_thesis.config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    SimulatorConfig,
    TrainingConfig,
)
from surrogate_thesis.pipeline import run_experiment


class SmokePipelineTests(unittest.TestCase):
    def test_smoke_pipeline_runs_end_to_end(self) -> None:
        config = ExperimentConfig(
            run_name="pytest_smoke",
            seed=21,
            simulator=SimulatorConfig(
                days=10,
                internal_substeps=4,
                include_stage2_microgrid=False,
            ),
            dataset=DatasetConfig(
                lookback=16,
                horizon=1,
                feature_columns=["load_kw", "hour_sin", "hour_cos", "is_weekend"],
                target_column="load_kw",
            ),
            model=ModelConfig(
                lstm_hidden_size=16,
                lstm_num_layers=1,
                transformer_d_model=32,
                transformer_nhead=4,
                transformer_num_layers=2,
                transformer_dim_feedforward=64,
                dropout=0.1,
            ),
            training=TrainingConfig(
                device="cpu",
                batch_size=32,
                learning_rate=1e-3,
                weight_decay=1e-4,
                max_epochs=2,
                patience=1,
            ),
            evaluation=EvaluationConfig(
                latency_samples=12,
                prediction_plot_points=48,
            ),
        )

        with TemporaryDirectory() as temp_dir:
            summary = run_experiment(config=config, output_dir=Path(temp_dir) / "run")
            self.assertIn(
                summary["best_model"],
                {"persistence", "linear_regression", "lstm", "transformer"},
            )
            self.assertTrue((Path(temp_dir) / "run" / "summary.json").exists())
            self.assertTrue((Path(temp_dir) / "run" / "metrics.csv").exists())
            self.assertTrue((Path(temp_dir) / "run" / "plots" / "prediction_overview.png").exists())


if __name__ == "__main__":
    unittest.main()
