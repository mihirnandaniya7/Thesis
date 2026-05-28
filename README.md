# Surrogate Modeling Based on Machine Learning Approaches for Hospitals as Microgrids

This repository implements a surrogate-integrated simulation prototype for short-term microgrid forecasting. The current version combines a trusted reference simulator, multiple surrogate models, and a real decorator-based runtime layer with probe-based trust estimation and lightweight online recalibration.

The implemented workflow is:

1. Define the reference simulator.
2. Generate synthetic trajectories.
3. Construct supervised windows.
4. Train surrogate candidates.
5. Evaluate fidelity and runtime.
6. Integrate the surrogate through decorator-managed runtime switching.
7. Recalibrate the surrogate online from trusted high-fidelity labels.

## Structure

```text
docs/                 Thesis-facing written artifacts
configs/              Experiment configurations
scripts/              Entry points
surrogate_thesis/     Python package
tests/                Reproducibility and smoke tests
```

## Review Entry Points

For code review, the most important files are:

- `docs/problem_statement.md`: current thesis framing, methodology, and notation
- `scripts/run_pipeline.py`: command-line entry point
- `surrogate_thesis/pipeline.py`: end-to-end experiment orchestration
- `surrogate_thesis/simulation/reference_simulator.py`: trusted reference simulator
- `surrogate_thesis/data/dataset.py`: window construction and normalization
- `surrogate_thesis/training/trainer.py`: offline model training
- `surrogate_thesis/controller/hybrid_controller.py`: switching policy
- `surrogate_thesis/controller/surrogate_decorator.py`: common forecast interface, runtime decorators, probing, fallback, recalibration, and evaluation runner
- `tests/test_decorator.py`: focused decorator tests

## Quick Start

Run the full default experiment:

```bash
python3 scripts/run_pipeline.py --config configs/default_experiment.json
```

Run a fast smoke experiment:

```bash
python3 scripts/run_pipeline.py --config configs/smoke_experiment.json --output-dir artifacts/smoke_manual
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Inspect the main outputs of a run:

```bash
cat artifacts/<run_name>/summary.json
cat artifacts/<run_name>/metrics.csv
cat artifacts/<run_name>/decorator/decorator_summary.csv
```

## What Gets Produced

Each experiment writes:

- resolved configuration
- raw simulated dataset
- processed dataset artifacts
- model checkpoints and training history
- metrics summary
- prediction and residual plots
- runtime comparison plot
- normalized error comparison plot
- error-by-hour plot
- decorator threshold sensitivity and decision traces

## Default Modeling Line-up

- `persistence`: lower bound baseline
- `linear_regression`: classical regression baseline
- `lstm`: sequence baseline
- `transformer`: main surrogate model

## Current Default Task

- Stage 2 microgrid mode is enabled in the default experiment
- target: `net_load_kw`
- features: `load_kw`, `pv_kw`, `battery_soc_kwh`, `net_load_kw`, `hour_sin`, `hour_cos`, `is_weekend`
- lookback: `32`
- horizon: `1`

## Runtime Evaluation

The evaluation now reports:

- MAE, RMSE, MAPE, sMAPE
- NMAE and NRMSE normalized by the mean absolute target magnitude
- single-sample latency
- batched per-sample latency
- full test-set runtime
- speedup against the reference simulator using the full test set

## Decorator Runtime Supervision

The runtime architecture uses a common `ForecastProvider.forecast(index)` interface. The high-fidelity simulator adapter, surrogate adapter, recalibration decorator, and trust-managed surrogate decorator all expose this same interface, so the decorated provider can be used as a transparent replacement by the evaluation loop.

The decorator layer supports:

- probe-based trust estimation
- adaptive probe spacing when the surrogate remains stable
- fallback to the reference simulator under low trust
- lightweight online recalibration from trusted high-fidelity labels

At runtime, the decorator does four things:

1. selects whether the next-step metric comes from the surrogate or the high-fidelity simulator
2. executes the high-fidelity simulator lazily only on warmup, probe, fallback, or re-entry steps
3. updates trust from periodic probe comparisons against trusted simulator outputs
4. uses trusted high-fidelity labels to recalibrate surrogate predictions online

The experiment runner is intentionally separate from the runtime decorator. It iterates over the test split, collects traces, computes metrics, and uses the offline test ground truth only for final evaluation. Switching thresholds are calibrated from the validation split, not from the test split.

## Current Corrected Results

Use the corrected architecture artifact for reporting:

```text
artifacts/decorator_architecture_current
```

The standalone model results are stored in [`metrics.csv`](artifacts/decorator_architecture_current/metrics.csv).

| Model | MAE | RMSE | NMAE | NRMSE | Full-test speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.0623 kW | 0.0833 | 3.28% | 4.39% | 8.47x |
| Transformer | 0.0648 kW | 0.0876 | 3.41% | 4.61% | 4.48x |
| Linear Regression | 0.0700 kW | 0.0895 | 3.69% | 4.71% | 10035.58x |
| Persistence | 0.0954 kW | 0.1207 | 5.02% | 6.36% | 136090.95x |

The decorator architecture results are stored in [`decorator_summary.csv`](artifacts/decorator_architecture_current/decorator/decorator_summary.csv).

| Decorated provider | Decorator MAE | Decorator NMAE | Decorator NRMSE | Surrogate usage | Simulator usage | Decorator speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM decorator | 0.0432 kW | 2.28% | 3.41% | 78.19% | 21.81% | 0.86x |
| Transformer decorator | 0.0437 kW | 2.30% | 3.57% | 74.88% | 25.12% | 1.12x |

Interpretation for the thesis: the LSTM is currently the most accurate standalone surrogate, while the Transformer decorator is the stronger runtime-oriented result because it keeps error controlled and achieves speedup above the simulator-only baseline.

## Important Output Files

- [`summary.json`](artifacts/decorator_architecture_current/summary.json): compact run summary and generated result metadata
- [`metrics.csv`](artifacts/decorator_architecture_current/metrics.csv): standalone surrogate accuracy and runtime results
- [`decorator_summary.csv`](artifacts/decorator_architecture_current/decorator/decorator_summary.csv): preferred decorator results for LSTM and Transformer
- [`lstm_threshold_sensitivity.csv`](artifacts/decorator_architecture_current/decorator/lstm_threshold_sensitivity.csv): LSTM threshold trade-off sweep
- [`transformer_threshold_sensitivity.csv`](artifacts/decorator_architecture_current/decorator/transformer_threshold_sensitivity.csv): Transformer threshold trade-off sweep
- [`lstm_preferred_trace.csv`](artifacts/decorator_architecture_current/decorator/lstm_preferred_trace.csv): step-by-step LSTM decorator decisions
- [`transformer_preferred_trace.csv`](artifacts/decorator_architecture_current/decorator/transformer_preferred_trace.csv): step-by-step Transformer decorator decisions
- [`test_predictions.csv`](artifacts/decorator_architecture_current/test_predictions.csv): final test predictions for all standalone models

## Important Plots

- [`prediction_overview.png`](artifacts/decorator_architecture_current/plots/prediction_overview.png): true net load compared with model predictions
- [`residuals.png`](artifacts/decorator_architecture_current/plots/residuals.png): prediction residual distributions
- [`runtime_comparison.png`](artifacts/decorator_architecture_current/plots/runtime_comparison.png): runtime comparison against the reference simulator
- [`normalized_error_comparison.png`](artifacts/decorator_architecture_current/plots/normalized_error_comparison.png): normalized error comparison
- [`error_by_hour.png`](artifacts/decorator_architecture_current/plots/error_by_hour.png): model error grouped by hour of day
- [`lstm_decision_trace.png`](artifacts/decorator_architecture_current/decorator/lstm_decision_trace.png): LSTM decorator mode switches, probes, and rolling error
- [`transformer_decision_trace.png`](artifacts/decorator_architecture_current/decorator/transformer_decision_trace.png): Transformer decorator mode switches, probes, and rolling error
