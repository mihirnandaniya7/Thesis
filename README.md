# Surrogate Thesis Scaffold

This repository implements a surrogate-integrated simulation prototype for short-term microgrid forecasting. The current version combines a trusted reference simulator, multiple surrogate models, and a decorator-based switching layer with probe-based trust estimation and lightweight online recalibration.

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
- `surrogate_thesis/controller/surrogate_decorator.py`: runtime decorator, probing, fallback, and online recalibration
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

- single-sample latency
- batched per-sample latency
- full test-set runtime
- speedup against the reference simulator using the full test set

## Decorator Runtime Supervision

The decorator layer now supports:

- probe-based trust estimation
- fallback to the reference simulator under low trust
- lightweight online recalibration from trusted high-fidelity labels

At runtime, the decorator does three things:

1. selects whether the next-step metric comes from the surrogate or the high-fidelity simulator
2. updates trust from periodic probe comparisons against trusted simulator outputs
3. uses trusted high-fidelity labels to recalibrate surrogate predictions online
