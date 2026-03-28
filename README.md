# Surrogate Thesis Scaffold

This repository implements a staged surrogate-modeling thesis scaffold for short-term forecasting in microgrids.

The project follows the intended thesis order:

1. Define the reference system.
2. Generate synthetic data from the simulator.
3. Build supervised windows.
4. Train baselines.
5. Train the Transformer surrogate.
6. Evaluate accuracy and runtime.
7. Extend to richer microgrid behavior and hybrid switching.

## Structure

```text
docs/                 Thesis-facing written artifacts
configs/              Experiment configurations
scripts/              Entry points
surrogate_thesis/     Python package
tests/                Reproducibility and smoke tests
```

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
python3 -m pytest
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
