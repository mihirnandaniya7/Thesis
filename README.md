# Surrogate Thesis Scaffold

This repository implements a thesis-safe baseline for surrogate modeling based short-term load forecasting in microgrids.

The project follows the intended thesis order:

1. Define the reference system.
2. Generate synthetic data from the simulator.
3. Build supervised windows.
4. Train baselines.
5. Train the Transformer surrogate.
6. Evaluate accuracy and runtime.
7. Keep Stage 2 microgrid realism optional until the baseline is stable.

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

## Stage 2 Upgrade Path

The simulator already supports an optional richer microgrid mode with PV and battery state-of-charge. It is disabled by default so the baseline thesis pipeline stays focused and defensible.

