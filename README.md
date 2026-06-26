# Surrogate Modeling Based on Machine Learning Approaches for Hospitals as Microgrids

This repository implements a surrogate-integrated simulation prototype for short-term hospital-microgrid forecasting. The current version combines a trusted reference simulator, explicit component transition functions, multiple surrogate models, and decorator-based switching with probe-based trust estimation and lightweight online recalibration.

The implementation has two deliberately separate layers:

- **Architecture layer:** load, PV, and battery expose `state + parameters + action + delta_t -> next_state`, and `ComponentSurrogateDecorator` can wrap components that share this interface.
- **Current quantitative evaluation layer:** the final reported metrics use the stable offline `ForecastProvider.forecast(index)` adapter over prepared validation/test arrays.

The component-level decorator is implemented and unit-tested, but the reported metrics are **not** from a full component-level trajectory rollout. A component-trained surrogate and end-to-end rollout evaluator are future work.

The implemented workflow is:

1. Define the reference simulator.
2. Expose simulator behavior through component transitions.
3. Generate synthetic trajectories.
4. Construct supervised forecasting windows.
5. Train surrogate candidates.
6. Evaluate fidelity and runtime through the offline forecast-index adapter.
7. Evaluate decorator switching, probes, fallback, and recalibration on that adapter.

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
- `surrogate_thesis/simulation/component_interface.py`: common component contract
- `surrogate_thesis/simulation/reference_simulator.py`: trusted reference simulator and explicit load/PV/battery component functions
- `surrogate_thesis/data/dataset.py`: window construction and normalization
- `surrogate_thesis/training/trainer.py`: offline model training
- `surrogate_thesis/controller/hybrid_controller.py`: switching policy
- `surrogate_thesis/controller/surrogate_decorator.py`: component decorator, forecast adapter layer, probing, fallback, recalibration, and evaluation runner
- `tests/test_decorator.py`: focused decorator tests

## Quick Start

Create an environment and install the required libraries:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas scikit-learn matplotlib torch
```

The repository is run directly from its root directory. Editable installation with `pip install -e .` is not currently supported by the project packaging configuration.

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

Generate the timestamp-alignment evidence for a completed run:

```bash
python3 scripts/generate_alignment_evidence.py artifacts/<run_name>
```

## Verification

The current implementation was validated on 2026-06-25 with the following checks:

- `python3 -m compileall -q scripts surrogate_thesis tests` completed successfully.
- `python3 -m unittest discover -s tests -v` completed successfully with 16 passing tests.
- All JSON configuration files in `configs/` parsed successfully.

## Generated Files

The repository intentionally ignores generated experiment outputs and local runtime caches:

- `artifacts/`
- `__pycache__/`
- `.DS_Store`
- `.mplconfig/`

These files may still appear in the working directory after running experiments or plots, but they are not part of the source package and should not be treated as thesis code.

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

## Component Transition Contract

The thesis-facing simulator interface is the component transition:

```text
f(state, parameters, action) --delta_t--> next_state
```

In code, `delta_t` is passed as an argument so the simulator component and the surrogate component can use the same callable signature:

```python
next_state = component_step(state, parameters, action, delta_t)
```

The dictionaries separate the information needed by a model:

- `state`: dynamic model variables, for example battery state-of-charge, previous load, cloud state, or inertia state
- `parameters`: static configuration, for example battery capacity, PV peak power, efficiencies, or load-profile constants
- `action`: external inputs, disturbances, or control values, for example `load_kw`, `pv_kw`, timestamp context, or stochastic disturbance samples

The reference simulator now exposes this form for load, PV, and battery behavior through explicit component functions. The object-oriented simulator still exists for end-to-end trajectory generation, but its internal methods delegate to these state-explicit functions.

## Runtime Evaluation

The evaluation now reports:

- MAE, RMSE, MAPE, sMAPE
- NMAE and NRMSE normalized by the mean absolute target magnitude
- single-sample latency
- batched per-sample latency
- full test-set runtime
- speedup against the reference simulator using the full test set

## Decorator Runtime Supervision

The runtime architecture has two related layers:

- component layer: `state + parameters + action --delta_t--> next_state`
- evaluation adapter layer: `ForecastProvider.forecast(index)` for replaying prepared validation/test arrays during offline experiments

The component layer is the target architecture for simulator replacement. A trusted simulator component and a surrogate component can be wrapped by `ComponentSurrogateDecorator`, which compares their returned state dictionaries on probe steps and switches between them according to the hybrid controller. This decorator is implemented and tested, but it is not the object used to generate the current final metrics.

The `forecast(index)` layer is kept for the existing experiment pipeline. It lets the evaluation runner compare trained models against prepared test data without changing all current result-generation scripts. It should be understood as an offline adapter layer, not as the final component API.

The forecast-index decorator used for the current metrics supports:

- probe-based trust estimation
- adaptive probe spacing when the surrogate remains stable
- fallback to the reference simulator under low trust
- lightweight online recalibration from trusted high-fidelity labels

Within the current offline evaluation, the decorator does four things:

1. selects whether the next-step metric comes from the surrogate or the high-fidelity simulator
2. accesses the stored high-fidelity reference output only on warmup, probe, fallback, or re-entry steps
3. updates trust from periodic probe comparisons against trusted simulator outputs
4. uses trusted high-fidelity labels to recalibrate surrogate predictions online

In a future live component rollout, the high-fidelity component would be executed on those same warmup, probe, fallback, and re-entry steps.

The experiment runner is intentionally separate from the runtime decorator. It iterates over the test split, collects traces, computes metrics, and uses the offline test ground truth only for final evaluation. The threshold scale is derived from validation MAE; the preferred threshold multiplier is a predefined configuration value and is reported with the sensitivity analysis.

## Current Corrected Results

Use the corrected architecture artifact for reporting:

```text
artifacts/decorator_architecture_current
```

`artifacts/` is intentionally ignored by Git because it contains generated datasets, model checkpoints, and plots. The result directory must therefore be supplied separately to a reviewer, or regenerated with the default configuration. The table below is the versioned summary of the supplied final artifact.

| Model | MAE | RMSE | NMAE | NRMSE | Full-test speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.0623 kW | 0.0833 | 3.28% | 4.39% | 8.47x |
| Transformer | 0.0648 kW | 0.0876 | 3.41% | 4.61% | 4.48x |
| Linear Regression | 0.0700 kW | 0.0895 | 3.69% | 4.71% | 10035.58x |
| Persistence | 0.0954 kW | 0.1207 | 5.02% | 6.36% | 136090.95x |

The corresponding decorator summary is `artifacts/decorator_architecture_current/decorator/decorator_summary.csv`.

| Decorated provider | Decorator MAE | Decorator NMAE | Decorator NRMSE | Surrogate usage | Simulator usage | Decorator speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM decorator | 0.0432 kW | 2.28% | 3.41% | 78.19% | 21.81% | 0.86x |
| Transformer decorator | 0.0437 kW | 2.30% | 3.57% | 74.88% | 25.12% | 1.12x |

Interpretation for the thesis: the LSTM is currently the most accurate standalone surrogate, while the Transformer decorator is the stronger runtime-oriented result because it keeps error controlled and achieves speedup above the simulator-only baseline. Full-test speedups use batched inference; the single-step latency columns are the relevant online-style measurement. Runtime values are hardware-, PyTorch-, and configuration-dependent.

## Important Output Files

- `summary.json`: compact run summary and generated result metadata
- `metrics.csv`: standalone surrogate accuracy and runtime results
- `decorator/decorator_summary.csv`: preferred decorator results for LSTM and Transformer
- `decorator/lstm_threshold_sensitivity.csv`: LSTM threshold trade-off sweep
- `decorator/transformer_threshold_sensitivity.csv`: Transformer threshold trade-off sweep
- `decorator/lstm_preferred_trace.csv`: step-by-step LSTM decorator decisions
- `decorator/transformer_preferred_trace.csv`: step-by-step Transformer decorator decisions
- `test_predictions.csv`: final test predictions for all standalone models
- `alignment_check.csv`: target timestamp and persistence-window alignment evidence
- `alignment_shift_check.csv`: small-shift MAE comparison for each model

## Important Plots

- `plots/prediction_overview.pdf`: true net load compared with model predictions
- `plots/residuals.pdf`: density-normalized prediction residual distributions with shared axes
- `plots/runtime_comparison.png`: runtime comparison against the reference simulator
- `plots/normalized_error_comparison.png`: normalized error comparison
- `plots/error_by_hour.png`: model error grouped by hour of day
- `decorator/lstm_decision_trace.png`: LSTM decorator mode switches, probes, and rolling error
- `decorator/transformer_decision_trace.png`: Transformer decorator mode switches, probes, and rolling error

For thesis insertion, figures should be exported as vector graphics such as PDF or SVG. The prediction overview and residual figures already follow that convention. The remaining PNG plots are useful diagnostic artifacts and can be regenerated as vector graphics before being used as final thesis figures.

Figure captions in the thesis should provide the figure context. Therefore, final thesis figures should avoid redundant in-figure titles and use axis labels and legends with a font size comparable to the surrounding thesis text.
