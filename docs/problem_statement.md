# Problem Statement

## Title

Surrogate Modeling Based on Machine Learning Approaches for Short-Term Load Forecasting in Microgrids

## System Being Approximated

The reference system is a synthetic, high-fidelity microgrid load simulator that produces timestamped electrical demand time-series at 15-minute resolution. The simulator is sequential and stateful: current demand depends on time-of-day structure, weekday versus weekend effects, seasonal variation, correlated stochastic disturbances, and occasional peak events. In the optional Stage 2 mode, the simulator also models photovoltaic generation, battery state-of-charge, and net-load behavior.

## Forecasting Task

- Forecast type: short-term load forecasting
- Time resolution: 15 minutes
- Lookback window: 32 past steps
- Prediction horizon: 1 future step
- Input features: load, PV generation, battery state-of-charge, net load, hour-of-day sine/cosine features, and a weekday/weekend flag
- Target variable: next-step net load

## Research Contribution

The contribution is not a downloaded dataset benchmark. Instead, the simulator itself is part of the research artifact. Synthetic data is generated from the reference model, then machine-learning surrogates are trained to approximate its behavior. This creates a defensible surrogate-modeling workflow:

Reference simulator -> synthetic dataset -> surrogate training -> accuracy/runtime evaluation

## Why the Simulator Is the Ground Truth

The reference simulator is treated as the ground truth because it encodes the load-generation assumptions that define the problem setting. It is executed sequentially over long horizons, preserves internal state, and produces the signals that the surrogate is asked to emulate. Although it is still a simplified model, it is richer than a direct regression formula and provides the data-generation mechanism required by the thesis.

## Evaluation Criteria

The surrogate models are evaluated on two axes:

1. Predictive fidelity:
   - MAE
   - RMSE
   - sMAPE
2. Computational efficiency:
   - average inference latency
   - speedup relative to the simulator

## Initial Model Line-up

- Main surrogate: encoder-only Transformer
- Sequence baseline: LSTM
- Lower-bound baselines: persistence and linear regression

## Scope Boundary

The current implementation uses a richer Stage 2 simulator by default so that the surrogate is evaluated on a more meaningful microgrid forecasting task. Hybrid switching remains a later decision layer after surrogate quality is measured.
