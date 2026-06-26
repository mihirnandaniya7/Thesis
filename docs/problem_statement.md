# Problem Statement

## Title

Surrogate Modeling Based on Machine Learning Approaches for Hospitals as Microgrids

## Motivation

Microgrid studies often require long-horizon simulation over many time steps while still depending on detailed submodels for short-term operational behavior such as demand variation, photovoltaic generation, battery state-of-charge, and net-load dynamics. These detailed models are more trustworthy than simple approximations, but they are also more computationally expensive to execute repeatedly inside a larger simulation loop.

This creates a practical tension: the outer simulation needs fast repeated evaluations, while the inner model needs sufficient fidelity. A surrogate model can reduce this cost by learning to approximate the expensive model output, but a surrogate should not be trusted blindly. The thesis therefore focuses on both surrogate modeling and adaptive runtime integration.

## Problem Statement

The core problem is how to integrate a fast machine-learning surrogate into a simulation workflow without losing too much reliability. The reference simulator progresses at 15-minute external time steps and evaluates finer internal substeps using state, time-dependent structure, and microgrid-specific dynamics. Repeated execution of such a detailed model can become expensive in larger studies.

The thesis addresses this problem through two complementary implementation layers. First, it evaluates next-step surrogate forecasts and a trust-managed switching policy using a stable offline test-set adapter. Second, it implements a state-explicit component interface and decorator architecture for future live simulator integration. The architecture preserves a path back to the trusted simulator whenever the surrogate becomes unreliable.

## Research Question

The main research question is:

How can a machine-learning surrogate be integrated into a microgrid simulation loop so that computational cost is reduced while predictive fidelity remains acceptable?

This can be split into three practical subquestions:

1. Can a surrogate model learn the next-step output of the reference simulator accurately enough for short-term microgrid forecasting?
2. Which surrogate architecture provides the best tradeoff between prediction fidelity and execution speed?
3. Can a decorator-based switching layer use the surrogate most of the time while still falling back to the reference simulator when trust is low?

## System Being Approximated

The reference system is a synthetic, stateful microgrid simulator that produces timestamped electrical time-series at 15-minute resolution. The simulator contains internal dynamics and time-dependent structure:

- daily demand patterns
- weekday versus weekend effects
- seasonal variation
- correlated stochastic disturbances
- occasional peak events
- photovoltaic generation
- battery state-of-charge and dispatch
- net-load behavior

Conceptually, this simulator acts as the trusted inner model used by the outer simulation loop. It consumes the current system state and recent operating context, then returns the next-step metric required by the outer loop. In the current implementation, the main returned metric is next-step `net_load_kw`.

## Surrogate Task Definition

The surrogate approximates the mapping from recent system history to the next-step metric needed by the simulation loop.

- Forecast type: short-term next-step forecasting
- Time resolution: 15 minutes
- Lookback window: 32 past steps
- Prediction horizon: 1 future step
- Input features: load, PV generation, battery state-of-charge, net load, hour-of-day sine/cosine features, and a weekday/weekend flag
- Target variable: next-step net load

In simplified notation, the surrogate approximates a function of the form:

`(current state, recent history, current context) -> next-step system metric`

## Methodology

The proposed methodology combines offline surrogate learning with a trust-managed runtime architecture. The current quantitative evaluation replays prepared validation and test samples through a forecast-index adapter. The implemented component interface defines how the same trust policy can later be used inside a live discrete-time simulation loop.

### 1. Discrete-Time Problem Formulation

Let the system evolve over discrete time steps indexed by `k`. In general form, the system transition is expressed as:

`F(S_k, p, a_k) \xrightarrow{\Delta t} S_{k+1}`

where:

- `S_k` denotes the system state at time step `k`
- `p` denotes static model parameters or configuration values
- `a_k` denotes the external inputs, disturbances, or control-relevant context at time step `k`
- `\Delta t` denotes the transition interval or forecast duration
- `F(.)` denotes the underlying system dynamics

For the outer simulation loop, the full internal state `S_{k+1}` is not necessarily required. Instead, the loop typically requires a returned metric derived from that state. This metric is written as:

`y_{k+1} = G(S_{k+1})`

In the current implementation, `y_{k+1}` is primarily the next-step `net_load_kw`.

The surrogate does not directly observe the complete hidden simulator state. Instead, it receives a reduced representation built from recent state history and contextual inputs:

`x_k = \phi(S_k, a_k, S_{k-1}, a_{k-1}, ..., S_{k-L+1}, a_{k-L+1})`

where `\phi(.)` extracts the fixed lookback window and the contextual features used by the learning model. The surrogate then approximates the mapping

`\hat{y}_{k+1} = f_\theta(x_k)`

Thus, the implemented learning problem is not to reconstruct the full state `S_{k+1}`, but to predict the next-step metric `y_{k+1}` required by the simulation loop.

### 2. High-Fidelity Reference Simulator

The reference simulator is the trusted high-fidelity model within the framework. It produces sequential microgrid trajectories using structured internal dynamics, including:

- daily demand variation
- weekday versus weekend effects
- seasonal variation
- stochastic disturbances
- occasional peak events
- photovoltaic generation
- battery state-of-charge evolution
- net-load behavior

Methodologically, the reference simulator has three roles:

1. it defines the system behavior that the surrogate must approximate
2. it generates the synthetic trajectories used for offline learning
3. it provides the trusted output used for runtime supervision, fallback, and validation

### 3. Dataset Construction

The reference simulator is executed over a long horizon at 15-minute resolution. The generated trajectories are then transformed into supervised learning samples through sliding-window extraction.

For each time step `k`, the input vector `x_k` is formed from:

- recent load values
- recent PV generation values
- recent battery state-of-charge values
- recent net-load values
- hour-of-day sine/cosine encoding
- weekday versus weekend indicator

The prediction target is the returned next-step metric `y_{k+1}`, which in the present prototype is next-step `net_load_kw`.

The dataset is split chronologically into training, validation, and test partitions. Normalization statistics are estimated on the training split only and are then reused for validation and test data in order to avoid temporal leakage.

### 4. Surrogate Model Development

Multiple surrogate models are trained to approximate the mapping `x_k -> y_{k+1}`:

- persistence as a naive lower-bound baseline
- linear regression as a simple classical baseline
- LSTM as a sequence-model baseline
- an encoder-only Transformer as the main surrogate architecture

This model lineup enables a structured comparison of approximation quality, temporal modeling capability, and computational efficiency.

### 5. Offline Training Stage

Each surrogate is trained on simulator-generated data and validated on a held-out chronological validation split. For neural surrogates, the training stage includes:

- mini-batch optimization
- checkpointing
- early stopping
- fixed random seeds

This stage yields a trained surrogate candidate `f_\theta` for the current forecast-index evaluation. The same training stage also provides the basis for future component-transition surrogate training.

### 6. Decorator-Managed Runtime Integration

After offline training, the thesis-facing target runtime architecture is organized around a common component-transition interface:

`f(state, parameters, action) --delta_t--> next_state`

In the implementation, `delta_t` is passed as an argument to the component call, while the mathematical interpretation is that it defines the transition period. Load, PV, and battery simulator functions expose this interface, and `ComponentSurrogateDecorator` can wrap any simulator and surrogate component that share it. This preserves substitutability: a surrounding simulator can call a component without needing to know whether the returned next state comes from the raw simulator, a future component surrogate, or a decorated component.

The current LSTM and Transformer are window-based forecasters of `net_load_kw`; they are not yet trained as component-transition surrogates. They are therefore evaluated through the separate forecast-index adapter described below.

The implementation separates two roles:

- runtime component functions and decorators, which implement the state-explicit transition contract
- an offline evaluation adapter, which can still expose `forecast(k)` over prepared validation or test arrays for result generation
- an evaluation runner, which iterates over a dataset, records traces, and computes thesis metrics

In a future live component rollout, the trust-managed decorator would decide which component to execute at every time step `k`. If the high-fidelity simulator were selected, its next state would become the official state for that step. If the surrogate were selected and no probe were due, only the surrogate component would execute and return its next state.

In the current offline evaluation, the analogous decisions are replayed over prepared test arrays. The `HighFidelitySimulationAdapter` supplies the stored trusted simulator output only on warmup, probe, fallback, or re-entry steps. This supports controlled evaluation of the switching policy, but it is not a live component-level trajectory rollout.

### 7. Probe-Based Trust Estimation

The surrogate is not compared against the high-fidelity reference at every time step. Instead, trust is estimated from a subset of time steps used as probes. Let `P` denote the set of probe indices. In a live rollout, the trusted simulator would be executed in addition to the surrogate. In the current offline evaluation, the stored trusted output is exposed through the high-fidelity adapter. In both cases, the observed error is computed as:

`e_k = || \hat{y}_{k+1} - y_{k+1}^{HF} || , \quad k \in P`

This makes explicit that the prediction error is only directly observable on probe steps or on steps where the high-fidelity path is selected. On surrogate-only steps, a live runtime decorator would not know the high-fidelity answer and therefore could not update observed error.

To obtain a stable trust signal, the system maintains a rolling error estimate over the most recent probe steps. Let `W_k` denote the rolling probe window. Then:

`\bar{e}_k = \frac{1}{|W_k|} \sum_{i \in W_k} e_i`

The value `\bar{e}_k` is the main trust indicator used by the switching logic. In a live rollout, this would supervise the surrogate without forcing the high-fidelity simulator to run at every time step.

### 8. Switching Policy

The decorator follows an adaptive threshold-based switching policy with warmup, fallback, and re-entry behavior.

During the initial warmup phase, the high-fidelity simulator is used more frequently in order to establish an initial trust history. After warmup, the surrogate is preferred whenever the rolling error estimate remains within the accepted trust region:

`\bar{e}_k \leq \varepsilon`

If the rolling error exceeds the threshold, the system falls back to the high-fidelity simulator:

`\bar{e}_k > \varepsilon`

The base switching threshold is derived from validation MAE before final test evaluation. The preferred threshold multiplier is a predefined configuration value, and the configured multiplier sweep is reported as sensitivity analysis rather than used to select a best result from the test set. In the current implementation, the policy includes:

- periodic validation probes
- rolling error tracking
- fallback execution under low trust
- cooldown and re-entry checks
- hysteresis to avoid unstable rapid switching

Therefore, the switching mechanism defines a continuous runtime supervision policy rather than a one-time model choice. Its current quantitative assessment is an offline replay of that policy.

### 9. Relation to Training Continuation

Whenever the high-fidelity path is selected, the system obtains an additional trusted labeled output. In the offline evaluation, this label is supplied by the stored reference output; in a live rollout, it would come from the executed simulator component. The forecast-index decorator supports lightweight online recalibration from these trusted labels. The same mechanism provides a basis for future component-level retraining or more extensive online adaptation.

### 10. Relation to Multi-Scale Simulation

The broader architectural motivation is multi-scale simulation. In such settings, an outer simulation loop may operate on a coarser decision timescale, while an inner high-fidelity model may require finer internal computation. The surrogate becomes especially valuable when the outer simulation only needs the returned metric `y_{k+1}`, rather than the full internal trajectory of the detailed inner model.

The reference simulator demonstrates a limited version of this motivation by using finer internal substeps within each 15-minute external interval. The thesis does not claim a full FMI-, mosaik-, or HELICS-style co-simulation experiment. Instead, multi-scale simulation motivates the component interface and the future live surrogate-replacement direction.

### 11. Evaluation Methodology

The final stage evaluates the resulting system along two complementary axes:

1. predictive fidelity relative to the high-fidelity simulator
2. computational efficiency relative to simulator-only execution

For clean evaluation, validation MAE determines the base threshold scale, while the test split is reserved for final reporting. During test execution, offline ground truth is used by the evaluation runner to compute metrics. The forecast-index decorator receives the corresponding trusted value only when its own warmup, probe, fallback, or re-entry policy selects the high-fidelity path.

The evaluation includes:

- MAE
- RMSE
- MAPE
- sMAPE
- NMAE
- NRMSE
- runtime per step
- full-run runtime
- speedup
- surrogate usage ratio
- fallback simulation ratio

Overall, the methodological pipeline can be summarized as:

reference simulation -> dataset construction -> surrogate training -> forecast-index decorator evaluation -> probe-based trust update -> system-level evaluation

### 12. Reporting and Figure Preparation

The final thesis distinguishes between reproducibility artifacts and presentation-ready figures. The experiment pipeline writes plots for inspection and result tracking, while figures included in the written thesis should use vector formats such as PDF or SVG so they remain sharp when zoomed.

Figure captions provide the surrounding context in the written document. The figures themselves therefore avoid redundant in-figure titles and use axis labels, tick labels, and legends with a font size comparable to the surrounding body text.

The prediction overview and residual distribution figures are prepared as PDF outputs for thesis use. Other plots, such as runtime comparison, normalized error comparison, hourly error, and decorator traces, remain useful diagnostic artifacts in the generated experiment directory and can be regenerated as vector figures when needed.

### 13. Formula Summary

For clarity, the main symbols used in the methodology have the following meaning:

- `S_k`: system state at time step `k`
- `p`: static model parameters or configuration values
- `a_k`: external input, disturbance, or control context at time step `k`
- `\Delta t`: transition interval or forecast duration
- `F(.)`: underlying system dynamics
- `y_{k+1}`: next-step output metric required by the outer simulation loop
- `G(.)`: mapping from system state to the returned output metric
- `x_k`: surrogate input built from recent history and contextual features
- `\phi(.)`: feature extraction and history construction function
- `f_\theta`: surrogate model with learned parameters `\theta`
- `\hat{y}_{k+1}`: surrogate prediction of the next-step output metric
- `y_{k+1}^{HF}`: trusted high-fidelity simulator output
- `e_k`: observed prediction error at a checked step
- `P`: set of probe steps at which the surrogate is compared with the high-fidelity simulator
- `W_k`: rolling window of recent probe errors
- `\bar{e}_k`: rolling trust/error estimate used by the switching logic
- `\varepsilon`: switching threshold that determines whether the surrogate remains trusted

## Why the Simulator Is the Ground Truth

The reference simulator is treated as the ground truth because it defines the system behavior that the surrogate is intended to emulate. It preserves internal state, is executed sequentially over time, and generates the output signals used for training and offline reference validation. Although it is still synthetic, it is the authoritative model within the scope of this thesis.

## Evaluation Criteria

The system is evaluated on two axes:

1. Predictive fidelity:
   - MAE
   - RMSE
   - MAPE
   - sMAPE
   - NMAE
   - NRMSE
   - output-trace agreement with the reference simulator
2. Computational efficiency:
   - single-step latency
   - full-run runtime
   - speedup relative to the reference simulator
   - surrogate usage ratio versus fallback simulation usage

## Model Line-up

- Trusted reference model: synthetic microgrid simulator
- Main surrogate: encoder-only Transformer regressor
- Strong sequence baseline: LSTM
- Lower-bound baselines: persistence and linear regression

## Scope Boundary

The current implementation focuses on a Stage 2 microgrid use case with PV, battery, and net-load dynamics. It includes a forecast-index switching evaluation and a component-level decorator architecture. The final reported metrics come from the former; a full component-level quantitative rollout is outside the stabilized result pipeline and is future work.
