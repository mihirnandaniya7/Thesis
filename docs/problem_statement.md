# Problem Statement

## Title

Surrogate Modeling Based on Machine Learning Approaches for Short-Term Load Forecasting in Microgrids

## Motivation

Microgrid studies often require long-horizon simulation over many time steps while still depending on detailed submodels for short-term operational behavior such as demand variation, photovoltaic generation, battery state-of-charge, and net-load dynamics. These detailed models are more trustworthy than simple approximations, but they are also more computationally expensive to execute repeatedly inside a larger simulation loop.

This creates a practical tension: the outer simulation needs fast repeated evaluations, while the inner model needs sufficient fidelity. A surrogate model can reduce this cost by learning to approximate the expensive model output, but a surrogate should not be trusted blindly. The thesis therefore focuses on both surrogate modeling and adaptive runtime integration.

## Problem Statement

The core problem is how to integrate a fast machine-learning surrogate into a simulation workflow without losing too much reliability. In the current implementation, the outer simulation loop progresses in 15-minute time steps over a long horizon. At each step, a richer reference simulator produces the next-step system response using internal state, time-dependent structure, and microgrid-specific dynamics. Executing this detailed model at every step provides a trusted result, but it becomes inefficient when repeated over long runs.

The thesis addresses this problem by replacing most calls to the reference model with a surrogate model while preserving a path back to the trusted simulator whenever the surrogate becomes unreliable. This requires not only a predictive model, but also a switching mechanism that can decide when to use the surrogate and when to fall back to the high-fidelity reference simulation.

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

The proposed methodology combines offline surrogate learning with online trust-based execution inside a discrete-time simulation loop. The core idea is that the outer simulation requires a next-step metric at every time step, while a decorator decides whether this metric should be produced by the trusted high-fidelity simulator or by a faster surrogate model.

### 1. Discrete-Time Problem Formulation

Let the system evolve over discrete time steps indexed by `k`. In general form, the system transition is expressed as:

`S_{k+1} = F(S_k, a_k)`

where:

- `S_k` denotes the system state at time step `k`
- `a_k` denotes the external inputs, disturbances, or control-relevant context at time step `k`
- `F(.)` denotes the underlying system dynamics

For the outer simulation loop, the full internal state `S_{k+1}` is not necessarily required. Instead, the loop typically requires a returned metric derived from that state. This metric is written as:

`y_{k+1} = G(S_{k+1}) = G(F(S_k, a_k))`

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

This stage yields a trained surrogate candidate `f_\theta` that can subsequently be inserted into the runtime simulation loop.

### 6. Decorator-Managed Runtime Integration

After offline training, runtime execution is organized around a decorator that wraps two alternative implementations of the same functional role:

- the high-fidelity simulator
- the learned surrogate

At every time step `k`, the decorator first decides whether the next-step metric should be produced by the high-fidelity path or by the surrogate path. This decision is therefore the entry point of runtime execution.

If the decorator selects the high-fidelity simulator, the system obtains the trusted output `y_{k+1}^{HF}`. This value is used as the official output for the current step and also provides an additional trusted labeled sample for supervision.

If the decorator selects the surrogate, the system obtains the fast prediction `\hat{y}_{k+1}`, which becomes the provisional output unless a probe check is triggered.

### 7. Probe-Based Trust Estimation

The surrogate is not compared against the high-fidelity simulator at every time step. Instead, trust is estimated from a subset of time steps used as probes. Let `P` denote the set of probe indices. For `k \in P`, the trusted simulator is executed in addition to the surrogate, and the observed error is computed as:

`e_k = || \hat{y}_{k+1} - y_{k+1}^{HF} || , \quad k \in P`

This makes explicit that the prediction error is only directly observable on probe steps or on steps where the high-fidelity path is already active.

To obtain a stable trust signal, the system maintains a rolling error estimate over the most recent probe steps. Let `W_k` denote the rolling probe window. Then:

`\bar{e}_k = \frac{1}{|W_k|} \sum_{i \in W_k} e_i`

The value `\bar{e}_k` is the main trust indicator used by the switching logic. In this way, the surrogate is supervised during runtime without forcing the high-fidelity simulator to run at every time step.

### 8. Switching Policy

The decorator follows an adaptive threshold-based switching policy with warmup, fallback, and re-entry behavior.

During the initial warmup phase, the high-fidelity simulator is used more frequently in order to establish an initial trust history. After warmup, the surrogate is preferred whenever the rolling error estimate remains within the accepted trust region:

`\bar{e}_k \leq \varepsilon`

If the rolling error exceeds the threshold, the system falls back to the high-fidelity simulator:

`\bar{e}_k > \varepsilon`

In the current implementation, this policy is refined through:

- periodic validation probes
- rolling error tracking
- fallback execution under low trust
- cooldown and re-entry checks
- hysteresis to avoid unstable rapid switching

Therefore, the switching mechanism is a continuous runtime supervision policy rather than a one-time model choice.

### 9. Relation to Training Continuation

Whenever the high-fidelity simulator is executed, the system obtains an additional trusted labeled output. These outputs support continued supervision of the surrogate. In the current prototype, the decorator-managed runtime layer already supports lightweight online recalibration based on these trusted labels, while the same mechanism also provides a basis for future retraining or more extensive online adaptation. Because the decorator determines when new trusted labels become available, it also governs when supervised updating can occur.

### 10. Relation to Multi-Scale Simulation

The broader architectural motivation is multi-scale simulation. In such settings, an outer simulation loop may operate on a coarser decision timescale, while an inner high-fidelity model may require finer internal computation. The surrogate becomes especially valuable when the outer simulation only needs the returned metric `y_{k+1}`, rather than the full internal trajectory of the detailed inner model.

The current implementation demonstrates this principle on a sequential microgrid forecasting task by learning the returned next-step metric directly. This is important because the largest computational gain is achieved when the surrogate predicts the quantity actually required by the outer loop.

### 11. Evaluation Methodology

The final stage evaluates the resulting system along two complementary axes:

1. predictive fidelity relative to the high-fidelity simulator
2. computational efficiency relative to simulator-only execution

The evaluation includes:

- MAE
- RMSE
- sMAPE
- runtime per step
- full-run runtime
- speedup
- surrogate usage ratio
- fallback simulation ratio

Overall, the methodological pipeline can be summarized as:

reference simulation -> dataset construction -> surrogate training -> decorator-managed runtime execution -> probe-based trust update -> system-level evaluation

### 12. Formula Summary

For clarity, the main symbols used in the methodology have the following meaning:

- `S_k`: system state at time step `k`
- `a_k`: external input, disturbance, or control context at time step `k`
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

The reference simulator is treated as the ground truth because it defines the system behavior that the surrogate is intended to emulate. It preserves internal state, is executed sequentially over time, and generates the output signals used both for training and for runtime validation. Although it is still synthetic, it is the authoritative model within the scope of this thesis.

## Evaluation Criteria

The system is evaluated on two axes:

1. Predictive fidelity:
   - MAE
   - RMSE
   - sMAPE
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

The current implementation focuses on a Stage 2 microgrid use case with PV, battery, and net-load dynamics. The implementation already includes a decorator-based switching layer, but the main thesis task now is to present the architecture, workflow, and multiscale simulation role of this switching logic more clearly.
