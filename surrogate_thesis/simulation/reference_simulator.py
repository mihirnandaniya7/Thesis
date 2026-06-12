"""Synthetic high-fidelity simulator used as the reference system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from surrogate_thesis.config import SimulatorConfig
from surrogate_thesis.simulation.component_interface import (
    ComponentAction,
    ComponentParameters,
    ComponentState,
)


@dataclass(slots=True)
class _SimulatorState:
    """Mutable simulator memory carried from one time step to the next."""

    noise_state: float = 0.0
    event_state_kw: float = 0.0
    inertia_state_kw: float = 0.0
    cloud_state: float = 0.0
    battery_soc_kwh: float = 0.0
    previous_load_kw: float = 0.0


def simulator_parameters_from_config(config: SimulatorConfig) -> ComponentParameters:
    """Convert fixed simulator configuration into the component parameter dictionary."""

    return dict(asdict(config))


def load_component_step(
    state: ComponentState,
    parameters: ComponentParameters,
    action: ComponentAction,
    delta_t: float,
) -> ComponentState:
    """Compute the next load state from explicit state, parameters, and action."""

    timestamp = pd.Timestamp(action["timestamp"])
    hour = float(action.get("hour", timestamp.hour + timestamp.minute / 60))
    is_weekend = bool(action.get("is_weekend", timestamp.dayofweek >= 5))

    base_load_kw = _parameter(parameters, "base_load_kw")
    daily_amplitude_kw = _parameter(parameters, "daily_amplitude_kw")
    morning_peak_kw = _parameter(parameters, "morning_peak_kw")
    evening_peak_kw = _parameter(parameters, "evening_peak_kw")
    weekend_multiplier = _parameter(parameters, "weekend_multiplier")
    weekday_multiplier = _parameter(parameters, "weekday_multiplier")
    seasonal_amplitude = _parameter(parameters, "seasonal_amplitude")
    ar_coefficient = _parameter(parameters, "ar_coefficient")

    # Combine regular daily behavior with sharper morning and evening peaks.
    daily_cycle = 0.55 + 0.45 * np.sin(2 * np.pi * (hour - 6) / 24)
    morning_peak = morning_peak_kw * np.exp(-0.5 * ((hour - 7.5) / 1.2) ** 2)
    evening_peak = evening_peak_kw * np.exp(-0.5 * ((hour - 19.0) / 1.8) ** 2)
    midday_dip = -0.12 * np.exp(-0.5 * ((hour - 13.0) / 2.4) ** 2)
    seasonal_scale = 1.0 + seasonal_amplitude * np.sin(
        2 * np.pi * (timestamp.dayofyear - 30) / 365
    )
    day_multiplier = weekend_multiplier if is_weekend else weekday_multiplier

    # Inertia and autoregressive noise make adjacent samples physically smoother
    # than independent random draws, which is important for sequence forecasting.
    inertia_state_kw = (
        0.82 * _state_value(state, "inertia_state_kw")
        + 0.18 * (0.15 if is_weekend and 9 <= hour <= 22 else 0.05)
    )
    noise_state = ar_coefficient * _state_value(state, "noise_state") + float(
        action.get("noise_kw", 0.0)
    )

    event_state_kw = _state_value(state, "event_state_kw")
    if bool(action.get("peak_event", False)):
        event_state_kw += float(action.get("peak_event_kw", 0.0))
    event_state_kw *= 0.9

    base_component = base_load_kw * seasonal_scale * day_multiplier
    pattern_component = (
        daily_amplitude_kw * daily_cycle
        + morning_peak
        + evening_peak
        + midday_dip
    )
    candidate = (
        base_component
        + pattern_component
        + inertia_state_kw
        + noise_state
        + event_state_kw
    )
    previous_load_kw = _state_value(state, "previous_load_kw", default=base_load_kw)
    # Smooth the candidate load to mimic thermal/device inertia in the microgrid.
    smoothed = 0.62 * max(candidate, 0.05) + 0.38 * previous_load_kw
    load_kw = float(max(smoothed, 0.05))

    return {
        **state,
        "noise_state": float(noise_state),
        "event_state_kw": float(event_state_kw),
        "inertia_state_kw": float(inertia_state_kw),
        "previous_load_kw": load_kw,
        "load_kw": load_kw,
        "delta_t_hours": float(delta_t),
    }


def pv_component_step(
    state: ComponentState,
    parameters: ComponentParameters,
    action: ComponentAction,
    delta_t: float,
) -> ComponentState:
    """Compute photovoltaic output from daylight, seasonality, and cloud state."""

    timestamp = pd.Timestamp(action["timestamp"])
    hour = float(action.get("hour", timestamp.hour + timestamp.minute / 60))
    sunrise = 6.0
    sunset = 19.5

    if not sunrise <= hour <= sunset:
        return {
            **state,
            "pv_kw": 0.0,
            "delta_t_hours": float(delta_t),
        }

    daylight_fraction = (hour - sunrise) / (sunset - sunrise)
    solar_shape = np.sin(np.pi * daylight_fraction)
    seasonal_adjustment = 0.78 + 0.22 * np.sin(
        2 * np.pi * (timestamp.dayofyear - 80) / 365
    )
    cloud_state = 0.85 * _state_value(state, "cloud_state") + 0.15 * float(
        action.get("cloud_disturbance", 0.0)
    )
    cloud_factor = float(np.clip(1.0 - abs(cloud_state), 0.15, 1.0))
    pv_kw = float(
        _parameter(parameters, "pv_peak_kw")
        * solar_shape
        * seasonal_adjustment
        * cloud_factor
    )

    return {
        **state,
        "cloud_state": float(cloud_state),
        "pv_kw": pv_kw,
        "delta_t_hours": float(delta_t),
    }


def battery_component_step(
    state: ComponentState,
    parameters: ComponentParameters,
    action: ComponentAction,
    delta_t: float,
) -> ComponentState:
    """Apply battery charge or discharge to the current net-load balance.

    Input dictionaries follow the agreed component form:
    state + parameters + action --delta_t--> next_state.
    """

    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive for the battery component.")

    capacity = _parameter(parameters, "battery_capacity_kwh")
    power_limit = _parameter(parameters, "battery_power_kw")
    soc = _state_value(state, "battery_soc_kwh", default=capacity * 0.5)
    load_kw = float(action["load_kw"])
    pv_kw = float(action["pv_kw"])
    net_without_battery = load_kw - pv_kw

    charge_kw = 0.0
    discharge_kw = 0.0
    reserve = 0.15 * capacity
    ceiling = 0.95 * capacity

    # The battery discharges during deficits and charges during PV surplus while
    # respecting power, reserve, and capacity limits.
    if net_without_battery > 0.0 and soc > reserve:
        discharge_kw = min(power_limit, net_without_battery, (soc - reserve) / delta_t)
    elif net_without_battery < 0.0 and soc < ceiling:
        charge_kw = min(power_limit, -net_without_battery, (ceiling - soc) / delta_t)

    next_soc = float(
        np.clip(soc - discharge_kw * delta_t + charge_kw * delta_t, 0.0, capacity)
    )
    battery_power_kw = float(discharge_kw - charge_kw)
    net_load_kw = float(net_without_battery - battery_power_kw)

    return {
        **state,
        "battery_soc_kwh": next_soc,
        "battery_power_kw": battery_power_kw,
        "net_load_kw": net_load_kw,
        "delta_t_hours": float(delta_t),
    }


def _parameter(parameters: ComponentParameters, key: str) -> float:
    return float(parameters[key])


def _state_value(state: ComponentState, key: str, default: float = 0.0) -> float:
    return float(state.get(key, default))


class ReferenceSimulator:
    """Sequential simulator used as the thesis ground-truth reference."""

    def __init__(self) -> None:
        self.config: SimulatorConfig | None = None
        self.rng: np.random.Generator | None = None
        self.state = _SimulatorState()

    def reset(self, config: SimulatorConfig, seed: int) -> None:
        """Reset configuration, random generator, and dynamic simulator state."""

        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state = _SimulatorState(
            battery_soc_kwh=config.battery_capacity_kwh * 0.5,
            previous_load_kw=config.base_load_kw,
        )

    def component_state(self) -> ComponentState:
        """Return the explicit state dictionary used by component functions."""

        return {
            "noise_state": float(self.state.noise_state),
            "event_state_kw": float(self.state.event_state_kw),
            "inertia_state_kw": float(self.state.inertia_state_kw),
            "cloud_state": float(self.state.cloud_state),
            "battery_soc_kwh": float(self.state.battery_soc_kwh),
            "previous_load_kw": float(self.state.previous_load_kw),
        }

    def component_parameters(self) -> ComponentParameters:
        """Return fixed configuration as a parameter dictionary."""

        if self.config is None:
            raise RuntimeError("Simulator must be reset before parameters are available.")
        return simulator_parameters_from_config(self.config)

    def replace_component_state(self, next_state: ComponentState) -> None:
        """Update the simulator's object state from an explicit state dictionary."""

        for key in (
            "noise_state",
            "event_state_kw",
            "inertia_state_kw",
            "cloud_state",
            "battery_soc_kwh",
            "previous_load_kw",
        ):
            if key in next_state:
                setattr(self.state, key, float(next_state[key]))

    def generate_series(self, config: SimulatorConfig, seed: int) -> pd.DataFrame:
        """Generate the full chronological simulation frame for one experiment."""

        self.reset(config, seed)
        steps_per_day = int(24 * 60 / config.time_resolution_minutes)
        total_steps = config.days * steps_per_day
        timestamps = pd.date_range(
            start=config.start,
            periods=total_steps,
            freq=f"{config.time_resolution_minutes}min",
        )
        rows: list[dict[str, float | bool | int | str | pd.Timestamp]] = []
        for timestamp in timestamps:
            start = perf_counter()
            row = self._simulate_interval(timestamp)
            # Runtime is stored per external time step so surrogate speedups can
            # be compared against the same test timestamps.
            row["runtime_ms"] = (perf_counter() - start) * 1000
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.insert(0, "timestamp", timestamps)
        return frame

    def _simulate_interval(self, timestamp: pd.Timestamp) -> dict[str, float | bool | int]:
        """Simulate one external time interval, such as a 15-minute sample."""

        if self.config is None or self.rng is None:
            raise RuntimeError("Simulator must be reset before use.")

        config = self.config
        substep_minutes = config.time_resolution_minutes / config.internal_substeps
        substep_hours = substep_minutes / 60

        load_values: list[float] = []
        pv_values: list[float] = []
        battery_values: list[float] = []
        net_values: list[float] = []

        for substep in range(config.internal_substeps):
            substep_timestamp = timestamp + pd.Timedelta(minutes=substep * substep_minutes)
            load_kw = self._simulate_load_substep(substep_timestamp)
            # Stage 2 enriches the target from raw load to net load by adding PV
            # generation and battery dispatch inside the same interval.
            if config.include_stage2_microgrid:
                pv_kw = self._simulate_pv_substep(substep_timestamp)
                battery_kw, net_load_kw = self._apply_battery(
                    load_kw=load_kw,
                    pv_kw=pv_kw,
                    dt_hours=substep_hours,
                )
            else:
                pv_kw = 0.0
                battery_kw = 0.0
                net_load_kw = load_kw

            load_values.append(load_kw)
            pv_values.append(pv_kw)
            battery_values.append(battery_kw)
            net_values.append(net_load_kw)

        day_of_year = int(timestamp.dayofyear)
        hour = timestamp.hour + timestamp.minute / 60
        return {
            "load_kw": float(np.mean(load_values)),
            "pv_kw": float(np.mean(pv_values)),
            "battery_power_kw": float(np.mean(battery_values)),
            "battery_soc_kwh": float(self.state.battery_soc_kwh),
            "net_load_kw": float(np.mean(net_values)),
            "hour": hour,
            "day_of_year": day_of_year,
            "day_of_week": int(timestamp.dayofweek),
            "is_weekend": bool(timestamp.dayofweek >= 5),
        }

    def _simulate_load_substep(self, timestamp: pd.Timestamp) -> float:
        """Simulate load dynamics for one internal substep."""

        if self.config is None or self.rng is None:
            raise RuntimeError("Simulator must be reset before use.")

        config = self.config
        hour = timestamp.hour + timestamp.minute / 60
        event_probability = config.peak_event_probability / max(config.internal_substeps, 1)
        noise_kw = float(self.rng.normal(0.0, config.noise_std_kw))
        # Peak events are rare external disturbances spread across substeps so
        # their daily probability remains stable when internal resolution changes.
        peak_event = float(self.rng.random()) < event_probability
        action: ComponentAction = {
            "timestamp": timestamp,
            "hour": hour,
            "is_weekend": bool(timestamp.dayofweek >= 5),
            "noise_kw": noise_kw,
            "peak_event": peak_event,
        }
        if peak_event:
            action["peak_event_kw"] = config.peak_event_scale_kw * float(
                self.rng.uniform(0.7, 1.3)
            )

        next_state = load_component_step(
            state=self.component_state(),
            parameters=self.component_parameters(),
            action=action,
            delta_t=config.time_resolution_minutes / config.internal_substeps / 60,
        )
        self.replace_component_state(next_state)
        return float(next_state["load_kw"])

    def _simulate_pv_substep(self, timestamp: pd.Timestamp) -> float:
        """Simulate photovoltaic generation for one internal substep."""

        if self.config is None or self.rng is None:
            raise RuntimeError("Simulator must be reset before use.")

        config = self.config
        hour = timestamp.hour + timestamp.minute / 60
        sunrise = 6.0
        sunset = 19.5
        action: ComponentAction = {
            "timestamp": timestamp,
            "hour": hour,
            "cloud_disturbance": (
                float(self.rng.normal(0.0, 0.35)) if sunrise <= hour <= sunset else 0.0
            ),
        }
        next_state = pv_component_step(
            state=self.component_state(),
            parameters=self.component_parameters(),
            action=action,
            delta_t=config.time_resolution_minutes / config.internal_substeps / 60,
        )
        self.replace_component_state(next_state)
        return float(next_state["pv_kw"])

    def _apply_battery(self, load_kw: float, pv_kw: float, dt_hours: float) -> tuple[float, float]:
        """Apply battery dispatch and return battery power plus net load."""

        if self.config is None:
            raise RuntimeError("Simulator must be reset before use.")

        next_state = battery_component_step(
            state=self.component_state(),
            parameters=self.component_parameters(),
            action={"load_kw": load_kw, "pv_kw": pv_kw},
            delta_t=dt_hours,
        )
        self.replace_component_state(next_state)
        return float(next_state["battery_power_kw"]), float(next_state["net_load_kw"])
