from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from surrogate_thesis.config import SimulatorConfig


@dataclass(slots=True)
class _SimulatorState:
    noise_state: float = 0.0
    event_state_kw: float = 0.0
    inertia_state_kw: float = 0.0
    cloud_state: float = 0.0
    battery_soc_kwh: float = 0.0
    previous_load_kw: float = 0.0


class ReferenceSimulator:
    """Sequential synthetic load simulator used as the thesis ground truth."""

    def __init__(self) -> None:
        self.config: SimulatorConfig | None = None
        self.rng: np.random.Generator | None = None
        self.state = _SimulatorState()

    def reset(self, config: SimulatorConfig, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state = _SimulatorState(
            battery_soc_kwh=config.battery_capacity_kwh * 0.5,
            previous_load_kw=config.base_load_kw,
        )

    def generate_series(self, config: SimulatorConfig, seed: int) -> pd.DataFrame:
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
            row["runtime_ms"] = (perf_counter() - start) * 1000
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.insert(0, "timestamp", timestamps)
        return frame

    def _simulate_interval(self, timestamp: pd.Timestamp) -> dict[str, float | bool | int]:
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
        if self.config is None or self.rng is None:
            raise RuntimeError("Simulator must be reset before use.")

        config = self.config
        hour = timestamp.hour + timestamp.minute / 60
        is_weekend = timestamp.dayofweek >= 5

        daily_cycle = 0.55 + 0.45 * np.sin(2 * np.pi * (hour - 6) / 24)
        morning_peak = config.morning_peak_kw * np.exp(-0.5 * ((hour - 7.5) / 1.2) ** 2)
        evening_peak = config.evening_peak_kw * np.exp(-0.5 * ((hour - 19.0) / 1.8) ** 2)
        midday_dip = -0.12 * np.exp(-0.5 * ((hour - 13.0) / 2.4) ** 2)
        seasonal_scale = 1.0 + config.seasonal_amplitude * np.sin(
            2 * np.pi * (timestamp.dayofyear - 30) / 365
        )
        day_multiplier = config.weekend_multiplier if is_weekend else config.weekday_multiplier

        self.state.inertia_state_kw = (
            0.82 * self.state.inertia_state_kw
            + 0.18 * (0.15 if is_weekend and 9 <= hour <= 22 else 0.05)
        )
        self.state.noise_state = (
            config.ar_coefficient * self.state.noise_state
            + float(self.rng.normal(0.0, config.noise_std_kw))
        )

        event_probability = config.peak_event_probability / max(config.internal_substeps, 1)
        if float(self.rng.random()) < event_probability:
            jump = config.peak_event_scale_kw * float(self.rng.uniform(0.7, 1.3))
            self.state.event_state_kw += jump
        self.state.event_state_kw *= 0.9

        base_component = config.base_load_kw * seasonal_scale * day_multiplier
        pattern_component = config.daily_amplitude_kw * daily_cycle + morning_peak + evening_peak + midday_dip
        candidate = (
            base_component
            + pattern_component
            + self.state.inertia_state_kw
            + self.state.noise_state
            + self.state.event_state_kw
        )
        smoothed = 0.62 * max(candidate, 0.05) + 0.38 * self.state.previous_load_kw
        self.state.previous_load_kw = smoothed
        return float(max(smoothed, 0.05))

    def _simulate_pv_substep(self, timestamp: pd.Timestamp) -> float:
        if self.config is None or self.rng is None:
            raise RuntimeError("Simulator must be reset before use.")

        config = self.config
        hour = timestamp.hour + timestamp.minute / 60
        sunrise = 6.0
        sunset = 19.5
        if not sunrise <= hour <= sunset:
            return 0.0

        daylight_fraction = (hour - sunrise) / (sunset - sunrise)
        solar_shape = np.sin(np.pi * daylight_fraction)
        seasonal_adjustment = 0.78 + 0.22 * np.sin(2 * np.pi * (timestamp.dayofyear - 80) / 365)
        self.state.cloud_state = 0.85 * self.state.cloud_state + 0.15 * float(
            self.rng.normal(0.0, 0.35)
        )
        cloud_factor = float(np.clip(1.0 - abs(self.state.cloud_state), 0.15, 1.0))
        return float(config.pv_peak_kw * solar_shape * seasonal_adjustment * cloud_factor)

    def _apply_battery(self, load_kw: float, pv_kw: float, dt_hours: float) -> tuple[float, float]:
        if self.config is None:
            raise RuntimeError("Simulator must be reset before use.")

        config = self.config
        soc = self.state.battery_soc_kwh
        capacity = config.battery_capacity_kwh
        power_limit = config.battery_power_kw
        net_without_battery = load_kw - pv_kw

        charge_kw = 0.0
        discharge_kw = 0.0
        reserve = 0.15 * capacity
        ceiling = 0.95 * capacity

        if net_without_battery > 0.0 and soc > reserve:
            discharge_kw = min(power_limit, net_without_battery, (soc - reserve) / dt_hours)
        elif net_without_battery < 0.0 and soc < ceiling:
            charge_kw = min(power_limit, -net_without_battery, (ceiling - soc) / dt_hours)

        self.state.battery_soc_kwh = float(
            np.clip(soc - discharge_kw * dt_hours + charge_kw * dt_hours, 0.0, capacity)
        )
        battery_power_kw = discharge_kw - charge_kw
        net_load_kw = net_without_battery - battery_power_kw
        return float(battery_power_kw), float(net_load_kw)

