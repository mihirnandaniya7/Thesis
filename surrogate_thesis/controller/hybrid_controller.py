from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HybridController:
    threshold: float = 0.15
    hysteresis_ratio: float = 1.15
    validation_interval_steps: int = 12
    simulation_cooldown_steps: int = 6
    reentry_probe_interval_steps: int = 3

    @property
    def entry_threshold(self) -> float:
        return self.threshold

    @property
    def exit_threshold(self) -> float:
        return self.threshold * self.hysteresis_ratio

    def decide(self, error_estimate: float, current_mode: str = "simulation") -> str:
        if current_mode == "surrogate":
            return "simulation" if self.should_fallback_to_simulation(error_estimate) else "surrogate"
        return "surrogate" if self.should_enter_surrogate(error_estimate) else "simulation"

    def should_enter_surrogate(self, error_estimate: float) -> bool:
        return error_estimate <= self.entry_threshold

    def should_fallback_to_simulation(self, error_estimate: float) -> bool:
        return error_estimate > self.exit_threshold

    def should_probe(
        self,
        *,
        current_mode: str,
        steps_since_last_observation: int,
        cooldown_remaining: int,
    ) -> bool:
        if current_mode == "surrogate":
            return steps_since_last_observation >= self.validation_interval_steps
        if cooldown_remaining > 0:
            return False
        return steps_since_last_observation >= self.reentry_probe_interval_steps
