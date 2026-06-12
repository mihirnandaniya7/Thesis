"""Decision policy for switching between simulation and surrogate execution."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class HybridController:
    """Encapsulate the trust thresholds used by the runtime decorator.

    The controller does not run a model by itself. It only decides when the
    decorator may use the surrogate and when it must fall back to the reference
    simulator based on the rolling error estimate.
    """

    threshold: float = 0.15
    hysteresis_ratio: float = 1.15
    validation_interval_steps: int = 12
    max_validation_interval_steps: int = 36
    simulation_cooldown_steps: int = 6
    reentry_probe_interval_steps: int = 3

    @property
    def entry_threshold(self) -> float:
        """Maximum rolling error allowed before entering surrogate mode."""

        return self.threshold

    @property
    def exit_threshold(self) -> float:
        """Higher fallback threshold used to avoid rapid mode oscillation."""

        return self.threshold * self.hysteresis_ratio

    def decide(self, error_estimate: float, current_mode: str = "simulation") -> str:
        """Return the next execution mode for the current error estimate."""

        if current_mode == "surrogate":
            return "simulation" if self.should_fallback_to_simulation(error_estimate) else "surrogate"
        return "surrogate" if self.should_enter_surrogate(error_estimate) else "simulation"

    def should_enter_surrogate(self, error_estimate: float) -> bool:
        """Check whether the surrogate is accurate enough to be trusted."""

        return error_estimate <= self.entry_threshold

    def should_fallback_to_simulation(self, error_estimate: float) -> bool:
        """Check whether the observed error requires simulator fallback."""

        return error_estimate > self.exit_threshold

    def should_probe(
        self,
        *,
        current_mode: str,
        steps_since_last_observation: int,
        cooldown_remaining: int,
        error_estimate: float | None = None,
        allow_relaxed_interval: bool = True,
    ) -> bool:
        """Decide whether the decorator should run a validation probe.

        A probe executes both paths so the controller can refresh the rolling
        error estimate without evaluating the simulator on every step.
        """

        if current_mode == "surrogate":
            interval = (
                self.validation_interval_for_error(error_estimate)
                if allow_relaxed_interval
                else max(int(self.validation_interval_steps), 1)
            )
            return steps_since_last_observation >= interval
        if cooldown_remaining > 0:
            return False
        return steps_since_last_observation >= self.reentry_probe_interval_steps

    def validation_interval_for_error(self, error_estimate: float | None) -> int:
        """Adapt the probe interval from the current rolling error estimate."""

        base_interval = max(int(self.validation_interval_steps), 1)
        max_interval = max(int(self.max_validation_interval_steps), base_interval)
        if error_estimate is None or not math.isfinite(error_estimate) or error_estimate <= 0.0:
            return base_interval

        normalized_error = error_estimate / max(self.entry_threshold, 1e-9)
        # Very low error allows longer surrogate-only stretches; near-threshold
        # error keeps validation closer to the base interval.
        if normalized_error <= 0.5:
            return max_interval
        if normalized_error <= 0.8:
            return min(max_interval, base_interval * 2)
        return base_interval
