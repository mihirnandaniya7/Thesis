from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HybridController:
    threshold: float = 0.15

    def decide(self, error_estimate: float) -> str:
        return "surrogate" if error_estimate <= self.threshold else "simulation"

