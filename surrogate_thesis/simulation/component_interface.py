"""Shared component-transition types for simulator and surrogate components."""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeAlias

# Dictionaries keep the component contract flexible while the thesis prototype
# is still evolving across load, PV, and battery state variables.
ComponentState: TypeAlias = dict[str, Any]
ComponentParameters: TypeAlias = dict[str, Any]
ComponentAction: TypeAlias = dict[str, Any]


class ComponentModel(Protocol):
    """Pure component transition contract.

    The mathematical notation is:

    f(state, parameters, action) --delta_t--> next_state

    In code, delta_t is passed as an argument so both simulator and surrogate
    implementations can use the same callable signature.
    """

    def __call__(
        self,
        state: ComponentState,
        parameters: ComponentParameters,
        action: ComponentAction,
        delta_t: float,
    ) -> ComponentState:
        ...


ComponentStep: TypeAlias = Callable[
    [ComponentState, ComponentParameters, ComponentAction, float],
    ComponentState,
]
