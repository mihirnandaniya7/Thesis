"""Controller and decorator interfaces for runtime surrogate switching."""

from surrogate_thesis.controller.hybrid_controller import HybridController
from surrogate_thesis.controller.surrogate_decorator import (
    ComponentModelAdapter,
    ComponentStepProvider,
    ComponentStepResult,
    ComponentSurrogateDecorator,
    DecoratorEvaluationArtifacts,
    DecoratorEvaluationRunner,
    ForecastProvider,
    ForecastProviderDecorator,
    ForecastResult,
    HighFidelitySimulationAdapter,
    RecalibratingForecastDecorator,
    RollingErrorTracker,
    SurrogateModelAdapter,
    SurrogateDecorator,
    evaluate_decorator_thresholds,
)

__all__ = [
    "HybridController",
    "ComponentModelAdapter",
    "ComponentStepProvider",
    "ComponentStepResult",
    "ComponentSurrogateDecorator",
    "ForecastProvider",
    "ForecastProviderDecorator",
    "ForecastResult",
    "HighFidelitySimulationAdapter",
    "SurrogateModelAdapter",
    "RecalibratingForecastDecorator",
    "RollingErrorTracker",
    "SurrogateDecorator",
    "DecoratorEvaluationRunner",
    "DecoratorEvaluationArtifacts",
    "evaluate_decorator_thresholds",
]
