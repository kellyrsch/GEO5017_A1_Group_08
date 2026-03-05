"""Models for linear regression inputs and outputs.
These models exist mainly for readability reasons."""

from dataclasses import dataclass

@dataclass
class GradientDescentConfig:
    learning_rate: float
    max_iterations: int
    min_error_gain: float
    reruns: int = 0
    seed: str | None = None

@dataclass
class LinearRegressionModel:
    weights: list[float]
    intercept: float
    gradient_descent_config: GradientDescentConfig
    residual_sum_of_squares: float