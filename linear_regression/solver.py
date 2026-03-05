"""Solver for linear regression problems."""

from gradient_descent.gradient import Value
from gradient_descent.solver import gradient_descent_solver
from linear_regression.models import GradientDescentConfig, LinearRegressionModel

def create_polynomial_features(x: float, degree: int) -> list[float]:
    """Creates unweighted polynomials of x"""
    return [x**(i+1) for i in range(degree)]

def univariate_polynomial_regression_function(weights: list[Value], intercept: Value, x) -> Value:
    """Creates a univariate polynomial regression function of the form:
        f(x) = intercept + w1*x + w2*x^2 + ... + wn"""
    features = create_polynomial_features(x, len(weights))
    prediction = intercept
    for weight, feature in zip(weights, features):
        prediction += weight * feature
    return prediction

def residual_squared_error(regression_function: callable, weights: list[Value], intercept: Value, x, y) -> Value:
    prediction = regression_function(weights, intercept, x)
    return (y - prediction)**2

def residual_sum_of_squares(input_data_matrix: list[tuple],
                            weights: list[Value],
                            intercept: Value,
                            regression_function: callable) -> Value:
    total_error = 0
    for x, y in input_data_matrix:
        total_error += residual_squared_error(regression_function, weights, intercept, x, y)
    return total_error

def lin_regression_solver(input_data_matrix: list[tuple],
                          gradient_descent_config: GradientDescentConfig,
                          polynomial_degree: int = 1,) -> LinearRegressionModel:
    """Solves for the best fit univariate polynomial regression function using gradient descent."""
    weight_ranges = [(-10, 10) for _ in range(polynomial_degree+1)] # in a polynomial equation most weights are assumed to be reasonably small - this logic could potentially be fruther refined / fine-tuned to the data

    error_function = residual_sum_of_squares # could potentially experiment with other error functions

    descent_function = lambda x: error_function(
        input_data_matrix=input_data_matrix,
        weights=x[:-1],
        intercept=x[-1],
        regression_function=univariate_polynomial_regression_function
    ) # this is what will be called by the gradient descent solver.
    # It is agnostic to the function, it simply tries to minimise the output by tweaking the given weights and intercept.
    
    estimated_params = gradient_descent_solver(
        func=descent_function,
        num_estimations=polynomial_degree+1, # +1 for the intercept
        input_ranges=weight_ranges,
        config=gradient_descent_config,
    )

    residual_error = residual_sum_of_squares(input_data_matrix,
                                             [Value(w) for w in estimated_params[:-1]], Value(estimated_params[-1]),
                                             univariate_polynomial_regression_function).val # calculate the final residual error using the estimated parameters
    
    return LinearRegressionModel(
        weights=estimated_params[:-1],
        intercept=estimated_params[-1],
        residual_sum_of_squares=residual_error,
        gradient_descent_config=gradient_descent_config
    )