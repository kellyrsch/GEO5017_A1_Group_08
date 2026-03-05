"""This module contains the implementation of a gradient descent solver that can be used to find the minimum of a given function.
It is designed to be flexible and can be applied to various optimization problems, including linear regression."""

import logging
import random
import string

from gradient_descent.gradient import Value
from linear_regression.models import GradientDescentConfig

def gradient_descent_solver(func: callable,
                            num_estimations: int,
                            config: GradientDescentConfig,
                            input_ranges: list[tuple[float, float]] | None = None) -> list[float]:
    random.seed(config.seed) # use a seed to be able to exactly reproduce results
    attempt_values = {}
    for attempt in range(config.reruns + 1): # only relevant for polynomials with a non-convex function, where the solver might get stuck in a local minima. In such cases we can simply re-run the solver with a different seed to get a different starting point, and therefore potentially find a better minima.
        new_seed = ''.join(random.choices(string.ascii_uppercase + string.digits, k=25))
        random.seed(new_seed)
        seeded_vals = []
        for input_index in range(num_estimations):
            if input_ranges is not None and len(input_ranges) > input_index:
                min_val, max_val = input_ranges[input_index]
                sigma = (max_val - min_val) / 6
                median = min_val + (max_val - min_val) / 2
                val = random.gauss(median, sigma) # make a random guess based on the provided range
            else:
                val = random.gauss() # make a completely random guess if no range is provided
            seeded_vals.append(Value(val))
        new_error, new_inputs = _attempt_gradient_descent(func(seeded_vals), seeded_vals, config) # attempt gradient descent with the seeded values and get the resulting error and new input values after descent
        attempt_values[new_error] = new_inputs
    best_attempt = attempt_values[min([k for k in attempt_values.keys()])] # choose best result from all attempts based on the lowest error
    return [i.val for i in best_attempt]

def _attempt_gradient_descent(func_out: Value,
                              inputs: list[Value],
                              config: GradientDescentConfig) -> tuple[Value, list[Value]]:
    iterations = 0
    current_val = func_out.val
    input_labels = [chr(97 + i) for i in range(len(inputs))]  # a, b, c, ... (this is only for debugging)
    input_str = ", ".join([f"{label}={inputs[i].val:.5f}" for i, label in enumerate(input_labels)]) # (also only relevant for debugging)
    logging.debug(f"Starting gradient descent for {input_str}. Current Output: {current_val:.3f}")
    while iterations < config.max_iterations:
        try:
            func_out.backpropagate() # calculate gradients for weights
        except OverflowError as e:
            logging.debug(f"OverflowError during backpropagation: {e}. Ending this attempt.")
            break
        for i in inputs:
            i_change = i.gradient * config.learning_rate # get step size
            i.val -= i_change # apply step
        func_out.recalculate() # get new error
        iterations += 1
        gain = current_val - func_out.val
        # check if we have converged
        if abs(gain) < config.min_error_gain:
            logging.debug(f"Iteration {iterations}: Output change = {gain:.6f}, breaking")
            break
        current_val = func_out.val
        input_str = ", ".join([f"{label}={inputs[i].val:.5f}" for i, label in enumerate(input_labels)])
        logging.debug(f"Iteration {iterations} | {input_str} | Output: {current_val:.3f}")
    return func_out, inputs