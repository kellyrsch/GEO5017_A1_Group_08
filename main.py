"""Main file to run to get assignment answers."""

from data import KNOWN_POSITIONS
from linear_regression.models import GradientDescentConfig, LinearRegressionModel
from linear_regression.solver import lin_regression_solver

import matplotlib.pyplot as plt

times = [time for time, pos in KNOWN_POSITIONS]
positions = [pos for time, pos in KNOWN_POSITIONS]

x_positions = [pos[0] for pos in positions]
y_positions = [pos[1] for pos in positions]
z_positions = [pos[2] for pos in positions]

def plot_positions(x_pos: list[float], y_pos: list[float], z_pos: list[float]):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_pos, y_pos, z_pos, color='blue', marker='o', markersize=4, label='Object Path')
    for index in range(len(x_pos)):
        ax.text(x_pos[index]+0.125, y_pos[index]+0.125, z_pos[index]+0.125, f"t={index+1}", fontsize=8) # offset text labels for better readability
    return ax

def get_regression_models(polynomial_degree: int, config: GradientDescentConfig) -> tuple[LinearRegressionModel, LinearRegressionModel, LinearRegressionModel]:    
    x_model = lin_regression_solver(list(zip(times, x_positions)),
                                    polynomial_degree=polynomial_degree,
                                    gradient_descent_config=config
                                    )
    y_model = lin_regression_solver(list(zip(times, y_positions)),
                                    polynomial_degree=polynomial_degree,
                                    gradient_descent_config=config
                                    )
    z_model = lin_regression_solver(list(zip(times, z_positions)),
                                    polynomial_degree=polynomial_degree,
                                    gradient_descent_config=config
                                    )
    return (x_model, y_model, z_model)

def question_1_plot_positions():
    plot = plot_positions(x_positions, y_positions, z_positions)
    plot.scatter(x_positions[0], y_positions[0], z_positions[0], color='green', s=100, label='Start')
    plot.scatter(x_positions[-1], y_positions[-1], z_positions[-1], color='red', s=100, label='End')

    plt.show()

def question_2a_model_position_constant_velocity():
    config = GradientDescentConfig(
        learning_rate=0.001,
        max_iterations=10000,
        min_error_gain=1e-10,
        reruns=1,
        seed="fixed_seed_for_reproducibility"
    )
    models = get_regression_models(polynomial_degree=1, config=config)
    velocity = tuple([model.weights[0] for model in models])
    residual_errors = tuple([model.residual_sum_of_squares for model in models])
    return velocity, sum(residual_errors), models[0].gradient_descent_config.learning_rate, models[0].gradient_descent_config.max_iterations

def question_2b_model_position_constant_acceleration():
    config = GradientDescentConfig(
        learning_rate=0.0001,
        max_iterations=450000,
        min_error_gain=1e-10,
        reruns=1,
        seed="fixed_seed_for_reproducibility"
    )
    models = get_regression_models(polynomial_degree=2, config=config)
    acceleration = tuple([model.weights[1] for model in models])
    residual_errors = tuple([model.residual_sum_of_squares for model in models])
    return acceleration, sum(residual_errors), models[0].gradient_descent_config.learning_rate, models[0].gradient_descent_config.max_iterations, models

def _predict_value(model: LinearRegressionModel, time: float) -> float:
    prediction = model.intercept
    for index, weight in enumerate(model.weights):
        prediction += weight * (time ** (index + 1))
    return prediction

def question_2c_model_position_prediction(models):
    time_to_predict = 7
    predicted_position = tuple([_predict_value(model, time_to_predict) for model in models])
    plot = plot_positions(x_positions + [predicted_position[0]], y_positions + [predicted_position[1]], z_positions + [predicted_position[2]])
    
    plot.scatter(x_positions[0], y_positions[0], z_positions[0], color='green', s=75, label='Start')
    plot.scatter(x_positions[-1], y_positions[-1], z_positions[-1], color='red', s=75, label='End (of data)')
    plot.scatter(predicted_position[0], predicted_position[1], predicted_position[2], color='yellow', s=125, label='Prediction')

    plt.show()

    return predicted_position

if __name__ == "__main__":
    print("Question 1: Plotting positions")
    question_1_plot_positions()
    print("-" * 50)
    print("Question 2a: Constant Velocity Model")
    velocity, error, learning_rate, iterations = question_2a_model_position_constant_velocity()
    print(f"Estimated velocity: {velocity}, Residual Error: {error}, Learning Rate: {learning_rate}, Iterations: {iterations}")
    print("-" * 50)
    print("Question 2b: Constant Acceleration Model")
    acceleration, error, learning_rate, iterations, models = question_2b_model_position_constant_acceleration()
    print(f"Estimated acceleration: {acceleration}, Residual Error: {error}, Learning Rate: {learning_rate}, Iterations: {iterations}")
    print("-" * 50)
    print("Question 2c: Constant Acceleration Prediction")
    predicted_position = question_2c_model_position_prediction(models) # re-use the models from 2b to save runtime
    print(f"Predicted position at time 7: {predicted_position}")