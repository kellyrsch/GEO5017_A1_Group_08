import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import shape

P = np.array([[2.0, 0.0, 1.0],
             [1.08, 1.68, 2.38],
             [-0.83, 1.82, 2.49],
             [-1.97, 0.28, 2.15],
              [-1.31, -1.51, 2.59],
              [0.57, -1.91, 4.32]])

T = np.array([1, 2, 3, 4, 5, 6])

print("P: ", P)
print("T: ", T)

# Extract the coordinates
x = P[:, 0]
y = P[:, 1]
z = P[:, 2]

print("x: ", x)
print("y: ", y)
print("z: ", z)

## Plot tracked points
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(x, y, z)
# ax.scatter(x, y, z, c=T, cmap='viridis')
# for i in range(len(T)):
#     ax.text(x[i], y[i], z[i], str(T[i]))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# #plt.show()


## Gradient Descent Solver function
def gradient_descent_solver(D, p, learning_rate, max_iterations, tolerance=0.01):
    """
    Function to compute the gradient descent.
    :param D: matrix to be initialised earlier in the form [1, T]
    :param p: one dimension of the tracked positions (x, y, z)
    :param learning_rate:
    :param max_iterations:
    :param tolerance: set by default to 0.01
    :return: matrix with optimised parameters
    """
    deg = np.shape(D)[1] #determine the degree
    A = np.zeros(deg) #initialise the matrix with parameters to optimise (starting point)
    steps = [A] #history tracking

    for i in range(max_iterations):
        predicted = D @ A #predicted values with current parameters
        err = predicted - p #error vector of predicted values compared to measured values
        gradient = 2 * D.T @ err #compute the gradient
        diff = learning_rate * gradient
        if np.all(np.abs(diff))<tolerance: #stop the loop if the difference is smaller than given tolerance
            break
        A = A - diff #update the parameters
        steps.append(A) #history tracking
    return A


### Constant velocity
# Initialise matrix D with [1, T]
#D = np.ones((len(T), 1))
D = np.column_stack([np.ones(len(T)), T])
#print("D: ", D)


# Set learning rate and max number of iterations
learning_rate = 0.001
max_i = 10000

## Drone flies with constant velocity, equation in the form: p(t) = a0 + a1*t
# calculate optimised parameters for each dimension
params_x = gradient_descent_solver(D, x, learning_rate, max_i)
params_y = gradient_descent_solver(D, y, learning_rate, max_i)
params_z = gradient_descent_solver(D, z, learning_rate, max_i)
# extract parameter that represents velocity
v_x = params_x[1]
v_y = params_y[1]
v_z = params_z[1]
# make one velocity array
velocity = np.array([v_x, v_y, v_z])
print("velocity: ", velocity)

# Compute the residual error (SSE) of the estimated positions
# compute the predicted position for each dimension
predicted_x = D @ params_x
predicted_y = D @ params_y
predicted_z = D @ params_z
# compute SSE between predicted positions and tracked positions
SSE_x = np.sum((x - predicted_x)**2)
SSE_y = np.sum((y - predicted_y)**2)
SSE_z = np.sum((z - predicted_z)**2)
# total SSE
SSE = SSE_x + SSE_y + SSE_z
print("SSE for constant velocity is: ", SSE)


### Constant acceleration
# Initialise matrix D with [1, T, T**2]
D2 = np.column_stack([np.ones(len(T)), T, T**2])
#print("D2: ", D2)


# Set learning rate and max number of iterations
learning_rate2 = 0.0001
max_i2 = 450000

## Drone flies with constant velocity, equation in the form: p(t) = a0 + a1*t
# calculate optimised parameters for each dimension
params_x2 = gradient_descent_solver(D2, x, learning_rate2, max_i2)
params_y2 = gradient_descent_solver(D2, y, learning_rate2, max_i2)
params_z2 = gradient_descent_solver(D2, z, learning_rate2, max_i2)

# Compute the residual error (SSE) of the estimated positions
# compute the predicted position for each dimension
predicted_x2 = D2 @ params_x2
predicted_y2 = D2 @ params_y2
predicted_z2 = D2 @ params_z2
# compute SSE between predicted positions and tracked positions
SSE_x2 = np.sum((x - predicted_x2)**2)
SSE_y2 = np.sum((y - predicted_y2)**2)
SSE_z2 = np.sum((z - predicted_z2)**2)
# total SSE
SSE2 = SSE_x2 + SSE_y2 + SSE_z2
print("SSE for constant acceleration is: ", SSE2)


## Compute the drone's most likely next position at t=7
# initialise matrix D with t=7
D_t7 = np.array([1, 7, 7**2])
# compute the predicted position for each dimension
x7 = D_t7 @ params_x2
y7 = D_t7 @ params_y2
z7 = D_t7 @ params_z2

p7 = np.array([x7, y7, z7])
print("predicted next point: ", p7)

# plot the new position together with the previous ones
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z)
ax.scatter(x, y, z, c=T, cmap='viridis')
for i in range(len(T)):
    ax.text(x[i], y[i], z[i], str(T[i]))
ax.scatter(x7, y7, z7, c='red', marker='*')
ax.text(x7, y7, z7, 't=7')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()