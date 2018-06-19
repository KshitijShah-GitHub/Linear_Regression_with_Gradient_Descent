# import required modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import random

# Generate random linear equation in form of y = m*x + b to form data with
slope: float = random.uniform(-10.0, 10.0)
y_int: float = random.uniform(-10.0, 10.0)
seed_line = "Seed function: y = " + str(round(slope, 2)) + "x + " + \
            str(round(y_int, 2))

# Generate array of points at random distance from generated line
x: 'array' = np.array([round(i - random.uniform(-10.0, 10.0), 2) for i in
                      range(0, 50)])
y: 'array' = np.array([round(i * slope + y_int +
                      random.uniform(-30.0, 30.0), 2) for i in x])

# Plot data points on grid for better visulalization
plt.plot(x, y, 'bo', markersize=2)
plt.xlabel("input (x)")
plt.ylabel("label (y)")
min_x, max_x = min(x), max(x)

# This is convenient shortening of a commonly used value
n: int = len(x)

# Plot line of best fit with linear regression equation to see optimal answer
# y = coeff * x + const
# Use least square fit method to find correct linear equation, useful to
# compare to grad desc progress later
coeff: float = \
 ((n * np.sum([i * j for i, j in zip(x, y)])) -
  (sum(x) * sum(y))) / ((n * np.sum([i ** 2 for i in x])) - (sum(x) ** 2))
const: float = round(((sum(y)/n) - coeff * (sum(x)/n)), 2)
coeff: float = round(coeff, 2)
lobf_cost: float = round((1 / n) * np.sum([((y_obs - (x_obs * coeff + const))
                                          ** 2) for x_obs, y_obs in
                                          zip(x, y)]), 2)

theoretical_lobf = "Actual Line of Best Fit: y = " + str(coeff) + \
                   "x + " + str(const) + " cost = " + str(lobf_cost)
print(seed_line + "\n" + theoretical_lobf)

# plot line of best fit
line1 = plt.plot([min_x, max_x], [(min_x * coeff) + const, (max_x * coeff) +
                                  const], label="Theoretical Line of Best Fit")
line2 = plt.plot([min_x, max_x], [(min_x * slope) + y_int, (max_x * slope) +
                                  y_int], label="Seed Line")
plt.legend()
plt.show()


# Find mean squared error (L2 loss => | || \n || |-- get it? its the loss meme)
def cost_function(x: 'array', y: 'array', weight: float, bias: float) -> float:
    # weight * x_i_observed + bias = y_i_predicted
    # cost = 1/n * sum_{0}^{n-1} ((y_i_observed - y_i_predicted) ^ 2)
    n = len(x)
    cost = (1 / n) * np.sum([((y_obs - (x_obs * weight + bias)) ** 2)
                             for x_obs, y_obs in zip(x, y)])

    return cost


# do the real stuff - gradient descent function: default learning rate to 0.01
def gradient_descent(x: 'array', y: 'array', weight: float, bias: float,
                     learning_rate: float) -> Tuple[float]:
    w_grad_asc: float = 0  # partial derivative (deriv func wrt weight)
    b_grad_asc: float = 0  # partial derivative (deriv func wrt bias)
    n = len(x)

    # y_i_predicted = weight * x_i_observed + bias
    # w gradient ascent:
    # (df/dw) = (2 / n) * sum_{i = 0} ^ {n - 1} (x_i_observed *
    # (y_i_observed - y_i_predicted))
    w_grad_asc = (-2 / n) * np.sum([(x_obs * (y_obs - (weight * x_obs +
                                    bias))) for x_obs, y_obs in zip(x, y)])

    # b gradient ascent (df/db) = (2 / n) * sum_{i = 0} ^ {n - 1}
    # (y_i_observed - y_i_predicted)
    b_grad_asc = (-2 / n) * np.sum([(y_obs - (weight * x_obs +
                                    bias)) for x_obs, y_obs in zip(x, y)])

    # we will need to subtract (descent => -ascent)
    # then multiply by learning rate
    weight -= w_grad_asc * learning_rate
    bias -= b_grad_asc * learning_rate

    return weight, bias


# feed data through grad desc over many iterations to update values
def train(x: 'array', y: 'array', weight: float, bias: float,
          learning_rate: float, epochs: int) -> Tuple['array']:
    # To log progress of training
    costs: 'array' = np.zeros(epochs)
    weights: 'array' = np.zeros(epochs)
    biases: 'array' = np.zeros(epochs)

    for i in range(epochs):
        # update values over iterations based on reducing cost
        weight, bias = gradient_descent(x, y, weight, bias, learning_rate)
        weights[i] = weight
        biases[i] = bias
        costs[i] = cost_function(x, y, weight, bias)

    return weights, biases, costs


# train to optimize to the generated data set
weight: float = 10.0  # random init value
bias: float = 10.0  # random init value
learning_rate: float = 0.001  # generic learning_rate
epochs: int = 20000
# number of iterations to train data with, more => better more accurate value,
# possible overfitting
weights, biases, costs = train(x, y, weight, bias, learning_rate, epochs)
weight: float = round(weights[-1], 2)
bias: float = round(biases[-1], 2)
cost: float = round(costs[-1], 2)

learned_line: str = "Learned Line of Best Fit: y = " + str(weight) + "x + " + \
                    str(bias) + ", cost = " + str(cost)
print(seed_line + "\n" + theoretical_lobf + "\n" + learned_line)

# Plot cost reduction over iterations (only few iters where cost drop is steep)
div = 40
shift = int(div / 10)
plt.plot(range(int(epochs / div - shift)), [costs[div * shift:][i] for i in
                                            range(len(costs[div * shift:]))
                                            if i % div == 0])
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Reduction Over Iterations')
plt.show()


# This is essentially the same computation as the cost function (MSE)
def MSE(w: 'array', b: 'array'):
    n = len(x)
    return (1/n) * sum([((y[i] - (w * x[i] + b)) ** 2) for i in range(n)])


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

min_w, max_w = min(weights), max(weights)
min_b, max_b = min(biases), max(biases)

# Generate array for weights
w = np.linspace(int(min_w) - 1.5, int(max_w) + 1.5, 100)
# Generate array for biases
b = np.linspace(int(min_b) - 1, int(max_b) + 1, 100)
wx, by = np.meshgrid(w, b)  # Meshgrid of weights and biases (x vals, y vals)
cz = MSE(wx, by)  # Get cost for meshgrid of weights and biases (z vals)

ax.set_title("Plot of Gradient Descent Progress over $n$ Iterations")
ax.set_xlabel("Weights")
ax.set_ylabel("Biases")
ax.set_zlabel("Costs")

# Surface of gradient
ax.plot_surface(wx, by, cz, cmap="Spectral", alpha=0.8, label="Cost function")
# Progress over training
ax.plot(weights, biases, costs, 'ob-', markersize=2, alpha=0.7,
        label="Gradient descent")
# Goal, least square fit minimum
ax.plot([coeff], [const], [lobf_cost], 'r*', markersize=8,
        label="Found Minimum")

plt.show()
