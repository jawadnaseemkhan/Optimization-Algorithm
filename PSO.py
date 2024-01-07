import numpy as np
from pyswarm import pso

# Benchmark functions
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Add more benchmark functions as needed

# List of benchmark functions
benchmark_functions = [sphere, rastrigin, rosenbrock]

# Number of dimensions for each test
dimensions = [2, 5, 10]  # Add more dimensions as needed

# Number of runs for each test
num_runs = 5

# Particle Swarm Optimization
def pso_optimizer(objective_function, dimension):
    lb = [-5.12] * dimension  # Lower bounds for variables
    ub = [5.12] * dimension   # Upper bounds for variables

    def objective_function_wrapper(x):
        return objective_function(x)

    best_results = []

    for _ in range(num_runs):
        best_position, _ = pso(objective_function_wrapper, lb, ub)
        best_value = objective_function(best_position)
        best_results.append(best_value)

    return np.mean(best_results)

# Run benchmark tests
for function in benchmark_functions:
    print(f"Benchmarking {function.__name__} function:")
    for dim in dimensions:
        result = pso_optimizer(function, dim)
        print(f"  Dimensions: {dim}, Average Best Value: {result}")

