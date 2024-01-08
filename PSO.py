import numpy as np
from pyswarm import pso

# Benchmark functions
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt((1/n) * np.sum(x**2))) - \
           np.exp((1/n) * np.sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def griewank(x):
    return (1/4000) * np.sum(x**2) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1)))) + 1

def michalewicz(x, m=10):
    return -np.sum(np.sin(x) * np.sin((np.arange(1, len(x)+1) * x**2) / np.pi)**(2 * m))

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))

def bohachevsky(x):
    return np.sum(x**2) + 2 * np.sum(x[:-1]**2) - 0.3 * np.sum(np.cos(3 * np.pi * x[:-1])) - 0.4 * np.sum(np.cos(4 * np.pi * x[:-1])) + 0.7

def schaffers_f6(x):
    return 0.5 + ((np.sin(np.sqrt(x[0]**2 + x[1]**2))**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2)**2))


# List of benchmark functions
benchmark_functions = [sphere, rastrigin, rosenbrock, ackley, schwefel, griewank, michalewicz, easom, bohachevsky, schaffers_f6]

# Number of dimensions for each test
dimensions = [2, 5, 9]  

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

