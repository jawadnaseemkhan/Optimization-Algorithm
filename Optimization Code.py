import numpy as np
from pyswarm import pso

class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-5.0, 5.0, dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_value = float('inf')

def objective_function(x):
    # Example objective function: sum of squares
    return sum(x**2)

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

def particle_swarm_optimization(objective_function, num_particles, dim, num_iterations, inertia_weight, c1, c2):
    particles = [Particle(dim) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_value = float('inf')

    for _ in range(num_iterations):
        for particle in particles:
            # Evaluate objective function for the particle
            value = objective_function(particle.position)

            # Update personal best
            if value < particle.best_value:
                particle.best_position = particle.position.copy()
                particle.best_value = value

            # Update global best
            if particle.best_value < global_best_value:
                global_best_position = particle.best_position.copy()
                global_best_value = particle.best_value

        for particle in particles:
            # Update velocities and positions
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            particle.velocity = (
                inertia_weight * particle.velocity +
                c1 * r1 * (particle.best_position - particle.position) +
                c2 * r2 * (global_best_position - particle.position)
            )
            particle.position += particle.velocity

    return global_best_position, global_best_value

if __name__ == "__main__":
    # Set PSO parameters
    num_particles = 30
    dim = 3
    num_iterations = 100
    inertia_weight = 0.7
    c1, c2 = 1.5, 1.5

    # Run PSO
    best_position, best_value = particle_swarm_optimization(objective_function, num_particles, dim, num_iterations, inertia_weight, c1, c2)

    # Print results
    print("Best Position:", max(best_position))
    print("Best Value:", best_value)

benchmark_functions = [sphere, rastrigin, rosenbrock, ackley, schwefel, griewank, michalewicz, easom, bohachevsky, schaffers_f6]

for function in benchmark_functions:
    print(f"Benchmarking {function.__name__} function:")
    for d in range(2, 5):  # Ensure at least two dimensions for easom
        best_values = []
        for _ in range(3):  # Assuming you want to run the optimization 10 times for each dimension
            result = particle_swarm_optimization(function, num_particles, d, num_iterations, inertia_weight, c1, c2)
            best_values.append(result[1])  # Append the best value to the list

        max_result = np.max(best_values)
        print(f"  Dimensions: {d}, Average Best Value: {max_result}")


