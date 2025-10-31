from scipy.optimize import differential_evolution

# Define your black-box objective function
def objective(params):
    x, y = params
    return (x - 0.5)**2 + (y - 0.3)**2  # Replace with your function

# Define bounds for each parameter (0 to 1)
bounds = [(0.0, 1.0), (0.0, 1.0)]

# Run Differential Evolution
result = differential_evolution(objective, bounds, seed=42)

# Best parameters and value
print("Best parameters:", result.x)
print("Best value:", result.fun)