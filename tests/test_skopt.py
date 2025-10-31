from skopt import gp_minimize
from skopt.space import Real

# Define your black-box objective function
def objective(params):
    x, y = params  # Example: two float parameters
    return (x - 0.5)**2 + (y - 0.3)**2  # Replace with your function

# Define parameter bounds (all floats between 0 and 1)
space = [Real(0.0, 1.0, name='x'), Real(0.0, 1.0, name='y')]

# Run Bayesian Optimization
result = gp_minimize(objective, space, n_calls=20, random_state=42)

# Best parameters and value
print("Best parameters:", result.x)
print("Best value:", result.fun)