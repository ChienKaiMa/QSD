import cvxpy as cp
import torch
import numpy as np
import time

if __name__ == "__main__":
    # Example: Minimize (x - p1)^2 + (y - p2)^2 subject to x + y <= c
    # Parameters p1, p2 (affect objective), c (affects constraint)
    p = cp.Parameter(3)  # Parameters [p1, p2, c]
    x = cp.Variable(2)  # Variables [x, y]

    # Objective and constraints
    objective = cp.Minimize(cp.sum_squares(x - p[:2]))
    constraints = [x[0] + x[1] <= p[2]]
    problem = cp.Problem(objective, constraints)

    # Check DPP compliance
    assert problem.is_dpp(), "Problem is not DPP-compliant"

    # Parameter bounds
    lower_bounds = np.array([1.5, 2.5, 4.0])  # Lower bounds for [p1, p2, c]
    upper_bounds = np.array([2.5, 3.5, 6.0])  # Upper bounds for [p1, p2, c]

    # Convert initial parameters to torch tensor
    initial_params = torch.tensor(
        [2.1, 3.1, 5.0], requires_grad=True
    )  # Your good initial solution
    lower_bounds = torch.tensor(lower_bounds)
    upper_bounds = torch.tensor(upper_bounds)

    # AdamW optimizer
    optimizer = torch.optim.AdamW([initial_params], lr=0.01, weight_decay=0.01)

    # Early stopping parameters
    patience = 10
    tol = 1e-4
    best_value = float("inf")
    no_improvement_count = 0
    best_params = initial_params.clone()

    # Optimization loop
    start_time = time.time()
    max_iters = 100
    for i in range(max_iters):
        # Zero gradients
        optimizer.zero_grad()

        # Set CVXPY parameter values
        p.value = initial_params.detach().numpy()

        # Solve the CVXPY problem
        problem.solve(
            solver=cp.SCS,
            warm_start=True,
            # requires_grad=True,
        )  # Use ECOS for small problems; try SCS for warm-start
        if problem.status != cp.OPTIMAL:
            print(f"Iteration {i}: Solver failed with status {problem.status}")
            break

        # Get the objective value and compute gradients
        objective_value = torch.tensor(problem.value, requires_grad=True)
        objective_value.backward()

        # Perform AdamW step
        optimizer.step()

        # Project parameters onto bounds
        with torch.no_grad():
            initial_params.clamp_(min=lower_bounds, max=upper_bounds)

        # Early stopping check
        current_value = problem.value
        if current_value < best_value - tol:
            best_value = current_value
            best_params = initial_params.clone()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {i}. Best value: {best_value}")
            break

    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Best parameters: {best_params.detach().numpy()}")
    print(f"Best objective value: {best_value}")
