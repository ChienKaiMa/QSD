import cvxpy as cp
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Problem setup
n = 10  # Dimension of decision variable
m = 5   # Number of matrices/output probabilities
k = 3   # Number of parameters

# Target probability distribution
target_dist = torch.softmax(torch.randn(m), dim=0).numpy()

def define_cvxpy_problem():
    """Define the DPP-compliant CVXPY problem for the inner loop."""
    theta = cp.Parameter(k)
    A_matrices = [cp.Parameter((n, n)) for _ in range(m)]
    x = cp.Variable(n)
    success_rates = [cp.matmul(A_matrices[i], x) for i in range(m)]
    success_rate_sum = cp.sum([cp.sum(success_rates[i]) for i in range(m)])
    objective = cp.Maximize(success_rate_sum)
    constraints = [cp.sum(x) == 1, x >= 0]
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), "Problem is not DPP-compliant"
    return problem, theta, A_matrices, success_rates

def generate_A_matrices(theta_val):
    """Generate parametrized matrices based on theta."""
    # Ensure theta_val is a numpy array for matrix generation
    theta_val = theta_val.detach().numpy() if isinstance(theta_val, torch.Tensor) else theta_val
    base = np.eye(n) * theta_val[0] + np.ones((n, n)) * theta_val[1]
    return [base + theta_val[2] * np.random.randn(n, n) for _ in range(m)]

def kl_divergence(p, q):
    """Compute KL divergence between distributions p and q."""
    return torch.sum(p * torch.log(p / q))

import diffcp  # Ensure diffcp is imported

def optimize_parameters(initial_theta, lower_bounds, upper_bounds, max_iters=100, lr=0.1, weight_decay=0.01):
    """Optimize parameters using AdamW with data collection."""
    initial_theta = torch.tensor(initial_theta, dtype=torch.float32, requires_grad=True)
    lower_bounds = torch.tensor(lower_bounds, dtype=torch.float32)
    upper_bounds = torch.tensor(upper_bounds, dtype=torch.float32)

    optimizer = torch.optim.AdamW([initial_theta], lr=lr, weight_decay=weight_decay)
    problem, cp_theta, A_matrices, success_rates = define_cvxpy_problem()

    # Custom layer for diffcp
    class CVXPYDiffLayer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta_tensor):
            theta_np = theta_tensor.detach().numpy()
            cp_theta.value = theta_np
            for A, A_val in zip(A_matrices, generate_A_matrices(theta_np)):
                A.value = A_val
            problem.solve(solver=cp.SCS, requires_grad=True)
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"CVXPY failed with status {problem.status}")
            success_sums = torch.tensor(
                [success_rates[i].value.sum() for i in range(m)],
                dtype=torch.float32,
                requires_grad=True
            )
            ctx.save_for_backward(theta_tensor, success_sums)
            ctx.problem = problem
            return success_sums

        @staticmethod
        def backward(ctx, grad_output):
            theta_tensor, _ = ctx.saved_tensors
            problem = ctx.problem
            # Use diffcp to compute gradients
            problem.backward()
            grad_theta = torch.tensor(cp_theta.gradient, dtype=torch.float32)
            return grad_theta

    data = {
        'iteration': [],
        'theta': [],
        'kl_divergence': [],
        'derived_dist': []
    }

    best_value = float('inf')
    best_theta = initial_theta.clone()

    for i in range(max_iters):
        optimizer.zero_grad()

        # Forward pass through diffcp layer
        success_sums = CVXPYDiffLayer.apply(initial_theta)
        derived_dist = torch.softmax(success_sums, dim=0)
        target_dist_torch = torch.tensor(target_dist, dtype=torch.float32)

        # Outer loop objective
        outer_objective = kl_divergence(derived_dist, target_dist_torch)
        outer_objective.backward()

        # Debug gradients
        if initial_theta.grad is None or torch.all(initial_theta.grad == 0):
            print(f"Iteration {i}: No gradients for theta: {initial_theta.grad}")
        else:
            print(f"Iteration {i}: Gradients: {initial_theta.grad}")

        # AdamW step
        optimizer.step()

        # Project parameters
        with torch.no_grad():
            initial_theta.clamp_(min=lower_bounds, max=upper_bounds)

        # Collect data
        data['iteration'].append(i)
        data['theta'].append(initial_theta.detach().numpy().copy())
        data['kl_divergence'].append(outer_objective.item())
        data['derived_dist'].append(derived_dist.detach().numpy().copy())

        # Track best solution
        current_value = outer_objective.item()
        if current_value < best_value:
            best_value = current_value
            best_theta = initial_theta.clone()

    data_df = pd.DataFrame({
        'iteration': data['iteration'],
        'theta_0': [t[0] for t in data['theta']],
        'theta_1': [t[1] for t in data['theta']],
        'theta_2': [t[2] for t in data['theta']],
        'kl_divergence': data['kl_divergence'],
        'derived_dist': data['derived_dist']
    })

    return data_df, best_theta.detach().numpy(), best_value

def plot_optimization_data(df):
    """Plot optimization results."""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df['iteration'], df['kl_divergence'], label='KL Divergence')
    plt.xlabel('Iteration')
    plt.ylabel('KL Divergence')
    plt.title('Outer Loop Objective Convergence')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for i in range(k):
        plt.plot(df['iteration'], df[f'theta_{i}'], label=f'theta_{i}')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Trajectories')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for i in range(m):
        plt.plot(df['iteration'], [d[i] for d in df['derived_dist']], label=f'Derived Dist[{i}]')
        plt.axhline(target_dist[i], linestyle='--', label=f'Target Dist[{i}]')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.title('Convergence of Probability Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    lower_bounds = np.array([0.0, 0.0, 0.0])
    upper_bounds = np.array([1.0, 1.0, 1.0])
    initial_theta = np.array([0.5, 0.5, 0.5])

    start_time = time.time()
    data_df, best_theta, best_value = optimize_parameters(
        initial_theta, lower_bounds, upper_bounds, max_iters=100, lr=0.01, weight_decay=0.01
    )

    data_df.to_csv('optimization_data.csv', index=False)
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Best parameters: {best_theta}")
    print(f"Best KL divergence: {best_value}")
    plot_optimization_data(data_df)