import math
import numpy as np


def entropy(probs):
    return -sum(p * math.log2(p) if p > 0 else 0 for p in probs)


def mutual_information(prob_mat, prior_prob, k):
    # prob_mat: k x (k+1) joint probabilities p(Y=y_m, X=H_i), prior_prob: k-length array
    prob_mat = np.array(prob_mat)
    p_y = np.sum(prob_mat, axis=0)  # Marginal p(y_m), shape (k+1,)
    print(p_y)
    H_Y = entropy(p_y)
    print("H_Y", H_Y)
    H_Y_given_X = 0
    for i in range(k):
        if prior_prob[i] > 0:
            cond_probs = prob_mat[i, :] / prior_prob[i]  # p(Y=y_m | X=H_i)
            print(cond_probs)
            H_Y_given_X += prior_prob[i] * entropy(cond_probs)
    print("H_Y_given_X", H_Y_given_X)
    return H_Y - H_Y_given_X


def von_neumann_entropy(rho):
    """Compute von Neumann entropy of a density matrix."""
    eigenvalues = np.linalg.eigvals(rho)
    eigenvalues = eigenvalues[
        np.abs(eigenvalues.real) > 1e-10
    ]  # Ignore numerical zeros
    return -np.sum(eigenvalues.real * np.log2(eigenvalues.real))


def apply_depolarization(rho, dep_noise, d):
    """Apply depolarizing channel to a density matrix."""
    return (1 - dep_noise) * rho + dep_noise * np.eye(d) / d


def holevo_bound(dense_mat, prior_prob, dep_noise=0.0):
    """Compute Holevo bound for states in dense_mat with given priors."""
    dense_mat_copy = dense_mat.copy()
    k = len(dense_mat_copy)
    d = dense_mat_copy[0].shape[0]  # Dimension of the Hilbert space
    # Apply depolarization if needed
    if dep_noise > 0:
        dense_mat_copy = [
            apply_depolarization(rho, dep_noise, d) for rho in dense_mat_copy
        ]
    # Average state
    rho_avg = sum(p * rho for p, rho in zip(prior_prob, dense_mat_copy))
    # Entropies
    S_avg = von_neumann_entropy(rho_avg)
    S_individual = sum(
        p * von_neumann_entropy(rho)
        for p, rho in zip(prior_prob, dense_mat_copy)
    )
    return S_avg - S_individual


if __name__ == "__main__":
    n = 2

    prob_mat = np.array(
        [
            [0.3, 0.15, 0.05],
            [0.4, 0.02, 0.08],
        ]
    )
    prob_mat = np.array(
        [
            [0.5, 0, 0],
            [0, 0.5, 0],
        ]
    )
    prob_mat = np.array(
        [
            [0, 0, 0.5],
            [0, 0.5, 0],
        ]
    )
    print(prob_mat)
    print(
        mutual_information(
            prob_mat=prob_mat,
            prior_prob=np.ones(n) * (1 / n),
            k=n,
        )
    )

# if __name__ == "__main__":
#     from qutip import coherent_dm
#
#     prior_prob = np.array([1 / 4] * 4)
#
#     # Example coherent states
#     N = 8
#     alpha = 2
#     dense_mat = [
#         coherent_dm(N, 0).full(),
#         coherent_dm(N, alpha / 2).full(),
#         coherent_dm(N, 2 * alpha / 4).full(),
#         coherent_dm(N, 3 * alpha / 4).full(),
#     ]
#
#     # Compute Holevo bound for different dep_noise values
#     for dep_noise in [0.0, 0.001, 0.1]:
#         chi = holevo_bound(dense_mat, prior_prob, dep_noise=dep_noise)
#         print(f"Holevo bound (dep_noise={dep_noise}): {chi:.4f} bits")
