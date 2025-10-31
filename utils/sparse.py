# Collaborated with Grok 3

import numpy as np
from qutip import coherent_dm
import matplotlib.pyplot as plt
import pandas as pd


def calculate_sparsity(density_matrix, zero_threshold=1e-10):
    """
    Calculate the sparsity of a square density matrix.

    Parameters:
    density_matrix : numpy.ndarray
        A square matrix (density matrix) assumed to be Hermitian with trace 1.

    Returns:
    float : Sparsity (fraction of zero elements)
    """
    # Ensure the matrix is square
    if density_matrix.shape[0] != density_matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Total number of elements
    total_elements = density_matrix.size

    # Count zero elements (using a small threshold for numerical stability)
    # Adjust the threshold as needed for floating-point precision
    zero_elements = np.sum(np.abs(density_matrix) < zero_threshold)

    # Sparsity is the fraction of zero elements
    sparsity = zero_elements / total_elements

    return sparsity


def evaluate_sparsity_for_dimensions(
    N_values, alpha_scales, thresholds, alpha_base
):
    """
    Evaluate sparsity of coherent state density matrices, return results, and plot sparsity vs. threshold.

    Parameters:
    N_values (list): Hilbert space dimensions.
    alpha_scales (list): Scaling factors for alpha.
    thresholds (list): Zero thresholds for sparsity calculation.
    alpha_base (float): Base alpha value.

    Returns:
    list: List of dictionaries with N, Alpha, Threshold, and Sparsity.
    """
    # Collect results
    results = []
    for N in N_values:
        for scale in alpha_scales:
            alpha = alpha_base * scale
            dm = coherent_dm(N=N, alpha=alpha).data.to_array()
            for threshold in thresholds:
                sparsity = calculate_sparsity(dm, zero_threshold=threshold)
                results.append(
                    {
                        "N": N,
                        "Alpha": alpha,
                        "Threshold": threshold,
                        "Sparsity": sparsity,
                    }
                )

    # Plotting
    plt.figure(figsize=(10, 6), dpi=600)  # Wider figure for more lines
    colors = plt.cm.tab10(
        np.linspace(0, 1, len(N_values) * len(alpha_scales))
    )  # Color map for distinct hues
    linestyles = ["-", "--"]  # Cycle linestyles by N
    markers = ["o", "s", "^", "d"]  # Cycle markers by alpha_scale

    for i, N in enumerate(N_values):
        for j, scale in enumerate(alpha_scales):
            alpha = alpha_base * scale
            subset = [r for r in results if r["N"] == N and r["Alpha"] == alpha]
            sparsities = [r["Sparsity"] for r in subset]
            plt.plot(
                thresholds,
                sparsities,
                label=f"N={N}, |α|={abs(alpha):.1f}",
                color=colors[i * len(alpha_scales) + j],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[j % len(markers)],
                markersize=6,
                linewidth=2,
                alpha=0.8,
            )

    plt.xscale("log")
    plt.xlabel("Zero Threshold", fontsize=12)
    plt.ylabel("Sparsity", fontsize=12)
    plt.title(
        "Sparsity of Coherent State Density Matrices", fontsize=14, pad=15
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        ncol=1,
    )
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

    return results


# Example usage
if __name__ == "__main__":
    # Example: Create a 4x4 density matrix (e.g., a simple mixed state)
    dim = 4
    rho = np.diag([0.5, 0.3, 0.2, 0.0])  # Diagonal density matrix
    sparsity = calculate_sparsity(rho)
    print(f"Sparsity of the density matrix: {sparsity:.4f}")
