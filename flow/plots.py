import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from seaborn import heatmap
from utils.prob_matrix import *

# Prevent segmentation fault
# plt.switch_backend("Tkagg")


def disable_excessive_logging():
    logging.getLogger("matplotlib").disabled = True
    logging.getLogger("matplotlib.colorbar").disabled = True
    logging.getLogger("matplotlib.legend").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.mathtext").disabled = True
    logging.getLogger("PIL.PngImagePlugin").disabled = True


def compute_event_probabilities(prior_prob, povm, state: np.array):
    """
    prior_prob: Prior probability of preparing the state, float
    povm: List of measurement operators
    state: Density matrix

    p * Tr(rho * M) for each M in povm
    Note that it does consider prior probabilities.
    """
    probs = []
    for m in povm:
        trace_value = np.trace(
            np.matmul(
                state,
                np.multiply(m[None].T.conj(), m),
            )
        )
        trace_value = trace_value.item()
        assert trace_value.real >= -1e-3
        assert abs(trace_value.imag) <= 1e-7
        probs.append(prior_prob * trace_value.real)
    return probs


def compute_event_probabilities_2(prior_prob, povm, state: np.array):
    """
    prior_prob: Prior probability of preparing the state, float
    povm: List of measurement operators
    state: Density matrix

    p * Tr(rho * M) for each M in povm
    Note that it does consider prior probabilities.
    """
    probs = []
    for m in povm:
        trace_value = np.trace(np.matmul(state, m)).item()
        # print(trace_value, flush=True)
        # TODO Find the potential bug
        # assert trace_value.real >= -1e-2
        assert abs(trace_value.imag) <= 1e-7
        probs.append(prior_prob * abs(trace_value.real))
    return probs


def save_prob_heatmap(
    prior_probs,
    povm,
    states,
    bitstring_to_target_state,
    strings_used,
    tag="",
    reuse_fig=None,
):
    """
    Save the heatmap of the probabilities of measuring each state.
    povm: List of measurement operators
    states: List of density matrices
    """
    probability_matrix = calculate_prob_matrix(
        prior_probs, povm, states, bitstring_to_target_state, strings_used
    )

    ## probability_matrix = np.array(probability_matrix)
    ## print("probability_matrix")
    ## print(probability_matrix)
    ## print(probability_matrix.shape)

    df = DataFrame(
        probability_matrix,
        index=[rf"$\rho_{i}$" for i in range(1, len(states) + 1)],
        columns=[rf"$\Pi_{i}$" for i in range(1, len(states) + 1)]
        + [r"$\Pi_?$"],
    )

    # Create and display heatmap
    if reuse_fig is not None:
        fig = reuse_fig
    else:
        fig = plt.figure(dpi=600)
        fig.set_size_inches(3.2, 2.7)

    heatmap(
        df,
        annot=True,
        cmap="viridis",
        vmin=0,
        vmax=1,
        fmt=".3f",
    )
    plt.title(rf"$Tr(𝜌_𝑖 Π_𝑖 )$ {tag}")
    # TODO fig name
    cleaned_tag = tag.replace("$", "")
    cleaned_tag = cleaned_tag.replace("\\", "")

    png_name = f"results/seaborn_heatmap_{cleaned_tag}.png"
    plt.tight_layout()
    plt.savefig(fname=png_name, bbox_inches="tight")
    logger = logging.getLogger(__name__)
    logger.info(f"The heatmap is saved as {png_name}")
    fig.clear()
    # plt.close()
    return probability_matrix


def plot_total_prob(points, tag, noise_level):
    """Verify that the total probability of the POVM sums to identity."""
    fig = plt.figure(dpi=900)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)
    fig.set_size_inches(8, 5)

    ax = fig.add_subplot(111, projection="3d")

    for point in points:
        ax.scatter(point[0], point[1], point[2])

    ax.set_zlim(-0.01, 1.01)

    # Labeling axes
    ax.set_xlabel(r"Tolerance $\alpha$")
    ax.set_ylabel(r"Tolerance $\gamma$")
    ax.set_zlabel("Total probability")

    # ax.legend(f"Noise level {noise_level}")

    # Adjust the view angle
    ax.view_init(elev=30.0, azim=60)

    plt.savefig(f"total_prob_alpha_gamma_{tag}.png")
    return


def plot_psucc(points, tag, noise_level):
    """Success probability of discrimination."""
    fig = plt.figure(dpi=900)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)
    fig.set_size_inches(8, 5)

    ax = fig.add_subplot(111, projection="3d")

    for point in points:
        ax.scatter(point[0], point[1], point[2])

    ax.set_zlim(-0.01, 1.01)

    # Labeling axes
    ax.set_xlabel(r"Tolerance $\alpha$")
    ax.set_ylabel(r"Tolerance $\beta$")
    ax.set_zlabel("Probability of successful discrimination")

    # ax.legend(f"Noise level {noise_level}")

    # Adjust the view angle
    ax.view_init(elev=30.0, azim=60)

    plt.savefig(f"success_prob_alpha_gamma_{tag}.png")

    return


# def save_3dplot(X, Y, Z, fig):
# TODO
def save_3dplot(points, fig, tag, noise_level):
    ## Make data
    ## X = np.arange(-5, 5, 0.25)
    ## Y = np.arange(-5, 5, 0.25)
    ## X, Y = np.meshgrid(X, Y)
    ## R = np.sqrt(X**2 + Y**2)
    ## Z = np.sin(R)
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface with heatmap-like coloring
    # https://stackoverflow.com/questions/51574861/plotting-3d-surface-using-python-raise-valueerrorargument-z-must-be-2-dimensi
    for point in points:
        ax.scatter(point[0], point[1], point[2])
    ## surf = ax.plot(
    ##     points[:, 0],
    ##     cmap="viridis",  # You can choose different colormaps: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ##     edgecolor="none",
    ## )  # Remove edges to give a smoother look

    # Customize the z axis
    ax.set_zlim(-0.01, 1.01)

    # Add a color bar which maps values to colors
    ## cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    ## cbar.set_label("Z Value")  # Label for the colorbar

    # Labeling axes
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Adjust the view angle
    ax.view_init(
        elev=30.0, azim=60
    )  # Elevation and azimuth can be adjusted for better visualization

    plt.savefig("3dplot_color.png")

    return


def plot_psucc_noise_tol(tol, noise_level, psucc, tag, num_levels):
    # Prevent segmentation fault
    # plt.switch_backend("Tkagg")
    fig = plt.figure(dpi=1200)
    fig.set_size_inches(3.2, 2.7)

    # Contour plot
    x = tol
    y = noise_level
    X, Y = np.meshgrid(x, y)
    Z = psucc

    # contour or contourf
    # TODO
    # Different levels
    # contour = plt.contour(X, Y, Z, levels=15, cmap="viridis")

    # https://stackoverflow.com/questions/53641644/set-colorbar-range-with-contourf
    vmin = -0
    vmax = 1
    levels = np.linspace(vmin, vmax, num_levels + 1)

    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    contour = plt.contour(
        X,
        Y,
        Z,
        levels=levels,
        # cmap="viridis",
        # cmap="cool",
        cmap="seismic",
        vmin=vmin,
        # vmax=vmax,
        extend="both",
        edgecolor="white",
        alpha=0.8,
    )

    # https://stackoverflow.com/questions/73886009/contourf-plots-with-streamline-numbers
    plt.clabel(contour, inline=True, fontsize=4, fmt="%.3f", colors="black")

    from numpy import linspace

    # Add a color bar to show the scale of Z
    plt.colorbar(
        contour,
        ticks=linspace(vmin, vmax, 11),
        label="Success probability",
    )

    # Add labels and title
    plt.xlabel(r"Tolerance $\alpha, \beta$")
    plt.ylabel("Depolarizing noise level")

    # TODO
    plt.title(f"{tag}")

    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"results/psucc_noise_tol_{tag}.png")
    plt.close()

    return
