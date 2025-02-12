import logging
import matplotlib.pyplot as plt
import numpy as np

# Prevent segmentation fault
plt.switch_backend("Tkagg")


def disable_excessive_logging():
    logging.getLogger("matplotlib").disabled = True
    logging.getLogger("matplotlib.colorbar").disabled = True
    logging.getLogger("matplotlib.legend").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.mathtext").disabled = True
    logging.getLogger("PIL.PngImagePlugin").disabled = True


def compute_event_probabilities(prior_prob, povm, state):
    """
    prior_prob: Prior probability of preparing the state, float
    povm: List of measurement operators
    state: Density matrix

    p * Tr(rho * M) for each M in povm
    Note that it does consider prior probabilities.
    """
    probs = []
    for m in povm:
        trace_value = (
            np.trace(
                np.matmul(
                    state,
                    np.multiply(m[None].T.conj(), m),
                )
            )
            .item()
            .real
        )
        probs.append(prior_prob * trace_value)
    return probs


def save_prob_heatmap(
    prior_probs, povm, states, bitstring_to_target_state, strings_used, tag=""
):
    """
    Save the heatmap of the probabilities of measuring each state.
    povm: List of measurement operators
    states: List of density matrices
    """
    probability_matrix = []
    for i in range(len(states)):
        probs = compute_event_probabilities(prior_probs[i], povm, states[i])
        updated_probs = [0] * (len(prior_probs) + 1)
        for j in range(strings_used):
            target_state_index = bitstring_to_target_state[j]
            updated_probs[target_state_index] += probs[j]
        # TODO
        # Postprocessing
        probability_matrix.append(updated_probs)

    ## probability_matrix = np.array(probability_matrix)
    ## print("probability_matrix")
    ## print(probability_matrix)
    ## print(probability_matrix.shape)

    from pandas import DataFrame

    df = DataFrame(
        probability_matrix,
        index=[rf"$\rho_{i}$" for i in range(1, len(states) + 1)],
        columns=[rf"$\Pi_{i}$" for i in range(1, len(states) + 1)] + [r"$\Pi_?$"],
    )

    # Create and display heatmap
    fig = plt.figure(dpi=900)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)
    fig.set_size_inches(8, 5)

    from seaborn import heatmap

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
    cleaned_tag = tag.replace('$', '')
    cleaned_tag = cleaned_tag.replace('\\', '')

    png_name = f"results/seaborn_heatmap_{cleaned_tag}.png"
    plt.tight_layout()
    plt.savefig(fname=png_name, bbox_inches="tight")
    logger = logging.getLogger(__name__)
    logger.info(f"The heatmap is saved as {png_name}")
    plt.close()
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


def plot_psucc_noise_tol(tol, noise_level, psucc, tag):
    fig = plt.figure(dpi=900)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)
    # fig.set_size_inches(8, 5)

    # Contour plot
    x = tol
    y = noise_level
    X, Y = np.meshgrid(x, y)
    Z = psucc

    # contour or contourf
    # TODO
    # Different levels
    # contour = plt.contour(X, Y, Z, levels=15, cmap="viridis")
    contour = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    # contour = plt.contour(X, Y, Z, cmap="viridis")

    # Add labels and title
    plt.xlabel(r"Tolerance $\alpha, \beta$")
    plt.ylabel("Depolarizing noise level")
    
    # TODO
    plt.title(f"{tag}")

    # Add a color bar to show the scale of Z
    plt.colorbar(contour)

    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"results/psucc_noise_tol_{tag}.png")
    plt.close()

    return
