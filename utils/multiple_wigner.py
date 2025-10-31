import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from qutip import wigner, isket, ket2dm
from math import sqrt


def plot_multiple_wigner(
    states,
    xvec=None,
    yvec=None,
    method="clenshaw",
    projection="2d",
    g=sqrt(2),
    sparse=False,
    parfor=False,
    colors=None,
    linestyles=None,
    colorbar=False,
    use_filled=False,
    suptitle=r"Multiple Wigner functions",
    save_path="wigner_plot.png",
    close_fig=False,
):
    """
    Plot Wigner functions for multiple quantum states on the same plot (contour lines)
    or in a grid (filled contours) and save as a .png.

    Parameters
    ----------
    states : list of :obj:`.Qobj`
        List of density matrices or kets to visualize.
    xvec : array_like, optional
        x-coordinates for Wigner function.
    yvec : array_like, optional
        y-coordinates for Wigner function.
    method : str {'clenshaw', 'iterative', 'laguerre', 'fft'}, default: 'clenshaw'
        Method for calculating the Wigner function.
    projection : str {'2d', '3d'}, default: '2d'
        Plot as contour ('2d') or surface ('3d').
    g : float
        Scaling factor, default `g = sqrt(2)`.
    sparse : bool
        Use sparse format.
    parfor : bool
        Use parallel calculation.
    colors : list of colors, optional
        Colors for each state’s contours (e.g., ['blue', 'red', 'green']).
        If None, uses ['b', 'r', 'g', 'k', ...].
    linestyles : list of linestyles, optional
        Linestyles for contours (e.g., ['-', '--', ':']).
        If None, all use '-'.
    colorbar : bool
        Whether to include a colorbar (only for filled contours).
    use_filled : bool
        If True, plot filled contours in a grid of subplots instead of overlapping contours.
    save_path : str
        Path to save the .png file, default 'wigner_plot.png'.
    close_fig : bool
        Whether to close the figure after saving, default False.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes (or list of axes for grid) used for the plot.
    """
    # Default coordinates
    xvec = np.linspace(-7.5, 7.5, 200) if xvec is None else xvec
    yvec = np.linspace(-7.5, 7.5, 200) if yvec is None else yvec
    wlim = 0
    Ws = []

    # Compute Wigner functions
    for state in states:
        if isket(state):
            state = ket2dm(state)
        W = wigner(
            state, xvec, yvec, method=method, g=g, sparse=sparse, parfor=parfor
        )
        Ws.append(W)
        wlim = max(abs(W).max(), wlim)

    # Set normalization
    norm = mpl.colors.Normalize(-wlim, wlim)

    # Default colors and linestyles
    if colors is None:
        colors = ["b", "r", "g", "k", "c", "m", "y"][: len(states)]
    if linestyles is None:
        linestyles = ["-"] * len(states)
    else:
        linestyles = linestyles[: len(states)]

    if use_filled:
        # Grid layout for filled contours
        n_states = len(states)
        if n_states > 3:
            ncols = int(np.ceil(np.sqrt(n_states)))
            nrows = int(np.ceil(n_states / ncols))
        else:
            ncols = n_states
            nrows = 1
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 4 * nrows),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_2d(axes).ravel() if n_states > 1 else [axes]

        for i, (W, ax) in enumerate(zip(Ws, axes[:n_states])):
            if projection == "2d":
                cf = ax.contourf(xvec, yvec, W, 100, norm=norm, cmap="RdBu")
                ax.contour(xvec, yvec, W, 10, colors="black", linewidths=0.5)
                # ax.set_title(rf"$N = {2 ** (i+1)}$")
                ax.set_title(f"State {i+1}")
                if colorbar:
                    plt.colorbar(cf, ax=ax)
            else:
                X, Y = np.meshgrid(xvec, yvec)
                ax.plot_surface(X, Y, W, rstride=5, cstride=5, cmap="RdBu")
            ax.set_xlabel(r"$\rm{Re}(\alpha)$")
            ax.set_ylabel(r"$\rm{Im}(\alpha)$")

        # Hide unused subplots
        for ax in axes[n_states:]:
            ax.set_visible(False)

        fig.tight_layout()
    else:
        # Single plot with contour lines
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, (W, color, linestyle) in enumerate(zip(Ws, colors, linestyles)):
            if projection == "2d":
                ax.contour(
                    xvec,
                    yvec,
                    W,
                    10,
                    colors=color,
                    linestyles=linestyle,
                    linewidths=1.5,
                    label=rf"$N = {2 ** (i+1)}$",
                )
            else:
                X, Y = np.meshgrid(xvec, yvec)
                ax.plot_wireframe(
                    X,
                    Y,
                    W,
                    color=color,
                    linestyle=linestyle,
                    rstride=5,
                    cstride=5,
                )
        ax.set_xlabel(r"$\rm{Re}(\alpha)$", fontsize=12)
        ax.set_ylabel(r"$\rm{Im}(\alpha)$", fontsize=12)
        ax.legend()
        ax.set_title("Wigner Functions of Coherent States", fontsize=14)

    # Save the plot
    plt.suptitle(
        # r"Wigner Function of Truncated Coherent States $|\alpha| = 4, \angle{\alpha}=\pi/4$",
        suptitle,
        y=0,
        fontsize=14,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Close figure only if requested
    if close_fig:
        plt.close(fig)

    return fig, ax