from exp_uqsd_med import *

# import qiskit.primitives
from qiskit_aer import AerSimulator
from qiskit_braket_provider import *


def theo_succ_rates(qsd_method="UQSD", num_points=20):
    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS)

    import sys, os

    func_name = sys._getframe().f_code.co_name
    exp_folder = func_name + "_" + str(datetime.now().date())
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    if not os.path.exists(f"{exp_folder}/results"):
        os.mkdir(f"{exp_folder}/results")
    if not os.path.exists(f"{exp_folder}/raw"):
        os.mkdir(f"{exp_folder}/raw")

    # x_axis = list(range(1, num_points)) * (1 / num_points)
    x_axis = np.array(range(1, num_points)) * (1 / num_points)
    # We fix the inner product of two random statevectors
    sqrt_x = [np.sqrt(i) for i in x_axis]
    fname = f"{exp_folder}/raw/sqrt_x_{num_points}.csv"
    np.savetxt(fname, sqrt_x, fmt="%.16f", delimiter=",")

    # Theoretical lines
    fig = plt.figure(dpi=900)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)
    colors = list(mcolors.TABLEAU_COLORS)
    # alpha=0.5 for near-transparent look
    theo_succ_rate = []
    if qsd_method == "UQSD":
        theo_succ_rate = [1 - 0.5 * np.sqrt(x) for x in x_axis]
        fname = f"{exp_folder}/raw/{qsd_method}_theo_succ_{num_points}.csv"
        np.savetxt(fname, theo_succ_rate, fmt="%.16f", delimiter=",")
        plt.plot(sqrt_x, theo_succ_rate, ".-", color=colors[0])
    elif qsd_method == "MED":
        prob_list = np.arange(0, 0.6, 0.1)
        for i in range(len(prob_list)):
            p1 = prob_list[i]
            theo_succ_rate = [
                0.5 * (1 + np.sqrt(1 - 4 * p1 * (1 - p1) * x)) for x in x_axis
            ]
            fname = f"{exp_folder}/raw/{qsd_method}_theo_succ_{num_points}_{p1:.1f}.csv"
            np.savetxt(fname, theo_succ_rate, fmt="%.16f", delimiter=",")
            plt.plot(
                sqrt_x, theo_succ_rate, ".-", label=r"$p_1$" f" = {p1:.1f}", color=colors[i]
            )
        plt.legend(loc="lower left")

    # plt.xlabel("c0 of Bob" + f" ({num_points} points)")
    plt.xlabel(r"$|\langle \phi_1 | \phi_2 \rangle|$")
    plt.ylabel("Success rate")
    plt.title(f"{qsd_method} Theoretical success rates")
    plt.grid(True)
    plt.savefig(
        fname=f"{exp_folder}/results/theo_succ_rates_{qsd_method}.png",
        bbox_inches="tight",
    )
    return


if __name__ == "__main__":
    theo_succ_rates(qsd_method="UQSD", num_points=20)
    theo_succ_rates(qsd_method="MED", num_points=20)
