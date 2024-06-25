from exp_uqsd_med import *
import qiskit.primitives
from qiskit_aer import AerSimulator
from qiskit_braket_provider import *
import os, sys


def theo_succ_rates(qsd_method="UQSD", num_points=20):
    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS)

    func_name = sys._getframe().f_code.co_name
    exp_folder = func_name + "_" + str(datetime.now().date())
    dir_list = [
        f"{exp_folder}/results",
        f"{exp_folder}/raw",
    ]
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

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
                sqrt_x,
                theo_succ_rate,
                ".-",
                label=r"$p_1$" f" = {p1:.1f}",
                color=colors[i],
            )
        plt.legend(loc="lower left")

    # plt.xlabel("c0 of Bob" + f" ({num_points} points)")
    plt.xlabel(r"$|\langle \psi_1 | \psi_2 \rangle|$")
    plt.ylabel("Success rate")
    plt.title(f"{qsd_method} Theoretical success rates")
    plt.grid(True)
    plt.savefig(
        fname=f"{exp_folder}/results/theo_succ_rates_{qsd_method}.png",
        bbox_inches="tight",
    )
    return


def scale_and_sim(
    qsd_method="UQSD",
    num_points=20,
    num_rounds=1,
    shots=500,
    n_qubit=2,
    p1=0.5,
    rand_init_state=True,
    print_circ=True,
    # backend="ibm_osaka",
):
    func_name = sys._getframe().f_code.co_name
    exp_folder = func_name + "_" + str(datetime.now().date())
    dir_list = [
        f"{exp_folder}/circuits",
        f"{exp_folder}/figures",
        f"{exp_folder}/results",
        f"{exp_folder}/raw",
    ]
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    def prepare_circuits():
        import qiskit.qasm2

        aer_sim = AerSimulator(method="statevector")
        circuits = []
        for i in range(1, num_points):
            for j in range(num_rounds):
                inner_product = np.sqrt(i / num_points)
                circuit = build_qsd_experiment_circuit(
                    qsd_method=qsd_method,
                    probability=p1,
                    inner_product=inner_product,
                    num_qubit=n_qubit,
                    rand_init_state=rand_init_state,
                    # seed=i * j,
                    # seed=i * num_rounds + j,
                )
                circuit_t = transpile(circuit[0], aer_sim, optimization_level=3)
                circuits.append(circuit_t)
                qiskit.qasm2.dump(
                    circuit[0],
                    f"{exp_folder}/circuits/{qsd_method}_full_"
                    f"{n_qubit}_{i}_{num_points}_{p1}.qasm",
                )
                qiskit.qasm2.dump(
                    circuit_t,
                    f"{exp_folder}/circuits/{qsd_method}_full_"
                    f"{n_qubit}_{i}_{num_points}_{p1}_transpiled.qasm",
                )
        print(len(circuits))
        return circuits

    circuits = prepare_circuits()
    sampler = qiskit.primitives.Sampler()

    # from qiskit.transpiler.passes import UnrollCustomDefinitions
    job = sampler.run(circuits, shots=shots)
    # job = aer_sim.run(circuits=circuits, shots=shots)
    # counts_list = job.result().get_counts()

    # Bo-Hung start
    probability = p1
    result = job.result()
    sim_succ_rates = []
    for i in range(1, num_points):
        for j in range(num_rounds):
            inner_product = i / num_points

            # ------ collect data -------#
            index = (i - 1) * num_rounds + j
            print(index)
            print(len(result.quasi_dists))
            if rand_init_state:
                outcome = vec_to_array(
                    result.quasi_dists[index].binary_probabilities(), n_qubit + 1
                )
            else:
                # TODO fix the index
                outcome = np.array(
                    [
                        np.array([probability, 1 - probability])[_]
                        * vec_to_array(
                            result.quasi_dists[
                                i * num_points + j
                            ].binary_probabilities(),
                            n_qubit,
                        )
                        for _ in range(2)
                    ]
                ).flatten()

            # ------- print result ------#
            # gate count
            print("gate count:", circuits[0].count_ops())

            # experimental result
            analysis_dict = data_analysis(outcome, qsd_method)
            sim_succ_rates.append(analysis_dict["psucc"])
            # print("Experiment result : ", data_analysis(outcome, qsd_method))
    ### Bo-hung end
    fname = (
        f"{exp_folder}/raw/"
        f"{qsd_method}_sim_succ_{n_qubit}_{num_points}_{p1:.1f}.csv"
    )
    np.savetxt(fname, sim_succ_rates, fmt="%.16f", delimiter=",")
    if qsd_method == "UQSD":
        fname = f"theo_succ_rates_2024-06-24/raw/UQSD_theo_succ_{num_points}.csv"
    elif qsd_method == "MED":
        fname = f"theo_succ_rates_2024-06-24/raw/MED_theo_succ_{num_points}_{p1}.csv"
    theo_succ_rates = np.loadtxt(fname=fname)
    fname = "theo_succ_rates_2024-06-24/raw/sqrt_x_20.csv"
    theo_sqrt_x = np.loadtxt(fname=fname)

    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS)
    fig = plt.figure(dpi=900)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)
    plt.plot(
        theo_sqrt_x,
        theo_succ_rates,
        "-",
        label=f"p1 = {p1}",
        color=colors[0],
        alpha=0.5,
    )

    # x_axis = list(range(1, num_points)) * (1 / num_points)
    x_axis = np.array(range(1, num_points)) * (1 / num_points)
    # We fix the inner product of two random statevectors
    sqrt_x = [np.sqrt(i) for i in x_axis]
    plt.plot(sqrt_x, sim_succ_rates, ".", label=f"p1 = {p1}", color=colors[0])
    plt.xlabel(
        r"$|\langle \psi_1 | \psi_2 \rangle|$"
        + f" ({num_points} points) (n = {n_qubit})"
    )
    plt.ylabel("Success rate")
    plt.legend(loc="lower left")
    plt.title(f"{qsd_method} (Qiskit Sampler)")
    plt.grid(True)
    plt.savefig(
        fname=f"{exp_folder}/results/sim_{qsd_method}_{p1}_{n_qubit}.png",
        bbox_inches="tight",
    )
    plt.close()
    return


if __name__ == "__main__":
    theo_succ_rates(qsd_method="UQSD", num_points=20)
    theo_succ_rates(qsd_method="MED", num_points=20)
    scale_and_sim(qsd_method="UQSD", n_qubit=10, num_points=20)
    scale_and_sim(qsd_method="MED", n_qubit=10, num_points=20)
