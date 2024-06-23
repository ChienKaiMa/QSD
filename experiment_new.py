from exp_uqsd_med import *
import qiskit.primitives
from qiskit_aer import AerSimulator
from qiskit_braket_provider import *


def exp0(
    qsd_method="UQSD",
    num_points=20,
    num_rounds=1,
    shots=500,
    n_qubit=2,
    p1=0.5,
    rand_init_state=True,
    print_circ=True,
    sim=False,
    ibmq=False,
    ionq=True,
    backend="ibm_osaka",
):
    # Generate circuits
    from qiskit.transpiler.passes import UnrollCustomDefinitions

    aer_sim = AerSimulator(method="statevector")
    sampler = qiskit.primitives.Sampler()
    circuits: list[QuantumCircuit] = []
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
    print(len(circuits))
    # print(circuits[0])
    # Run the circuits
    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS)

    import sys, os

    func_name = sys._getframe().f_code.co_name
    if print_circ:
        if not os.path.exists(f".//{func_name}"):
            os.mkdir(f".//{func_name}")
        if not os.path.exists(f".//{func_name}//figures"):
            os.mkdir(f".//{func_name}//figures")

    if sim:
        if not os.path.exists(f".//{func_name}"):
            os.mkdir(f".//{func_name}")
        if not os.path.exists(f".//{func_name}//results"):
            os.mkdir(f".//{func_name}//results")

    if ibmq or ionq:
        if not os.path.exists(f".//{func_name}"):
            os.mkdir(f".//{func_name}")
        if not os.path.exists(f".//{func_name}//results"):
            os.mkdir(f".//{func_name}//results")

    if sim:
        colors = list(mcolors.TABLEAU_COLORS)

        fig = plt.figure(dpi=900)
        fig.set_figwidth(6)
        fig.set_figheight(4.8)

        # x_axis = list(range(1, num_points)) * (1 / num_points)
        x_axis = np.array(range(1, num_points)) * (1 / num_points)
        # print(x_axis)
        # Theoretical lines

        theo = []
        if qsd_method == "UQSD":
            theo = [1 - 0.5 * np.sqrt(i) for i in x_axis]
        elif qsd_method == "MED":
            theo = [0.5 * (1 + np.sqrt(1 - 4 * p1 * (1 - p1) * i)) for i in x_axis]
        sqrt_x = [np.sqrt(i) for i in x_axis]

        # # theoretical result
        # if qsd_method in ["MED", "med"]:
        #     success_rate = (
        #         1
        #         + np.sqrt(
        #             1 - 4 * probability * (1 - probability) * (inner_product) ** 2
        #         )
        #     ) / 2
        #     print(f"MED success rate = {success_rate}")
        # else:
        #     print("UQSD success rate = ", 1 - inner_product / 2)

        plt.plot(sqrt_x, theo, "-", color=colors[0], alpha=0.5)
        #     plt.figure(
        #         dpi=600
        #     )  # https://stackoverflow.com/questions/39870642/how-to-plot-a-high-resolution-graph
        #     # Theoretical lines
        #     x_axis = np.linspace(0, 2, num_points)
        #     for t in range(max_shift):
        #         theo = 0.5 * (1 + max(1 - (1 / 2**n) * t, 0) * np.cos(angles))
        #         plt.plot(x_axis, theo, "-", color=colors[t], alpha=0.5)
        #

        job = sampler.run(circuits, shots=shots)
        # job = aer_sim.run(circuits=circuits, shots=shots)
        # counts_list = job.result().get_counts()

        # Bo-Hung start
        probability = p1
        result = job.result()
        avg_hit_rate_list = []
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
                    # TODO TODO fix the index
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
                avg_hit_rate_list.append(analysis_dict["psucc"])
                # print("Experiment result : ", data_analysis(outcome, qsd_method))
                ### Bo-hung end
        plt.plot(sqrt_x, avg_hit_rate_list, ".", label=f"p1 = {p1}", color=colors[0])
        plt.xlabel(r"$\sqrt{c0}$ of Bob" + f" ({num_points} points) (n = {n_qubit})")
        plt.ylabel("Average hit rate")
        plt.legend(loc="lower left")
        plt.title(f"{qsd_method} (Qiskit Sampler)")
        plt.grid(True)
        exp_folder = str(datetime.now().date())
        plt.savefig(
            fname=f"{os.getcwd()}/{exp_folder}/results/"
            + f"exp0_result_sim_{qsd_method}"
            + f"_{p1}_{n_qubit}"
            + ".png",
            bbox_inches="tight",
        )
        plt.close()

        return
        print(counts_list, flush=True)

        for i in range(1, num_points):
            # TODO Calculate the average atari rate
            atari_rate = 0
            for j in range(num_rounds):
                for key in counts_list[(i - 1) * num_rounds + j].keys():
                    if qsd_method == "UQSD":
                        if key[0] == "0" and key[2:] == "00":
                            atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                        if key[0] == "1" and key[2:] == "10":
                            atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                        if key[2:] == "01":
                            atari_rate += (
                                0.5 * counts_list[(i - 1) * num_rounds + j][key]
                            )
                        if key[2:] == "11":
                            atari_rate += (
                                0.5 * counts_list[(i - 1) * num_rounds + j][key]
                            )
                    if qsd_method == "MED":
                        if key[0] == "0" and key[2:] == "0":
                            atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                        if key[0] == "1" and key[2:] == "1":
                            atari_rate += counts_list[(i - 1) * num_rounds + j][key]

                    print(key)
                    pass
            atari_rate = atari_rate / shots / num_rounds
            avg_hit_rate_list.append(atari_rate)
        plt.plot(sqrt_x, avg_hit_rate_list, ".", label=f"p1 = {p1}", color=colors[0])

        plt.xlabel(r"$\sqrt{c0}$ of Bob" + f" ({num_points} points) (n = {n_qubit})")
        plt.ylabel("Average hit rate")
        plt.legend(loc="lower left")
        plt.title(f"{qsd_method} (Aer Simulator)")
        plt.grid(True)
        plt.savefig(
            fname=f"{os.getcwd()}/exp0/results/"
            + f"exp0_result_sim_{qsd_method}"
            + f"_{p1}_{n_qubit}"
            + ".png",
            bbox_inches="tight",
        )
        plt.close()

    if ibmq:
        from qiskit_ibm_runtime import QiskitRuntimeService, Batch

        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend(backend)
        # To run on hardware, select the backend with the fewest number of jobs in the queue
        # backend = service.least_busy(operational=True, simulator=False)
        print(
            f"Name: {backend.name}\n"
            f"Version: {backend.version}\n"
            f"No. of qubits: {backend.num_qubits}\n"
        )

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        target = backend.target

        from qiskit import qasm2

        circuits_ibm = []
        for circuit in circuits:
            # pm = generate_preset_pass_manager(target=target, optimization_level=3)
            # pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
            # circuit_ibm = pm.run(circuit)
            circuit_ibm = transpile(circuit, backend=backend, optimization_level=3)
            print(f"{circuit_ibm.count_ops()}")
            print(f"Layers: {circuit_ibm.depth()}")
            # For Qiskit 0.46
            # circuit_clean = QuantumCircuit.from_qasm_str(circuit_ibm.qasm())
            circuit_clean = QuantumCircuit.from_qasm_str(qasm2.dumps(circuit_ibm))
            circuits_ibm.append(circuit_clean)
            if print_circ:
                # circuit_ibm.draw(
                #     output="mpl",
                #     filename=f"./exp0/figures/"
                #     + "exp0"
                #     + f"_{experiment_type}_transpile"
                #     # + f"_{c0_bob:.3f}"
                #     # + f"_{n_qubit}_{seed}"
                #     + ".png",
                # )
                pass

        job = backend.run(circuits=circuits_ibm, shots=shots)
        print(job.job_id())

        theo = []
        if qsd_method == "UQSD":
            theo = [1 - 0.5 * np.sqrt(i) for i in x_axis]
        elif qsd_method == "MED":
            theo = [0.5 * (1 + np.sqrt(1 - 4 * p1 * (1 - p1) * i)) for i in x_axis]
        sqrt_x = [np.sqrt(i) for i in x_axis]
        plt.plot(sqrt_x, theo, "-", color=colors[0], alpha=0.5)
        # TODO
        # Plot later

    if ionq:
        ionq_backend = BraketProvider().get_backend("Aria 1")
        ionq_task = ionq_backend.run(circuits, shots=shots)
        print(ionq_task.job_id())

        import matplotlib.colors as mcolors

        colors = list(mcolors.TABLEAU_COLORS)

        fig = plt.figure(dpi=900)
        fig.set_figwidth(6)
        fig.set_figheight(4.8)

        # x_axis = list(range(1, num_points)) * (1 / num_points)
        x_axis = np.array(range(1, num_points)) * (1 / num_points)
        print(x_axis)
        # Theoretical lines
        theo = []
        if qsd_method == "UQSD":
            theo = [1 - 0.5 * np.sqrt(i) for i in x_axis]
        elif qsd_method == "MED":
            theo = [0.5 * (1 + np.sqrt(1 - 4 * p1 * (1 - p1) * i)) for i in x_axis]

        plt.plot(x_axis, theo, "-", color=colors[0], alpha=0.5)

        # Assume the job is retrieved
        counts_list = ionq_task.result().get_counts()
        avg_hit_rate_list = []
        # Calculate the average atari rate
        for i in range(1, num_points):
            atari_rate = 0
            for j in range(num_rounds):
                for key in counts_list[(i - 1) * num_rounds + j].keys():
                    if qsd_method == "UQSD":
                        if key[0] == "0" and key[2:] == "00":
                            atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                        if key[0] == "1" and key[2:] == "01":
                            atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                        if key[2:] == "10":
                            atari_rate += (
                                0.5 * counts_list[(i - 1) * num_rounds + j][key]
                            )
                        if key[2:] == "11":
                            atari_rate += (
                                0.5 * counts_list[(i - 1) * num_rounds + j][key]
                            )
                    if qsd_method == "MED":
                        if key[0] == "0" and key[2:] == "0":
                            atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                        if key[0] == "1" and key[2:] == "1":
                            atari_rate += counts_list[(i - 1) * num_rounds + j][key]

                    print(key)
                    pass
            atari_rate = atari_rate / shots / num_rounds
            avg_hit_rate_list.append(atari_rate)
        plt.plot(x_axis, avg_hit_rate_list, ".", label=f"p1 = {p1}", color=colors[0])

        backend = "ionq_aria_1"
        plt.xlabel("c0 of Bob" + f" ({num_points} points) (n = {n_qubit})")
        plt.ylabel("Average hit rate")
        plt.legend(loc="lower right")
        plt.title(f"{qsd_method} ({backend})")
        plt.grid(True)
        plt.savefig(
            fname=f"{os.getcwd()}/exp0/results/"
            + f"exp0_result_{qsd_method}_{backend}"
            + f"_{p1}_{n_qubit}"
            + ".png",
            bbox_inches="tight",
        )
        plt.close()
    return


if __name__ == "__main__":
    exp0(
        num_points=40,
        qsd_method="UQSD",
        rand_init_state=True,
        sim=True,
        ibmq=False,
        ionq=False,
        backend="ionq_aria_1",
    )
    exp0(
        num_points=40,
        qsd_method="MED",
        rand_init_state=True,
        sim=True,
        ibmq=False,
        ionq=False,
        backend="ionq_aria_1",
    )
    # exp0(
    #     experiment_type="MED",
    #     sim=False,
    #     ibmq=False,
    #     ionq=True,
    #     backend="ionq_aria_1",
    # )
