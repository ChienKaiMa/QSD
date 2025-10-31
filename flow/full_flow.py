from simulate_uqsd import *
from solve_mix import *

if __name__ == '__main__':
    # Generate target pure states
    noise_level = [0.1 ** (6 - 0.25 * i) for i in range(10)]
    for i in noise_level:
        # Generate corresponding noisy states
        states, disturbance_states, combined_states = ProblemSpec.gen_noisy_states(
            num_qubits=nq,
            num_states=ns,
            seeds=get_random_seeds(ns, seed=seed),
            noise_level=0.01,
            noise_rank=2,
        )
        # Solve for quantum circuits with different approaches
        # Apply the quantum circuits
            # test_dut(dut=)

        # Check how noise affects the result
    # Calculate how lost we are
    # How do you define inconclusivity
    # State x Measure close to 1
    # Average the wrong answers
    # Similar to integral?


    pass

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("TkAgg")  # To prevent segmentation fault
from matplotlib.colors import TABLEAU_COLORS
import logging
from argparse import *
import numpy as np
from qiskit.circuit import QuantumCircuit


def _prepare(__name__):
    parser = ArgumentParser()
    parser.add_argument("-q", "--nqubits", default=2)
    parser.add_argument("-n", "--nstates", default=3)
    parser.add_argument("-s", "--seed", default=42)
    parser.add_argument("--qasm")
    args = parser.parse_args()
    nq = int(args.nqubits)
    ns = int(args.nstates)
    seed = int(args.seed)
    case_id = f"q{nq}_n{ns}_s{seed}"

    logging.basicConfig(
        filename=f"sim_uqsd_{case_id}.log",
        filemode="a",
        format="{asctime} {levelname} {filename}:{lineno}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=logging.INFO,  # Qiskit dumps too many DEBUG messages
        encoding="utf-8",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Start simulate_uqsd.py")
    if args.qasm == None:
        qasm_name = f"qc_iso_{case_id}_no_backend.qasm"
        logger.info(f"No qasm file is provided. Will default to {qasm_name}")
    else:
        qasm_name = args.qasm
        logger.info(f"Load qasm file {qasm_name}")
    logger.info(f"nq = {nq}, ns = {ns}, seed = {seed}")
    return nq, ns, seed, case_id, qasm_name, logger


def _quick_check(case_id):
    # Debug POVM and Phi_tilde

    np.set_printoptions(precision=4)
    Phi = np.load(f"Phi_{case_id}.npy")
    Phi_tilde = np.load(f"Phi_tilde_{case_id}.npy")
    povm = np.load(f"povm_{case_id}.npy")

    print(Phi)  # target states
    print(Phi_tilde)
    print(povm)
    print(np.matmul(Phi_tilde.conj(), Phi))  # Should be identity
    return


def test_dut(dut, param, target_states: list[str] | None = None, noise_model=None):
    """Test DUT under the two-qubit depolarizing error"""
    noise_model = _two_qubit_depolarizing_noise_model(param)

    noise_result = sv_backend.run(dut, noise_model=noise_model).result()
    try:
        print(param)
        print("'001' count =", noise_result.get_counts()["001"])
    except:
        pass
    try:
        print("'0001' count =", noise_result.get_counts()["0001"])
    except:
        pass
    if target_states is None:
        return noise_result.get_statevector().probabilities()
    else:
        return [
            noise_result.get_statevector().probabilities()[state].item()
            for state in target_states
        ]


def _two_qubit_depolarizing_noise_model(param):
    noise_model = NoiseModel()
    two_qubit_error = depolarizing_error(param, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
    return noise_model


def plot_result(case_id, params, result_0, result_1, ideal_prob):

    colors = list(TABLEAU_COLORS)

    fig = plt.figure()
    fig = plt.figure(dpi=300)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)

    plt.xscale("log")
    # plt.plot(
    #     params,
    #     result_0,
    #     ".",
    #     label='"'
    #     + "0" * (int(case_id[1]) - 1)
    #     + '0" → "'
    #     + "0" * int(case_id[1])
    #     + '0"',
    # )
    plt.plot(
        params,
        result_1,
        "+",
        label='"'
        + "0" * (int(case_id[1]) - 1)
        + '1" → "'
        + "0" * int(case_id[1])
        + '1"',
    )

    ## plt.arrow(0.0018, 0.25, -0.0005, 0, head_width=0.01, head_length=0.0003)
    ## plt.text(0.0022, 0.24, "Sampling noise\n" + "starts to appear...")

    # plt.arrow(0.0006, 0.25, 0.0002, 0, head_width=0.01, head_length=0.0002)
    # plt.text(0.0001, 0.24, "Sampling noise\n" + "starts to appear...")
    # plt.axvline(5.623e-05, linestyle="--", alpha=0.3)

    # plt.axhline(ideal_prob, linestyle="--", alpha=0.3, color=colors[0])
    plt.axhline(0, linestyle="--", alpha=0.3, color=colors[1])

    # plt.ylim([-0.05, 0.5])
    plt.xlabel(r"$\theta / \pi$" + f" ({case_id})")
    plt.xlabel("Depolarizing noise parameter")
    plt.ylabel("Probability amplitude")
    plt.legend(loc="upper left")
    plt.title(f"UQSD with noise in two-qubit gates ({case_id})")
    plt.grid(True, alpha=0.1)

    plt.savefig(
        fname=f"uqsd_optimal_csd_{case_id}.png",
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    nq, ns, seed, case_id, qasm_name, logger = _prepare(__name__)

    qc = QuantumCircuit.from_qasm_file(qasm_name)
    print(qc.count_ops())
    print(qc.depth())
    # TODO Clean up and write a script for full experiments
    # Break these into functions

    # %%
    import sys
    import time
    import tracemalloc
    import qiskit.transpiler
    import qiskit.synthesis
    import qiskit
    from qiskit import transpile

    qc = transpile(qc, basis_gates=["rx", "ry", "rz", "cx"])
    print(int(qc.depth() * 0.05))
    qc_approx = transpile(
        circuits=qc,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config={
            "network_layout": "cart",
            "connectivity_type": "star",
            "depth": int(qc.depth() * 0.05),
        },
    )
    print("Approx")
    print(qc_approx.count_ops())
    print(qc_approx.depth())

    # %%
    ## # [:-15] Remove "no_backend.qasm" and add "ibm_brisbane.qasm"
    ## qc_full = QuantumCircuit.from_qasm_file(qasm_name[:-15] + "ibm_brisbane.qasm")
    ## print(qc_full.count_ops())
    ## print(qc_full.depth())

    # %% [markdown]
    # ## Define handy simulation backends

    from qiskit_aer import AerSimulator

    sv_backend = AerSimulator(method="statevector", seed_simulator=42)

    # Load states
    from problem_spec import *

    states = ProblemSpec.gen_states(
        num_qubits=nq,
        num_states=ns,
        seeds=get_random_seeds(ns, seed=seed),
        state_type="statevector",
    )

    # Construct DUT
    # The circuit with one of the target initial states

    # %%
    from qiskit_aer.library import SetStatevector, set_statevector, SetDensityMatrix
    from qiskit.quantum_info import Statevector

    dut = QuantumCircuit(qc.num_qubits)
    inst = dut.set_statevector(states[0].expand(Statevector([1, 0]))).instructions
    dut.append(inst[0], [_ for _ in range(qc.num_qubits)])
    # dut.save_statevector()
    dut.append(qc, [_ for _ in range(qc.num_qubits)])
    dut.id(0)
    dut.save_density_matrix()
    # dut.save_statevector()
    dut.measure_active()
    dut = dut.decompose(reps=3)
    # Noisy sim extrapolation
    from qiskit.providers import fake_provider

    fake_backend = qiskit.providers.fake_provider.Fake127QPulseV1()

    # %%
    # noise_model =  NoiseModel.from_backend()
    # Define several noise models
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Kraus, SuperOp
    from qiskit_aer import AerSimulator

    # from qiskit.visualization import plot_histogram

    # Import from Qiskit Aer noise module
    from qiskit_aer.noise import (
        NoiseModel,
        QuantumError,
        ReadoutError,
        pauli_error,
        depolarizing_error,
        thermal_relaxation_error,
    )

    # %% [markdown]
    # Even though in reality the noise should be complicated, we assume that if we target a certain type of error and optimize our circuits to perform well under this specific type of noise, it should perform better under general conditions. (Do we have something to support it? Not really. How does more noise increase the cost of simulation?)
    #
    # So we first consider only two-qubit gate errors and try to reduce two-qubit gate counts.
    #
    # Then we add other gate-unrelated noise?
    #
    # Then we add single-qubit gate errors

    # %% [markdown]
    # Let's see how bad the circuit performs with two-qubit gate error

    # %% [markdown]
    # ### How small should the two-qubit depolarizing error be for the circuit to perform well?
    #
    # If you follow the examples and give the param 0.05 or 0.01, the performance is terrible.
    # 1e-4? 1e-8?

    # %%
    # TODO collect different initial states
    # Brute force the plot, please

    # Should we save the noise model to a list? Or just initialize it every time?
    # I guess building DUTs is more expensive?

    # Add references for better parameters?
    # params = [
    #     1e-6, 2e-6, 5e-6, 1e-5,
    #     2e-5, 5e-5, 1e-4, 2e-4,
    #     5e-4, 1e-3, 2e-3, 5e-3,
    #     1e-2, 2e-2, 5e-2, 1e-1
    # ]
    # Check different params
    # params = [0.1 ** (6 - 0.25 * i) for i in range(25)]
    params = [0.1 ** (6 - 0.25 * i) for i in range(10)]

    result_0 = []
    result_1 = []
    for p in params:
        a, b = test_dut(dut, p, [0, 1])
        result_0.append(a)
        result_1.append(b)

    # TODO Save data to .npy or .npz

    plot_result(case_id, params, result_0, result_1, ideal_prob=test_dut(dut, 0, [0]))

    # %%
    # exp_x_axis = linspace(0, 2, num_points)
    # plt.plot(exp_x_axis, prob, ".", label=f"t = {t}", color=colors[t])
    # plt.xlabel(r"$\theta / \pi$" + f" ({num_points} points) (n = {n}, k = {k})")
    # plt.title("First-order Interference (Aer Simulator)")

    # %%
    # Plot the probability amplitudes of different parameters
    # Show the boundary of yes and no
    # Instead of optimizing it, we obtain some intuition
    # Check another file for this experiment
