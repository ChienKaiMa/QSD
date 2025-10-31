import qiskit.qasm2
import qiskit.qasm3
import numpy as np
import qiskit
import qiskit_ibm_runtime


def print_ckt(qsd_method, num_qubits, i):
    filename = f"upload_to_ibmq_2024-07-01/circuits/{qsd_method}_full_{num_qubits}_{i}_20_0.5_transpiled.qasm"
    # print(filename)
    # qiskit.qasm2.exceptions.QASM2ParseError: "UQSD_full_2_1_20_0.5.qasm:37,31: expected a word boundary after a float, but saw 'j'"
    circuit = qiskit.qasm2.load(
        filename=filename,
        custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    circuit.draw(
        idle_wires=False,
        output="mpl",
        filename=f"qq_{qsd_method}_{num_qubits}.png",
    )
    return


def optimize_1():
    service = qiskit_ibm_runtime.QiskitRuntimeService(
        channel="ibm_quantum",
        # instance="ibm-q/open/main",
        instance="ibm-q-hub-ntu/ntu-internal/default",
        token="e421d41292d0977e88ca2900d333e6b6789377af70e1923ba067e97afb929b2da3cd64bba701d1519067002f9c1fabe1e55c47a5539b12d8ec55b85864f6092d",
    )

    ibm_backend = service.backend("ibm_brisbane")

    for qsd_method in ["UQSD", "MED"]:
        print(qsd_method)
        for num_qubits in range(2, 7):
            depth_list = []
            two_qubit_gate_count_list = []
            for i in range(1, 20):
                filename = f"upload_to_ibmq_2024-07-01/circuits/{qsd_method}_full_{num_qubits}_{i}_20_0.5_transpiled.qasm"
                # print(filename)
                # qiskit.qasm2.exceptions.QASM2ParseError: "UQSD_full_2_1_20_0.5.qasm:37,31: expected a word boundary after a float, but saw 'j'"
                circuit = qiskit.qasm2.load(
                    filename=filename,
                    custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
                )
                # circuit = qiskit.qasm3.load(filename)

                circuit = qiskit.transpile(
                    circuit,
                    backend=ibm_backend,
                    approximation_degree=0.0,
                    layout_method="sabre",
                    routing_method="sabre",
                    # unitary_synthesis_method=
                )
                print("Duration =", circuit.duration())

                depth_list.append(circuit.depth())
                two_qubit_gate_count_list.append(circuit.num_nonlocal_gates())
                # Both properties are the same number. (121~125)
                # print("num_connected_components =", circuit.num_connected_components())
                # print("num_tensor_factors =", circuit.num_tensor_factors())
                # if circuit.num_connected_components() != circuit.num_tensor_factors():
                #     print("Different numbers")
                # print("Depth =", circuit.depth())
                # print(circuit.count_ops())

            # Get min, max, avg, median
            print("Min depth =", np.min(depth_list))
            print("Max depth =", np.max(depth_list))
            print("Avg depth =", np.mean(depth_list))
            print("Med depth =", np.median(depth_list))
            print("Min 2q_gates =", np.min(two_qubit_gate_count_list))
            print("Max 2q_gates =", np.max(two_qubit_gate_count_list))
            print("Avg 2q_gates =", np.mean(two_qubit_gate_count_list))
            print("Med 2q_gates =", np.median(two_qubit_gate_count_list))
            # print()

    return


def print_props():
    service = qiskit_ibm_runtime.QiskitRuntimeService(
        channel="ibm_quantum",
        # instance="ibm-q/open/main",
        instance="ibm-q-hub-ntu/ntu-internal/default",
        token="e421d41292d0977e88ca2900d333e6b6789377af70e1923ba067e97afb929b2da3cd64bba701d1519067002f9c1fabe1e55c47a5539b12d8ec55b85864f6092d",
    )

    ibm_backend = service.backend("ibm_brisbane")
    for q in range(2, 5):
        for n in range(3, (2**q - 1)):
            if n > 10:
                break
            depth_list = []
            two_qubit_gate_count_list = []
            for seed in range(0, 5):
                # filename = f"qc_iso_q{q}_n{n}_s{seed}.qasm"
                filename = f"qc_iso_q2_n3_s42.qasm"
                circuit = qiskit.qasm2.load(
                    filename=filename,
                    custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
                )
                circuit = qiskit.transpile(circuit, backend=ibm_backend)
                depth_list.append(circuit.depth())
                two_qubit_gate_count_list.append(circuit.num_nonlocal_gates())
                # Both properties are the same number. (121~125)
                # print("num_connected_components =", circuit.num_connected_components())
                # print("num_tensor_factors =", circuit.num_tensor_factors())
                # if circuit.num_connected_components() != circuit.num_tensor_factors():
                #     print("Different numbers")
                # print("Depth =", circuit.depth())
                # print(circuit.count_ops())

            # Get min, max, avg, median
            print("Min depth =", np.min(depth_list))
            print("Max depth =", np.max(depth_list))
            print("Avg depth =", np.mean(depth_list))
            print("Med depth =", np.median(depth_list))
            print("Min 2q_gates =", np.min(two_qubit_gate_count_list))
            print("Max 2q_gates =", np.max(two_qubit_gate_count_list))
            print("Avg 2q_gates =", np.mean(two_qubit_gate_count_list))
            print("Med 2q_gates =", np.median(two_qubit_gate_count_list))
            print()


if __name__ == "__main__":
    # optimize_1()
    # print_ckt(qsd_method="UQSD", num_qubits=2, i=1)
    # print_ckt(qsd_method="UQSD", num_qubits=3, i=1)
    # print_ckt(qsd_method="UQSD", num_qubits=4, i=1)
    # print_ckt(qsd_method="UQSD", num_qubits=5, i=1)
    print_props()
