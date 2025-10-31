# # Optimizing QSD circuits
# Assume no backend


import sys

sys.path.append("../")
from utils.utils import *


from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit import transpile


def get_unitary_circuit(qc) -> QuantumCircuit:
    # https://docs.quantum.ibm.com/guides/synthesize-unitary-operators#synthesize-unitary-operations

    U = Operator(qc)

    nq = qc.num_qubits
    unitary_circuit = QuantumCircuit(nq)
    unitary_circuit.unitary(U, range(nq))
    return unitary_circuit


def resynth_unitary(qc) -> QuantumCircuit:
    return get_unitary_circuit(qc).decompose(reps=3)


def resynth_unitary_approx(qc) -> QuantumCircuit:
    tmp_circuit = get_unitary_circuit(qc)

    # The approximation degree defaults to 1.0
    approx_circuit = transpile(
        tmp_circuit,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config={"seed": 42},
        approximation_degree=1.0,
        seed_transpiler=11,
    )

    return approx_circuit.decompose(reps=3)


def resynth_unitary_approx_max(qc) -> QuantumCircuit:
    tmp_circuit = get_unitary_circuit(qc)

    # It won't work without specifying aqc
    # It won't work without specifying basis_gates?
    approx_circuit = transpile(
        tmp_circuit,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config={"seed": 42},
        approximation_degree=0,
        basis_gates=["u", "cx"],
        seed_transpiler=11,
    )

    return approx_circuit.decompose(reps=3)


def resynth_aqc(qc, uni_synth_config=None) -> QuantumCircuit:
    tmp_circuit = get_unitary_circuit(qc)

    aqc_circuit = transpile(
        tmp_circuit,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config=uni_synth_config,
        approximation_degree=0,
        seed_transpiler=11,
    )

    return aqc_circuit.decompose(reps=3)


def try_synth(qc, case_id):
    # Show different results (depth and fidelity)
    resynth_circuit = resynth_unitary(qc)
    approx_circuit = resynth_unitary_approx(qc)
    approx_max_circuit = resynth_unitary_approx_max(qc)
    aqc_circuit_v0 = resynth_aqc(
        qc,
        {
            # "network_layout": "cart",
            "connectivity_type": "star",
            "depth": 10,
            "seed": 3,
        },
    )
    # TODO Approx again
    # Statistics

    print(qc.count_ops())
    print(qc.depth())
    print(resynth_circuit.count_ops())
    print(resynth_circuit.depth())
    print(approx_circuit.count_ops())
    print(approx_circuit.depth())
    print(approx_max_circuit.count_ops())
    print(approx_max_circuit.depth())
    print(aqc_circuit_v0.count_ops())
    print(aqc_circuit_v0.depth())

    from qiskit.quantum_info import process_fidelity

    # Two operators which differ only by phase
    op_a = Operator(qc)
    op_b = Operator(resynth_circuit)
    op_c = Operator(approx_circuit)
    op_c_1 = Operator(approx_max_circuit)
    op_d = Operator(aqc_circuit_v0)

    # Compute process fidelity
    F_resynth = process_fidelity(op_a, op_b)
    print("Process fidelity (resynth) =", F_resynth)
    F_approx = process_fidelity(op_a, op_c)
    print("Process fidelity (approx) =", F_approx)
    F_approx_max = process_fidelity(op_a, op_c_1)
    print("Process fidelity (approx_max) =", F_approx_max)
    F_aqc = process_fidelity(op_a, op_d)
    print("Process fidelity (aqc) =", F_aqc)

    import qiskit.qasm2

    qiskit.qasm2.dump(
        resynth_circuit,
        f"qc_iso_{case_id}_no_backend_resynth.qasm",
    )
    qiskit.qasm2.dump(
        approx_circuit,
        f"qc_iso_{case_id}_no_backend_approx.qasm",
    )
    qiskit.qasm2.dump(
        approx_circuit,
        f"qc_iso_{case_id}_no_backend_approx_max.qasm",
    )
    qiskit.qasm2.dump(
        aqc_circuit_v0,
        f"qc_iso_{case_id}_no_backend_aqc.qasm",
    )
    return


try_synth(
    QuantumCircuit.from_qasm_file("../coh_symm_q4_n3_optuqsd_reducedpovm_ccd_no_backend.qasm"),
    "coh_q5_n3",
)

