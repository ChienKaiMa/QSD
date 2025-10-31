# Visibility and distinguishability

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

# TODO
# quantum eraser
# measure visibility
# measure distinguishability

if __name__ == "__main__":
    num_info_qubits = 2
    num_intercept_qubits = 2
    info_qr = QuantumRegister(num_info_qubits, name="info_qr")
    info_cr = ClassicalRegister(num_info_qubits, name="info_cr")
    intercept_qr = QuantumRegister(num_intercept_qubits, name="intercept_qr")
    qc = QuantumCircuit(info_qr, intercept_qr, info_cr)
    # Prepare initial state
    # For example, a two-qubit Bell state
    qc.h(info_qr[0])
    qc.cx(info_qr[0], info_qr[1])
    qc.barrier()
    # TODO What is this?
    qc.h(info_qr[0])
    qc.h(info_qr[1])
    qc.barrier()
    # Inject eraser
    qc.cx(info_qr[0], intercept_qr[0])
    qc.cx(info_qr[1], intercept_qr[1])
    qc.barrier()
    # Phase shift
    phi_0 = np.pi / 2
    phi_1 = np.pi / 2
    qc.p(phi_0, info_qr[0])
    qc.p(phi_1, info_qr[1])
    qc.h(info_qr[0])
    qc.h(info_qr[1])
    qc.measure(info_qr[0], info_cr[0])
    qc.measure(info_qr[1], info_cr[1])
    # Undo the effect of the eraser (measure the info before or after the phase shift)
    # Or reveal a different order of interference
    print(qc)
    qc.draw(
        output="mpl",
        filename="temp/vis.png",
    )
