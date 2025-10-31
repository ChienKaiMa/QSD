from qiskit.synthesis import aqc as aqc
import numpy as np

aqc.FastCNOTUnitObjective(num_qubits=3, cnots=np.array())
aqc.CNOTUnitCircuit(
    num_qubits=3,
    cnots=10,
    tol=1e-5,
    name="cutie",
)