import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import random_statevector
num_qubits = 2
state = random_statevector(2**num_qubits, seed=3).data
print(state)
print(np.multiply(state[None].T.conj(), state))
print(np.outer(state, state.conj()))