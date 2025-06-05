# Aims to be handy, not generalized

from utils.printing import *
from utils.sic_expand import *
from utils.sic_fiducial_state import *
from qutip import coherent, coherent_dm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
import numpy as np


def sv_simple_1(theta = np.pi / 4):
    """Two quantum states with a pre-specified absolute value of
    inner product."""

    # Initialize |0⟩ state
    qc0 = QuantumCircuit(1)  # 1 qubit, |0⟩ by default
    statevector_0 = Statevector(qc0).data

    # Initialize θ-rotated state (using Ry rotation)
    qc_rot = QuantumCircuit(1)
    qc_rot.ry(theta, 0)
    statevector_rot = Statevector(qc_rot).data

    states = []
    states.append(statevector_0)
    states.append(statevector_rot)
    return {"num_qubits": 1, "num_states": 2, "states": states}


def sv_sic_symm_small():
    """Return statevectors derived from SIC POVM.
    The states are symmetric in some ways, and
    the number of qubits is small.
    """
    sic_states_psi = generate_sic_states(fiducial_states[4], 4)
    sic_states = [
        Statevector(sic_states_psi[0]),
        Statevector(sic_states_psi[4]),
        Statevector(sic_states_psi[8]),
    ]
    return {"num_qubits": 2, "num_states": 3, "states": sic_states}


def sv_coh_symm_small(num_qubits: int = 3):
    assert num_qubits > 0
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    alphas = [np.exp(angles[i] * 1j) for i in range(len(angles))]

    symm_states_1 = [
        coherent(N=2**num_qubits, alpha=1 * alphas[i])
        for i in range(len(alphas))
    ]
    symm_states_1_dm = [
        coherent_dm(N=2**num_qubits, alpha=1 * alphas[i])
        for i in range(len(alphas))
    ]
    symm_states_matrix_1 = [
        symm_states_1[i].data.to_array().flatten()
        for i in range(len(symm_states_1))
    ]
    symm_states_1_dm_matrix = [
        symm_states_1_dm[i].data.to_array()
        for i in range(len(symm_states_1_dm))
    ]

    return symm_states_matrix_1, symm_states_1_dm


def dm_coh_symm_small(num_qubits=3):
    assert num_qubits > 0
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    alphas = [np.exp(angles[i] * 1j) for i in range(len(angles))]

    symm_states_1_dm = [
        coherent_dm(N=2**num_qubits, alpha=1 * alphas[i])
        for i in range(len(alphas))
    ]
    symm_states_1_dm_matrix = [
        symm_states_1_dm[i].data.to_array()
        for i in range(len(symm_states_1_dm))
    ]
    return symm_states_1_dm_matrix


def sv_coh_symm_std():
    states = []
    return states
