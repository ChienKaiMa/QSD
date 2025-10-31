# Aims to be handy, not generalized

from utils.printing import *
from utils.sic_expand import *
from utils.sic_fiducial_state import *
from qutip import coherent, coherent_dm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
import numpy as np


def sv_simple_1(theta=np.pi / 4):
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


def simple_2(a=1 / 10, b=1 / 6, c=1 / 5):
    num_qubits = 2
    num_states = 3
    z00 = np.array([1, 0, 0, 0])
    z01 = np.array([0, 1, 0, 0])
    z10 = np.array([0, 0, 1, 0])
    z11 = np.array([0, 0, 0, 1])

    def normalize(v):
        return np.array(v) / np.linalg.norm(v)

    numpy_state_vec = [
        normalize(z00 + np.multiply(a, z11)),
        normalize(z01 + np.multiply(b, z11)),
        normalize(z10 + np.multiply(c, z11)),
    ]
    numpy_dense_mat = [
        DensityMatrix(numpy_state_vec[i]).data
        for i in range(len(numpy_state_vec))
    ]
    return {
        "num_qubits": num_qubits,
        "num_states": num_states,
        "state_vec": numpy_state_vec,
        "dense_mat": numpy_dense_mat,
    }


def sv_sic_asymm_small():
    """Return statevectors derived from SIC POVM.
    The states are symmetric in some ways, and
    the number of qubits is small.
    """
    sic_states_psi = generate_sic_states(fiducial_states[4], 4)
    sic_states = [
        Statevector(sic_states_psi[0]),
        Statevector(sic_states_psi[1]),
        Statevector(sic_states_psi[2]),
    ]
    return {"num_qubits": 2, "num_states": 3, "states": sic_states}


def get_coherent_states(num_qubits, amps, angles):
    alphas = [amps[i] * np.exp(angles[i] * 1j) for i in range(len(angles))]
    qutip_state_vec = [
        coherent(N=2**num_qubits, alpha=alphas[i]) for i in range(len(alphas))
    ]
    qutip_dense_mat = [
        coherent_dm(N=2**num_qubits, alpha=alphas[i])
        for i in range(len(alphas))
    ]
    numpy_state_vec = [
        qutip_state_vec[i].data.to_array().flatten()
        for i in range(len(qutip_state_vec))
    ]
    numpy_dense_mat = [
        qutip_dense_mat[i].data.to_array() for i in range(len(qutip_dense_mat))
    ]
    return {
        "num_qubits": num_qubits,
        "num_states": len(alphas),
        "state_vec": numpy_state_vec,
        "dense_mat": numpy_dense_mat,
        "qutip_dense_mat": qutip_dense_mat,  # For plotting and analysis
    }


def coh_asymm_small(num_qubits: int = 3):
    assert num_qubits > 0
    amps = [1, 1, 1]
    angles = [0, np.pi / 3, 2 * np.pi / 3]
    return get_coherent_states(num_qubits, amps, angles)


def coh_symm_small(num_qubits: int = 3):
    assert num_qubits > 0
    amps = [1, 1, 1]
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    return get_coherent_states(num_qubits, amps, angles)


def sv_coh_symm_std():
    states = []
    return states
