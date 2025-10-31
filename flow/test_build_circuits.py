import numpy as np
import qclib.isometry
import scipy.linalg
from itertools import combinations
from functools import partial

# TODO Redesign and save the important parts
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize
from scipy.linalg import null_space

# import cobyqa
from qiskit.quantum_info import (
    random_statevector,
    random_density_matrix,
    Statevector,
    DensityMatrix,
)

# from qiskit.synthesis import qs_decomposition
# from qiskit.circuit.library import Isometry

# copied from build_circuits.py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import qiskit
import qiskit_ibm_runtime
import qclib
import random
from datetime import datetime
import os
import time
import tracemalloc

from math import pi
from scipy.spatial.transform import Rotation as R
from scipy.linalg import qz

from qiskit import transpile
from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector
from qiskit.circuit import Parameter
from qiskit.providers.basic_provider import BasicProvider
from qiskit.primitives import Sampler
import qiskit.qasm2

from qiskit_aer import AerProvider

from qclib.isometry import decompose
from qclib.state_preparation import UCGInitialize, IsometryInitialize

# from qiskit.quantum_info import *
import sys

sys.path.append("../")
sys.path.append("./")
from problem_spec import *
from temp.get_random_seeds import *

# https://stackoverflow.com/questions/55132107/scipy-fitting-with-parameters-in-a-vector
from operator import add

import cvxpy as cp


def is_pos_semidef(x):
    # print(np.linalg.eigvals(x))
    return np.all(np.linalg.eigvals(x) >= 0)


# Modified from 2024 JL supplement materials
class POVM:
    """Base class that holds an arbitrary POVM <povm> as a list of <N> POVM elements."""

    def __init__(self, povm):
        """
        Constructor asserts that the given POVM is valid.
        """
        self.povm = povm
        self.N = len(povm)
        self.depth = int(np.ceil(np.log2(self.N)))  # required depth of the binary tree
        self.povm_dim = len(povm[0])  # dimension of the POVM operators
        self.n_qubits = int(np.log2(self.povm_dim))  # number of system qubits
        self.assert_valid()

    def assert_valid(self):
        """Verifies the hermiticity, positivity of the POVM and that
        the POVM resolves the identity.
        Returns:
            True: if all conditions are satisfied
        Raises:
            Assertion Error: if one of conditions is not satisfied
        """
        for E in self.povm:
            assert np.allclose(E.conj().T, E), "Some POVM elements are not hermitian"
            assert np.all(
                np.linalg.eigvals(E).round(3) >= 0
            ), "Some POVM elements are not positive semi-definite"
        assert np.allclose(
            sum(self.povm), np.eye(self.povm_dim), atol=1e-04
        ), "POVM does not resolve the identity"

        # TODO: extend to arbitrary N by padding with zero operators and possibly rearranging
        assert (
            self.N & (self.N - 1)
        ) == 0, "Number of POVM elements is not a power of 2"

    def get_measurement_op(self, start, end):
        """
        Returns a cumulative measurement operator by grouping together
        POVM elements from povm[start] to povm[end-1]
        """
        return np.sum(self.povm[start:end], axis=0).round(5)

    def get_diagonalization(self, start, end):
        """
        Returns:
            Kraus operator <M>, diagonal <D> and the modal matrix <V> for a given
            measurement operator by diagonalizing the measurement operator in the form:
            M = V@D@Vh such that Mh@M = E
        """
        E = self.get_measurement_op(start, end)
        d2, V = np.linalg.eig(E)
        D = np.diag(np.real(np.sqrt(d2)))
        M = V @ D @ np.linalg.inv(V)
        return M, D, V


class POVMCircuit:
    def __init__(self, povm_vectors=None, povm=None):
        self.povm_vectors = povm_vectors  # Assume rank-1 POVM
        self.povm = povm
        # TODO self.num_qubits = -1
        # TODO self.num_states = -1
        # TODO self.seed = 0
        return

    @classmethod
    def load(cls, case_id: str, num_qubits, num_states, seed):
        obj = cls.__new__(cls)  # Does not call __init__
        obj.num_qubits = num_qubits
        obj.num_states = num_states
        obj.seed = seed
        obj.case_id = case_id
        obj.num_amps = 2**num_qubits
        obj.povm_vectors = np.load(f"povm_{case_id}.npy")
        print(obj.povm_vectors.round(3))
        return obj

    def expand_precursor():
        return

    def fix(self):
        # Check the dimension of the POVM and calculate the rest
        logger = logging.getLogger(__name__)
        logger.info(f"# of qubits = {self.num_qubits}")
        logger.info(f"# of povm vectors = {len(self.povm_vectors)}")
        the_rest_povm = np.eye(self.num_amps, dtype="complex128")
        for i in range(len(self.povm_vectors)):
            op = np.multiply(self.povm_vectors[i][None].T.conj(), self.povm_vectors[i])
            logger.info(f"Operator is PSD?: {is_pos_semidef(op)}")
            the_rest_povm -= op
        logger.info(f"Operator is PSD?: {is_pos_semidef(the_rest_povm)}")
        u, s, v = np.linalg.svd(the_rest_povm, hermitian=True)
        # print()
        # print("the_rest_povm")
        # print(the_rest_povm.round(3))
        # print("u")
        # print(u.round(3))
        # print("s")
        # print(s.round(3))
        # print()
        for i in range(self.num_amps):
            # TODO assertion
            if s[i] >= 1e-4:
                last_povm = u[:, i] * np.sqrt(s[i])
                # print(last_povm)
                # self.povm_vectors = np.vstack((self.povm_vectors, last_povm.conj()))
                self.povm_vectors = np.vstack((self.povm_vectors, last_povm.conj()))
                # self.povm_vectors.append(last_povm.conj())
                check_op = np.multiply(last_povm[None].T.conj(), last_povm)
                # print(check_op.round(3))
        return

    def build_circuit(self):
        logger = logging.getLogger(__name__)

        isometry = self.naimark(self.povm_vectors)
        np.save(f"iso_{case_id}.npy", isometry)
        logger.info(f"The isometry is saved to iso_{case_id}.npy")
        ## Isometry with Qiskit
        ## iso = Isometry(
        ##     isometry,
        ##     num_ancillas_zero=0,
        ##     num_ancillas_dirty=0,
        ## )
        ## qc_iso = QuantumCircuit(3)
        ## qc_iso.append(iso, [0, 1, 2])

        # 2024/10/30
        # csd works, while ccd doesn't
        scheme = "ccd"
        logger.info(f"Transpile the isometry to quantum circuit using {scheme}")
        # qc_iso = decompose(isometry, scheme=scheme).inverse()

        lines = isometry.shape[0]
        cols = isometry.shape[1]

        log_lines = int(np.log2(lines))
        log_cols = int(np.log2(cols))
        unitary_gate = qclib.isometry._extend_to_unitary(isometry, log_lines, log_cols)
        qc_iso = decompose(unitary_gate, scheme=scheme).inverse()

        from qiskit.circuit.library import UnitaryGate

        # TODO unitary_gate.
        # UnitaryGate(data=unitary_gate)
        service = qiskit_ibm_runtime.QiskitRuntimeService(
            channel="ibm_quantum",
            # instance="ibm-q/open/main",
            instance="ibm-q-hub-ntu/ntu-internal/default",
            token="e421d41292d0977e88ca2900d333e6b6789377af70e1923ba067e97afb929b2da3cd64bba701d1519067002f9c1fabe1e55c47a5539b12d8ec55b85864f6092d",
        )

        t_start = time.time()
        qc_approx = transpile(
            circuits=qc_iso,
            unitary_synthesis_method="aqc",
            unitary_synthesis_plugin_config={
                "network_layout": "cart",
                "connectivity_type": "star",
                "depth": int(qc_iso.depth() * 0.5),
            },
        )
        qc_approx = qc_approx.decompose(reps=3)
        t_end = time.time()
        logger.info(
            f"Qiskit approximation time (rounded) = {round((t_end - t_start), 4)} seconds"
        )
        logger.info(f"count_ops {qc_approx.count_ops()}")
        logger.info(f"Depth {qc_approx.depth()}")
        qiskit.qasm2.dump(
            qc_approx,
            f"qc_iso_{self.case_id}_approx_no_backend.qasm",
        )
        logger.info(
            f"The quantum circuit is saved to qc_iso_{self.case_id}_approx_no_backend.qasm"
        )

        # Transpile first without the backend to avoid strange errors
        # qclib -> qiskit
        t_start = time.time()
        qc_iso = transpile(qc_iso)
        t_end = time.time()
        logger.info(
            f"qclib decompose time (rounded) = {round((t_end - t_start), 4)} seconds"
        )

        t_start = time.time()
        qc_iso = qc_iso.decompose(reps=3)
        t_end = time.time()
        logger.info(f"Decompose the qclib isometry circuit without backend")
        logger.info(f"Decompose time (rounded) = {round((t_end - t_start), 4)} seconds")
        logger.info(f"count_ops {qc_iso.count_ops()}")
        logger.info(f"Depth {qc_iso.depth()}")
        qiskit.qasm2.dump(
            qc_iso,
            f"qc_iso_{self.case_id}_no_backend.qasm",
        )
        logger.info(
            f"The quantum circuit is saved to qc_iso_{self.case_id}_no_backend.qasm"
        )

        # Transpile with the backend
        backend_name = "ibm_brisbane"
        ibm_backend = service.backend(backend_name)
        t_start = time.time()
        qc_iso = transpile(qc_iso, backend=ibm_backend)
        t_end = time.time()
        logger.info(f"Keep transpiling the isometry circuit with {backend_name}")
        logger.info(f"Decompose time (rounded) = {round((t_end - t_start), 4)} seconds")
        logger.info(f"count_ops {qc_iso.count_ops()}")
        logger.info(f"Depth {qc_iso.depth()}")
        qiskit.qasm2.dump(
            qc_iso,
            f"qc_iso_{self.case_id}_{backend_name}.qasm",
        )
        logger.info(
            f"The quantum circuit is saved to qc_iso_{self.case_id}_{backend_name}.qasm"
        )

        return

    def basis_extend(self, i, dims):
        a = np.zeros(dims)
        a[i] = 1
        # https://stackoverflow.com/questions/11885503/numpy-transpose-of-1d-array-not-giving-expected-result
        # print(np.matrix(a).T)
        return np.matrix(a).T

    def naimark(self, basis):
        V = sum(
            [
                np.multiply(
                    # self.basis_extend(i, self.num_amps * 2), np.matrix.getH(basis[i])
                    self.basis_extend(i, self.num_amps * 2),
                    basis[i],
                )
                for i in range(len(basis))
            ]
        )
        return V

    pass


if __name__ == "__main__":
    # Example
    parser = ArgumentParser()
    parser.add_argument("-q", "--nqubits", default=2)
    parser.add_argument("-n", "--nstates", default=3)
    parser.add_argument("-s", "--seed", default=42)
    args = parser.parse_args()
    nq = int(args.nqubits)
    ns = int(args.nstates)
    seed = int(args.seed)
    case_id = f"q{nq}_n{ns}_s{seed}"

    logging.basicConfig(
        filename=f"build_circuits_{case_id}.log",
        filemode="a",
        format="{asctime} {levelname} {filename}:{lineno}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=logging.INFO,  # Qiskit dumps too many DEBUG messages
        encoding="utf-8",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Start build_circuits.py")
    logger.info(f"nq = {nq}, ns = {ns}, seed = {seed}")

    tracemalloc.start()
    obj = POVMCircuit.load(case_id, nq, ns, seed)
    obj.fix()
    obj.build_circuit()
    logger.info(f"Memory (current, peak, in bytes) = {tracemalloc.get_traced_memory()}")
    tracemalloc.stop()
