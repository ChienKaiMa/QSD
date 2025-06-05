import sys

sys.path.append("../")
sys.path.append("./")
from flow.problem_spec import *
from temp.get_random_seeds import *

import time
import tracemalloc
import numpy as np

import qiskit_ibm_runtime
import qiskit
import qiskit.qasm2
from qiskit import transpile

# from qiskit.circuit.library import Isometry
# from qiskit.synthesis import qs_decomposition
import qclib.isometry

service = qiskit_ibm_runtime.QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-hub-ntu/jiang-jie-hong/default',
    token="b49854f9530507490413d6c62e39bfe88adeb005bca7c3e363ef889fd44bb4e751dba1f9ba90d7dd5269e220c176e9d6f0d80fa5489945a2a5a19466ac51f543",
)


def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)


class POVMCircuit:
    def __init__(self, povm_vectors=None, povm=None):
        self.povm_vectors = povm_vectors  # Assume rank-1 POVM
        self.povm = povm
        return

    @classmethod
    def load(cls, num_qubits, num_states, seed, case_id: str):
        """A suggested way to initialize"""
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
        """"""
        # TODO Check its correctness
        # clean
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
        np.save(f"iso_{self.case_id}.npy", isometry)
        logger.info(f"The isometry is saved to iso_{self.case_id}.npy")

        # 2024/10/30
        # csd works, while ccd doesn't
        # 2024/11/11
        # ccd works by extending to unitary first
        scheme = "ccd"
        logger.info(f"Transpile the isometry to quantum circuit using {scheme}")
        if scheme == "ccd":
            lines = isometry.shape[0]
            cols = isometry.shape[1]

            log_lines = int(np.log2(lines))
            log_cols = int(np.log2(cols))
            unitary_gate = qclib.isometry._extend_to_unitary(
                isometry, log_lines, log_cols
            )
            qc_iso = qclib.isometry.decompose(unitary_gate, scheme=scheme)
            from qiskit.circuit.library import UnitaryGate

            # TODO unitary_gate
            # UnitaryGate(data=unitary_gate)
        else:
            qc_iso = qclib.isometry.decompose(isometry, scheme=scheme)

        # Trying approximate compiling
        self.approx(logger, qc_iso)

        # Transpile first without the backend to avoid strange errors
        # qclib -> qiskit
        qc_iso = self.transpile_wo_backend(logger, qc_iso)

        # Transpile with the backend
        self.transpile_with_backend(logger, qc_iso, service)

        return

    def transpile_wo_backend(self, logger, qc_iso):
        """Transpile without backend"""
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

        return qc_iso

    def transpile_with_backend(self, logger, qc_iso, service):
        backend_name = "ibm_fez"
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

    def approx(self, logger, qc_iso):
        # TODO make it work
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

    def basis_extend(self, i, dims):
        a = np.zeros(dims)
        a[i] = 1
        # https://stackoverflow.com/questions/11885503/numpy-transpose-of-1d-array-not-giving-expected-result
        # print(np.matrix(a).T)
        return np.matrix(a).T

    def naimark(self, basis):
        # TODO clean
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


def _prepare(__name__):
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
    return nq, ns, seed, case_id, logger


if __name__ == "__main__":
    nq, ns, seed, case_id, logger = _prepare(__name__)

    tracemalloc.start()
    obj = POVMCircuit.load(nq, ns, seed, case_id)
    obj.fix()
    obj.build_circuit()
    logger.info(f"Memory (current, peak, in bytes) = {tracemalloc.get_traced_memory()}")
    tracemalloc.stop()
