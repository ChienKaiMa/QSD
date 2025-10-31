from test_scipy_opt_debug import *
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
from argparse import *


def build_circuit(prob: NullSpaceSearchProblem):
    return


if __name__ == "__main__":
    tracemalloc.start()
    parser = ArgumentParser()
    parser.add_argument("-q", "--nqubits", default=2)
    parser.add_argument("-n", "--nstates", default=3)
    parser.add_argument("-s", "--seed", default=42)
    args = parser.parse_args()
    num_qubits = int(args.nqubits)
    num_states = int(args.nstates)
    seed = int(args.seed)
    print("ns =", num_states, ", nq =", num_qubits, ", seed =", seed)

    np.set_printoptions(precision=4)

    prob = NullSpaceSearchProblem(num_qubits=num_qubits, num_states=num_states)
    seeds = get_random_seeds(size=num_states, seed=seed)
    states = NullSpaceSearchProblem.gen_states(
        num_qubits=num_qubits,
        num_states=num_states,
        seeds=seeds,
    )

    t1_start = time.process_time()
    prob.set_states(states)
    prob.expand_solve()
    prob.find_null_spaces()
    prob.build_cons()
    num_samples = 0
    stop = False
    

    while not stop:
        num_samples += 1
        print(num_samples)
        prob.find_init()
        prob.solve(method="SLSQP")
        # prob.solve()
        # prob.x = np.array(x_unit)
        prob.x = prob.norm_vars(prob.x)
        stop = prob.verify()
    # print(num_samples)
    # print(prob.x)

    iso_trans = []
    ops = prob.calc_final_ops()
    for i in range(prob.num_ops):
        print(ops[i])
        iso_trans.append(ops[i])
    # Apply null_space again
    s = null_space(iso_trans)
    print()
    print(iso_trans)
    print()
    print(np.conj(iso_trans).dot(np.array(iso_trans).T))
    print()
    # print(s)
    for _ in s.T:
        iso_trans.append(np.conj(_))

    iso_trans = np.array(iso_trans)
    qc_iso = decompose(iso_trans, scheme="ccd").inverse()

    service = qiskit_ibm_runtime.QiskitRuntimeService(
        channel="ibm_quantum",
        # instance="ibm-q/open/main",
        instance="ibm-q-hub-ntu/ntu-internal/default",
        token="e421d41292d0977e88ca2900d333e6b6789377af70e1923ba067e97afb929b2da3cd64bba701d1519067002f9c1fabe1e55c47a5539b12d8ec55b85864f6092d",
    )

    ibm_backend = service.backend("ibm_brisbane")
    qc_iso = transpile(qc_iso, backend=ibm_backend)

    qiskit.qasm2.dump(
        qc_iso,
        f"qc_iso_q{num_qubits}_n{num_states}_s{seed}.qasm",
    )
    t1_end = time.process_time()

    print("Elapsed time:", t1_end, t1_start)
    print("Elapsed time during the whole program in seconds:", t1_end - t1_start)

    print("Memory (current, peak) =", tracemalloc.get_traced_memory())
    tracemalloc.stop()
    # TODO
    # Decompose and save circuit file
    # qc_iso.draw(
    #     output="mpl",
    #     filename=f"0902_{}.png",
    # )
