# Initialize state
import qclib.isometry
import qiskit
import qiskit.circuit
import qclib
import time


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import qiskit
import qclib
import random
from datetime import datetime
import os

from math import pi
from scipy.spatial.transform import Rotation as R

from qiskit import transpile
from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector
from qiskit.circuit import Parameter
from qiskit.providers.basic_provider import BasicProvider
from qiskit.primitives import Sampler

from qiskit_aer import AerProvider

from qclib.isometry import decompose
from qclib.state_preparation import UCGInitialize, IsometryInitialize


def qclib_impl(num_qubits, inner_product=0.5, scheme="ccd"):
    rand_float = random.uniform(0, 2 * pi)

    v1 = random_statevector(2**num_qubits)
    v2 = random_statevector(2**num_qubits)
    v1p = (v2 - np.vdot(v1, v2) * v1) / np.sqrt(1 - np.abs(np.vdot(v1, v2)) ** 2)

    # v1p = v1p/np.linalg.norm(v1p)
    v2: Statevector = (
        inner_product * np.exp(1j * rand_float) * v1
        + np.sqrt(1 - (inner_product**2)) * v1p
    )
    v1_e = v1.copy()
    v2_e = v2.copy()
    v1_e = v1_e.expand(Statevector([1, 0]))
    v2_e = v2_e.expand(Statevector([0, 1]))

    # ----- restrict to one-qubit gate -----#
    iso_trans = np.array([v1, v1p]).T
    qc_iso = qclib.isometry.decompose(iso_trans, scheme=scheme).inverse()
    print(f"Decomposition ({num_qubits} qubits, {scheme}) success!")
    return


def test_fix_v1(num_qubits, inner_product=0.5, scheme="ccd"):
    # qiskit.circuit.library.Initialize()
    rand_float = random.uniform(0, 2 * pi)

    v1 = random_statevector(2**num_qubits)
    v2 = random_statevector(2**num_qubits)
    v1p = (v2 - np.vdot(v1, v2) * v1) / np.sqrt(1 - np.abs(np.vdot(v1, v2)) ** 2)

    # v1p = v1p/np.linalg.norm(v1p)
    v2: Statevector = (
        inner_product * np.exp(1j * rand_float) * v1
        + np.sqrt(1 - (inner_product**2)) * v1p
    )
    v1_e = v1.copy()
    v2_e = v2.copy()
    v1_e = v1_e.expand(Statevector([1, 0]))
    v2_e = v2_e.expand(Statevector([0, 1]))

    # ----- restrict to one-qubit gate -----#
    iso_trans = np.array([v1, v1p]).T
    # Default _EPS = 1e-10
    iso = qiskit.circuit.library.Isometry(iso_trans)
    # iso.add_decomposition()
    print(iso.definition)
    return


def test_fix_v2():
    # qiskit.circuit.library.Initialize()
    return


def plot_time(test_range=range(1, 11), scheme="ccd"):
    time_data = []
    for i in test_range:
        start_time = time.time()
        qclib_impl(num_qubits=i, scheme=scheme)
        end_time = time.time()
        execution_time = end_time - start_time
        time_data.append([i, execution_time])
    np.savetxt(f"{scheme}_decomp_time.csv", time_data, fmt="%.16f", delimiter=",")


if __name__ == "__main__":
    plot_time(test_range=range(1, 11), scheme="ccd")
    plot_time(test_range=range(1, 10), scheme="csd")
    plot_time(test_range=range(2, 13), scheme="knill")
    # for i in range(1, 11):
    #     try:
    #         to_fix(num_qubits=i, scheme="csd")
    #     except:
    #         print(f"Decomposition ({i} qubits, csd) Failed")
    #
    # for i in range(1, 11):
    #     try:
    #         to_fix(num_qubits=i, scheme="knill")
    #     except:
    #         print(f"Decomposition ({i} qubits, csd) Failed")
    # for scheme in ["ccd", "csd", "knill"]:
    #     try:
    #         to_fix(num_qubits=11, scheme=scheme)
    #     except:
    #         print(f"Decomposition (11 qubits, {scheme}) Failed")

    # to_fix(num_qubits=11, scheme="csd")
    # to_fix(num_qubits=10, scheme="knill")
    # to_fix(num_qubits=11, scheme="knill")

    # for i in range(10, 21):
    #     try:
    #         to_fix(num_qubits=i, scheme="knill")
    #     except:
    #         print(f"Decomposition ({i} qubits, knill) Failed")
    #
