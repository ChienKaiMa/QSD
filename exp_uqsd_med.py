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


# --------------------------------#
##        Vector analysis        ##
# --------------------------------#
#  Specify the vector by np.array().   #


def vec_to_array(v_dict, n_qbit):
    """
    Given a dict that specifies a state vector, produce the corresponding list.
    The output is in the conventional qubit order. (ref. Michael A. Nielsen, Isaac L. Chuang)
    """
    return np.array([v_dict.get(format(i, f"0{n_qbit}b"), 0) for i in range(2**n_qbit)])


def unit_vector_check(vector):
    """Normalize a vector if it's not already a unit vector."""
    norm = np.linalg.norm(vector)
    return vector / norm if abs(norm - 1) > 1e-6 else vector


# ----------------------------------------------#
##                  UQSD Gate                 ##
# ----------------------------------------------#
class QSD:
    def __init__(self, v1, v2):
        inner_v1v2 = np.vdot(v1, v2)
        self.inner = np.abs(inner_v1v2)

    def zyz_decomposition(self):
        angle_zx = np.arctan(np.sqrt(2 * self.inner / (1 - self.inner)))
        angle_y = (np.arccos(self.inner) / 2) - (pi / 4)
        rot_matrix_1 = R.from_rotvec(angle_zx * np.array([-1, 0, 1]) / np.sqrt(2))
        rot_matrix_2 = R.from_rotvec(angle_y * np.array([0, 1, 0]))
        rot_zyz = rot_matrix_1 * rot_matrix_2
        alpha, beta, gamma = rot_zyz.as_euler("zyz")
        return alpha, beta, gamma

    def med_initialize(self, probability):
        p1 = probability
        p2 = 1 - probability
        inner = self.inner
        numerator = 2 * p2 * inner * np.sqrt(1 - (inner**2))
        denominator = 1 - 2 * p2 * (inner**2) + np.sqrt(1 - 4 * p1 * p2 * (inner**2))
        alpha = 2 * np.arctan(numerator / denominator)
        # alpha = 2 * np.arctan2(numerator,denominator)

        return alpha

    def uqsd_circuit(self):
        alpha, beta, gamma = self.zyz_decomposition()
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cry(2 * alpha, 0, 1)
        qc.x(0)
        qc.x(1)
        qc.cry(-2 * beta, 1, 0)
        qc.x(1)
        qc.x(0)
        qc.cry(2 * gamma, 0, 1)
        qc.x(0)
        return qc.to_gate(label="UQSD")

    def med_circuit(self, probability):
        qc = QuantumCircuit(1)
        qc.ry(self.med_initialize(probability), 0)
        return qc.to_gate(label="MED")


def data_analysis(data: list, method: str):
    """
    Analyze the success rate of the quantum state discrimination.

    Parameters

    data(list) - Outcome probability distrobution. If the circuit
    input is pure state ("Fixed"), then join the two circuit
    outcome list as the input.

    method(str) - "UQSD" or "MED".


    Returns

    A dictionary of success rate.

    keys and values:
    psucc - Overall success rate.
    If method is "UQSD", then psucc = p1 + p2 + pu/2.
    If method is "MED", then psucc = p1 + p2.

    p1 - Success rate of guessing |v1>.
    p2 - Success rate of guessing |v2>.
    pu - Unknown outcome rate for UQSD.
    """
    len1 = int(len(data) / 2)
    if method in ["MED", "med"]:
        p1 = data[0]
        p2 = data[len1 + 1]
        psucc = p1 + p2

    elif method in ["UQSD", "uqsd"]:
        p1 = data[0]
        p2 = data[len1 + 1]
        pu = data[2] + data[len1 + 2]
        psucc = p1 + p2 + (pu / 2)

    pdict = {"psucc": psucc, "p1": p1, "p2": p2}

    if method in ["UQSD", "uqsd"]:
        pdict["pu"] = pu

    return pdict


def build_qsd_experiment_circuit(
    num_qubit: int,
    inner_product: float,
    qsd_method: str,
    rand_init_state: bool,
    probability=0.5,
):
    """
    Setting the experiment setup.

    Parameters

    num_qubit(int) - The qubit number.

    inner_product(float) -  The inner product of two given state.

    method(str) - Quantum discrimination method, "UQSD" or "MED".

    rand_init_state(bool) - The input states are given by a pure state
    or mixed state. True is "Random" for mixed state. False is "Fixed" for pure state.

    probability(float) - The probability of the given state |v1>.
    The default is 0.5. UQSD will not be affected by the probability,
    but MED will be.

    **karg

    Returns

    A list of qiskit circuits.
    """

    rand_float = random.uniform(0, 2 * pi)

    v1 = random_statevector(2**num_qubit)
    v2 = random_statevector(2**num_qubit)
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
    qc_iso = decompose(iso_trans, scheme="ccd").inverse()

    if rand_init_state:
        v_mix = np.sqrt(probability) * v1_e + np.sqrt(1 - probability) * v2_e
        # qc_pre = [UCGInitialize(v_mix)]
        qc_pre = [UCGInitialize(v_mix).definition]
        # qc_pre = [IsometryInitialize(v_mix).definition.decompose()]
        qc_bit_num = num_qubit + 1
    else:
        qc_pre = [UCGInitialize(v1).definition, UCGInitialize(v2).definition]
        qc_bit_num = num_qubit
    # else:
    #     raise ValueError("initial_state should be specified as Random or Fixed")

    if qsd_method in ["UQSD", "uqsd"]:
        qc_QSD = QSD(v1, v2).uqsd_circuit()
        append_list = [0, 1]
    elif qsd_method in ["MED", "med"]:
        qc_QSD = QSD(v1, v2).med_circuit(probability)
        append_list = [0]
    else:
        raise ValueError("Discrimination method should be chosen from UQSD and MED")

    # Show circuit
    exp_folder = str(datetime.now().date())
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
        os.mkdir(f"{exp_folder}/figures")
        os.mkdir(f"{exp_folder}/results")
    qc_QSD.definition.draw(
        output="mpl",
        filename=f"./{exp_folder}/figures/{qsd_method}_block.png",
    )
    plt.close()

    # ----- final circuit -----#
    qc_total = []
    for sub in qc_pre:
        qc = QuantumCircuit(qc_bit_num)
        qc.compose(sub, list(range(qc_bit_num)), inplace=True)
        qc.barrier()
        qc.compose(qc_iso, list(range(num_qubit)), inplace=True)
        qc.p(rand_float, 0)
        qc.barrier()
        qc.compose(qc_QSD, append_list, inplace=True)
        qc.measure_all()
        qc_total.append(qc)

    return qc_total


# ------------------------------------------------#
# ------------------------------------------------#


# ------  experiment example-------#
def build_qsd_example_circuit(
    num_qubit, inner_product, probability, method, rand_init_state
):
    qc = build_qsd_experiment_circuit(
        num_qubit=num_qubit,
        inner_product=inner_product,
        qsd_method=method,
        rand_init_state=rand_init_state,
        probability=probability,
    )

    # transpile circuit
    opt_qc = transpile(qc, optimization_level=3, basis_gates=["u", "cx"])
    return opt_qc, qc


if __name__ == "__main__":
    # parameters
    num_qubit = 4
    qsd_method = "UQSD"  # "UQSD" or "MED"
    random_init_state = "random"  # "Random" or "Fixed" incoming state.

    # construct circuit
    inner_product = 0.5  # 0 to 1
    probability = 0.3  # 0 to 1
    opt_qc = build_qsd_experiment_circuit(
        num_qubit, inner_product, probability, qsd_method, random_init_state
    )

    # ------  run circuit -------#
    """
    It should be replaced by IBMQ or IonQ provider.
    """
    sampler = Sampler()
    job = sampler.run(opt_qc, shots=5000)
    result = job.result()

    # ------ collect data -------#
    if random_init_state:
        outcome = vec_to_array(
            result.quasi_dists[0].binary_probabilities(), num_qubit + 1
        )
    else:
        outcome = np.array(
            [
                np.array([probability, 1 - probability])[i]
                * vec_to_array(result.quasi_dists[i].binary_probabilities(), num_qubit)
                for i in range(2)
            ]
        ).flatten()

    # ------- print result ------#
    # gate count
    print("gate count:", opt_qc[0].count_ops())

    # theoretical result
    if qsd_method in ["MED", "med"]:
        success_rate = (
            1 + np.sqrt(1 - 4 * probability * (1 - probability) * (inner_product) ** 2)
        ) / 2
        print(f"MED success rate = {success_rate}")
    else:
        print("UQSD success rate = ", 1 - inner_product / 2)

    # experimental result
    print("Experiment result : ", data_analysis(outcome, qsd_method))
