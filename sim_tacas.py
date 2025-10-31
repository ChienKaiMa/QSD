# # Quantum Circuit Simulation for Quantum State Discrimination


# UQSD for now


from utils import *
from qiskit.circuit import QuantumCircuit
import sys
import time
import tracemalloc
import qiskit.transpiler
import qiskit.synthesis
import qiskit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector, set_statevector, SetDensityMatrix
from qiskit.quantum_info import Statevector

# Load states
from flow.problem_spec import *
from utils.handy_states import *

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
)


import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# ## Load the target circuit


# nq = 2
nq = 6
ns = 3
# seed = 1
# case_id = f"q{nq}_n{ns}_s{seed}"
case_id = f"q{nq}_n{ns}"
# qasm_name = f"circuits/coherent/coh_{case_id}_no_backend_resynth.qasm"
# TODO add resynth back
qasm_name = f"coh_symm_q4_n3_optuqsd_reducedpovm_ccd_no_backend.qasm"

qc = QuantumCircuit.from_qasm_file(qasm_name)
print(qc.num_qubits)
print(qc.count_ops())
print(qc.depth())

qc = transpile(qc, basis_gates=["rx", "ry", "rz", "cx"])
print(int(qc.depth() * 0.05))
qc_approx = transpile(
    circuits=qc,
    unitary_synthesis_method="aqc",
    unitary_synthesis_plugin_config={
        "network_layout": "cart",
        "connectivity_type": "star",
        "depth": int(qc.depth() * 0.05),
    },
)
print("Approx")
print(qc_approx.count_ops())
print(qc_approx.depth())

# Define handy simulation backends


AerSimulator().available_methods()
sv_backend = AerSimulator(method="statevector", seed_simulator=42)


state_vec = coh_symm_small(num_qubits=6)["state_vec"]
states = [Statevector(_) for _ in state_vec]


# # Ideal sim verification


# Our original circuit does not contain final measurements
#
# It is copied because measure_active creates new ClassicalRegisters every time
#
# https://docs.quantum.ibm.com/guides/measure-qubits


# The state discrimination circuit without the initial state


qc_tmp = qc_approx.copy()
qc_tmp.save_statevector()
qc_tmp.measure_active()

# Assume the job always got done quickly
result = sv_backend.run(qc_tmp).result()
result.get_counts()
result.get_statevector()
result.get_statevector().probabilities().round(15)


# ## Construct DUT
# The circuit with one of the target initial states


for i in range(3):
    # dut = QuantumCircuit(qc.num_qubits)
    dut = QuantumCircuit(nq)
    inst = dut.set_statevector(states[i]).instructions
    # inst = dut.set_statevector(states[i].expand(Statevector([1, 0]))).instructions
    # inst = dut.set_statevector(states[i].expand(Statevector([1, 0])).expand(Statevector([1, 0]))).instructions
    dut.append(inst[0], [_ for _ in range(nq)])
    # dut.save_statevector()
    dut.append(qc, [_ for _ in range(qc.num_qubits)])
    # dut.id(0)
    dut.save_statevector()
    dut.measure_active()
    dut = dut.decompose(reps=3)
    result = sv_backend.run(dut).result()
    # Show results
    print(result.get_counts())
    print(result.get_statevector().probabilities().round(5))
    # print(result.get_statevector().probabilities().round(10))


# # Noisy sim extrapolation


def test_dut(
    dut, param, target_states: list[str] | None = None, noise_model=None
):
    """Test DUT under the two-qubit depolarizing error"""
    noise_model = _two_qubit_depolarizing_noise_model(param)

    noise_result = sv_backend.run(dut, noise_model=noise_model).result()
    try:
        print(param)
        print("'001' count =", noise_result.get_counts()["001"])
    except:
        pass
    if target_states is None:
        return noise_result.get_statevector().probabilities()
    else:
        return [
            noise_result.get_statevector().probabilities()[state].item()
            for state in target_states
        ]


def _two_qubit_depolarizing_noise_model(param):
    noise_model = NoiseModel()
    two_qubit_error = depolarizing_error(param, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
    return noise_model


# Check different params
# Add references for better parameters?
params = [
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
]


colors = list(TABLEAU_COLORS)

result_0 = []
result_1 = []
result_2 = []
for p in params:
    a, b, c = test_dut(dut, p, [0, 1, 2])
    # print(a, b)
    print(a)
    print(b)
    print(c)
    result_0.append(a)
    result_1.append(b)
    result_2.append(c)

# Write the results to a CSV file
with open("results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["result_0", "result_1", "result_2"])  # Header row
    for i in range(len(result_0)):
        writer.writerow([result_0[i], result_1[i], result_2[i]])


# TODO Save data to .npy or .npz

fig = plt.figure(dpi=600)
# fig = plt.figure(dpi=200)
fig.set_figwidth(6)
fig.set_figheight(4.8)
# fig.set_figwidth(3)
# fig.set_figheight(2.4)

# plt.axvline(0.001, linestyle="--", alpha=0.3)
# plt.arrow(0.0018, 0.25, -0.0005, 0, head_width=0.01, head_length=0.0003)
# plt.text(0.0022, 0.24, "Sampling noise\n" + "starts to appear...")
plt.axhline(test_dut(dut, 0, [0]), linestyle="--", alpha=0.3, color=colors[0])
plt.axhline(0, linestyle="--", alpha=0.3, color=colors[1])

plt.xscale("log")
plt.plot(params, result_0, ".", label="0")  # '"00" → "000"')
plt.plot(params, result_1, "+", label="1")  # '"01" → "001"')
plt.plot(params, result_2, "*", label="2")  # '"11" → "002"')

# plt.xlabel(r"$\theta / \pi$" + f" ({case_id})")
# plt.xlabel("Depolarizing noise parameter")
# plt.ylabel("Probability amplitude")
# plt.title(f"UQSD with noise in two-qubit gates ({case_id})")
plt.legend(loc="center left")
plt.grid(True, alpha=0.1)

plt.savefig(
    fname=f"results/OptUQSD_ccd_{case_id}.png",
    bbox_inches="tight",
)

plt.close()
