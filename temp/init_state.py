# Initialize state
import qclib.isometry
import qiskit
import qiskit.circuit
import qclib
import time


import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import qiskit.qasm2
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
    qc_iso: QuantumCircuit = qclib.isometry.decompose(
        iso_trans, scheme=scheme
    ).inverse()
    final_circuit: QuantumCircuit = qc_iso.decompose(reps=7)
    print(final_circuit.count_ops())
    print("num_qubits =", final_circuit.num_qubits)
    print("Depth =", final_circuit.depth())
    print(f"Decomposition ({num_qubits} qubits, {scheme}) success!")
    return


# TODO
# QSD for four states
def gen_states(
    num_qubits: int,
    num_states: int,
    **kwargs,
):
    assert num_qubits > 0
    assert num_states > 1
    states = []
    # TODO
    states = [random_statevector(2**num_qubits) for _ in range(num_states)]
    return states


def med_k_states(
    num_qubits,
    num_states,
):
    states = gen_states(num_qubits=num_qubits, num_states=num_states)
    # TODO Sum the POVMs and perform diagonalization

    #
    return


def uqsd_k_state(
    num_qubits,
    num_states,
    prior_prob=[],
    # num_povms, # Is it required?
):
    states: list[Statevector] = gen_states(num_qubits=num_qubits, num_states=num_states)

    # Formulate convex problem
    vec_size = 2**num_qubits

    print("Purity =", states[0].purity())
    assert states[0].purity()

    # TODO check the states are linear independent

    # for i in range(len(states)):
    #     print(states[i].data)

    def comp_square(vars, vec_size) -> cp.Expression:
        expr: cp.Expression = 0
        for i in range(vec_size):
            expr += vars[0, i] ** 2
        for i in range(vec_size):
            expr += vars[1, i] ** 2
        # print(expr)
        return expr

    def comp_inner_prod(vars, state_data) -> tuple[cp.Expression, cp.Expression]:
        print(state_data)
        vec_size = len(state_data)
        expr_re: cp.Expression = 0
        expr_im: cp.Expression = 0
        for i in range(vec_size):
            expr_re += (
                state_data[i].real * vars[0, i] - (-state_data[i].imag) * vars[1, i]
            )
            expr_im += (
                state_data[i].real * vars[1, i] + (-state_data[i].imag) * vars[0, i]
            )
        # print(expr_re)
        # print(expr_im)
        return expr_re, expr_im

    def comp_expand_opsum(var_list) -> cp.Expression:
        expr = 0
        for i in range(num_states):
            expr_re, expr_im = comp_inner_prod(var_list[i], states[i].data)
            expr += expr_re**2 + expr_im**2
        return expr

    # TODO
    var_list = []
    constraints = []
    for i in range(num_states):
        # POVM that is orthogonal to the rest of the states
        # The "2" in the shape is to express the real and imaginary part of the complex numbers
        ket_var = cp.Variable(shape=[2, vec_size], name=f"ket{i}")
        statevec_constraint = comp_square(ket_var, vec_size) == 1
        constraints.append(statevec_constraint)
        var_list.append(ket_var)
    for i in range(num_states):
        for j in range(num_states):
            # Exclude the corresponding state
            if i != j:
                expr_re, expr_im = comp_inner_prod(var_list[i], states[j].data)
                constraints.append(expr_re == 0)
                constraints.append(expr_im == 0)

    # Maximize the success probability
    target = comp_expand_opsum(var_list)
    # objective = cp.Maximize(expr=target)
    objective = cp.Minimize(expr=target)
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    print(var_list[0].value)
    print(var_list[1].value)
    print(var_list[2].value)
    return
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    print(constraints[0].dual_value)
    # Constraint the vectors' orthogonality relations
    # list of symbolic vectors
    # for vec in vec_list:
    #     print(vec)

    # Break the vector into re, im
    # 2 * 2 ** n real variables for each measurement operator
    # a2+b2+...+ = 1
    # inner product > 0 for the one (no need for this constraint)
    # inner product = 0 for the rest

    # Show the states are linear independent and structureless

    # TODO Analyze solution

    # Verify results
    # M = np.array()
    # np.matrix.trace(M)
    # np.trace(M)

    return


def uqsd_k_state_new(
    num_qubits,
    num_states,
    prior_prob=[],
):
    states: list[Statevector] = gen_states(num_qubits=num_qubits, num_states=num_states)

    # Formulate convex problem
    vec_size = 2**num_qubits

    print("Purity =", states[0].purity())
    assert states[0].purity()

    # TODO check the states are linear independent

    # for i in range(len(states)):
    #     print(states[i].data)

    def comp_inner_prod(vars, state_data) -> tuple[cp.Expression, cp.Expression]:
        print(state_data)
        vec_size = len(state_data)
        expr_re: cp.Expression = 0
        expr_im: cp.Expression = 0
        for i in range(vec_size):
            expr_re += (
                state_data[i].real * vars[0, i] - (-state_data[i].imag) * vars[1, i]
            )
            expr_im += (
                state_data[i].real * vars[1, i] + (-state_data[i].imag) * vars[0, i]
            )
        # print(expr_re)
        # print(expr_im)
        return expr_re, expr_im

    def comp_expand_opsum(var_list) -> cp.Expression:
        expr = 0
        for i in range(num_states):
            expr_re, expr_im = comp_inner_prod(var_list[i], states[i].data)
            expr += expr_re**2 + expr_im**2
        return expr

    var_list = []
    constraints = []
    for i in range(num_states + 1):
        # POVM that is orthogonal to the rest of the states
        # ket_var = cp.Variable(vec_size, complex=True, name=f"ket{i}")
        ket_var = cp.Variable(vec_size, complex=True, nonneg=False, name=f"ket{i}")
        var_list.append(ket_var)
        # constraints.append(cp.cumsum(cp.abs(ket_var)) == 1)
        # expr = cp.sum(ket_var ** 2) <= 1
        # expr = cp.sum(ket_var) <= 1
        # expr = cp.sum(cp.abs(ket_var)) == 1
        # constraints.append(expr)
        # assert expr.is_dcp()

    # TODO Sum = Id
    expr = 0
    for i in range(num_states + 1):
        expr += cp.inv_pos(cp.conj(cp.outer(var_list[i], var_list[i]))) * cp.outer(var_list[i], var_list[i])
    
    expr = (expr - np.identity(vec_size).data) == 0
    print(expr)
    assert expr.is_dcp()
    constraints.append(expr)


    for i in range(num_states):
        for j in range(num_states):
            # Exclude the corresponding state
            if i != j:
                expr = cp.multiply(var_list[i], states[j].data)
                constraints.append(expr == 0)

    # Maximize the success probability
    target = 0
    for i in range(num_states):
        # target += cp.abs(cp.sum(cp.multiply(states[i].data, var_list[i])))
        target += cp.abs(cp.sum(cp.multiply(states[i].data, var_list[num_states])))
    print(target)

    # objective = cp.Maximize(expr=target)
    objective = cp.Minimize(expr=target)
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(solver="OSQP", verbose=True, qcp=True)
    print(result)

    # The optimal value for x is stored in `x.value`.
    print(var_list[0].value)
    print(var_list[1].value)
    print(var_list[2].value)

    return


def cvxpy_example():
    # Problem data.
    m = 30
    n = 20
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Construct the problem.
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    print(x.value)
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    print(constraints[0].dual_value)


def test_fix_v1(num_qubits, inner_product=0.5):
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
    iso = qiskit.circuit.library.Isometry(
        iso_trans,
        num_ancillas_zero=num_qubits,
        num_ancillas_dirty=0,
    )
    # iso.add_decomposition()
    # print(iso.definition)
    circuit: QuantumCircuit = iso.definition
    # 4 < reps <= 7
    final_circuit: QuantumCircuit = circuit.decompose(reps=7)
    print(final_circuit.count_ops())
    print("num_qubits =", final_circuit.num_qubits)
    print("Depth =", final_circuit.depth())
    # OrderedDict({'u3': 1, 'u': 1})
    # num_qubits = 2
    # Depth = 2
    # OrderedDict({'u': 7, 'cx': 3, 'u3': 1})
    # num_qubits = 4
    # Depth = 8
    # OrderedDict({'u': 17, 'cx': 10, 'u3': 1})
    # num_qubits = 6
    # Depth = 22
    # OrderedDict({'u': 35, 'cx': 25, 'u3': 1})
    # num_qubits = 8
    # Depth = 51
    # OrderedDict({'u': 69, 'cx': 56, 'u3': 1})
    # num_qubits = 10
    # Depth = 112
    # OrderedDict({'u': 135, 'cx': 119, 'u3': 1})
    # num_qubits = 12
    # Depth = 235
    # OrderedDict({'u': 265, 'cx': 246, 'u3': 1})
    # num_qubits = 14
    # Depth = 484
    # OrderedDict({'u': 523, 'cx': 501, 'u3': 1})
    # num_qubits = 16
    # Depth = 987
    # OrderedDict({'u': 1037, 'cx': 1012, 'u3': 1})
    # num_qubits = 18
    # Depth = 1996
    # OrderedDict({'u': 2063, 'cx': 2035, 'u3': 1})
    # num_qubits = 20
    # Depth = 4027
    # OrderedDict({'u3': 1, 'u': 1})
    # OrderedDict({'u': 7, 'cx': 3, 'u3': 1})
    # OrderedDict({'u': 17, 'cx': 10, 'u3': 1})
    # OrderedDict({'u': 35, 'cx': 25, 'u3': 1})
    # OrderedDict({'u': 69, 'cx': 56, 'u3': 1})
    # OrderedDict({'u': 135, 'cx': 119, 'u3': 1})
    # OrderedDict({'u': 265, 'cx': 246, 'u3': 1})
    # OrderedDict({'u': 523, 'cx': 501, 'u3': 1})
    # OrderedDict({'u': 1037, 'cx': 1012, 'u3': 1})
    # OrderedDict({'u': 2063, 'cx': 2035, 'u3': 1})
    # OrderedDict({'u': 4113, 'cx': 4082, 'u3': 1})
    # OrderedDict({'u': 8211, 'cx': 8177, 'u3': 1})
    # OrderedDict({'u': 16405, 'cx': 16368, 'u3': 1})
    # OrderedDict({'u': 32791, 'cx': 32751, 'u3': 1})
    # qiskit.qasm2.dump(circuit, f"test_fix_v1_{num_qubits}.qasm")
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
    # plot_time(test_range=range(1, 11), scheme="ccd")
    # plot_time(test_range=range(1, 10), scheme="csd")
    # plot_time(test_range=range(2, 11), scheme="knill")
    # for i in range(1, 11):
    #     test_fix_v1(i)
    # for i in range(11, 15):
    #     test_fix_v1(i)
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
    # uqsd_k_state(3, 3)
    uqsd_k_state_new(3, 3)
