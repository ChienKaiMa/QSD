# Initialize state
import qclib.isometry
import qiskit
import qiskit.circuit
import qclib
import time


import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import dccp
from itertools import combinations
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
    states = [random_statevector(2**num_qubits, seed=_) for _ in range(num_states)]
    return states


def med_k_states(
    num_qubits,
    num_states,
):
    states = gen_states(num_qubits=num_qubits, num_states=num_states)
    # TODO Sum the POVMs and perform diagonalization

    #
    return


# multi-state uqsd: Find PVM
def uqsd_k_state(
    num_qubits,
    num_states,
    prior_prob=[],
    # num_povms, # Is it required?
):
    states: list[Statevector] = gen_states(
        num_qubits=num_qubits,
        num_states=num_states,
    )
    # for i in range(len(states)):
    #     print(states[i].data)

    print("Purity =", states[0].purity())
    assert states[0].purity()

    # TODO check the states are linear independent with Gram-Schmidt (QR decomposition)

    # Formulate convex problem
    # Assume variables are ??
    # Input: states

    def comp_square(vars, vec_size) -> cp.Expression:
        expr: cp.Expression = 0
        for i in range(vec_size):
            expr += vars[0, i] ** 2
        for i in range(vec_size):
            expr += vars[1, i] ** 2
        # print(expr)
        return expr

    def comp_inner_prod(vars, state_data) -> tuple[cp.Expression, cp.Expression]:
        # print(state_data)
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
    vec_size = 2**num_qubits
    var_list = []
    constraints = []
    for i in range(num_states):
        # POVM that is orthogonal to the rest of the states
        # The "2" in the shape is to express the real and imaginary part of the complex numbers
        ket_var = cp.Variable(shape=[2, vec_size], pos=True, name=f"ket{i}")
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
    result = prob.solve(solver="ECOS", qcp=False, gp=True)
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
        # PVM that is orthogonal to the rest of the states
        # ket_var = cp.Variable(vec_size, complex=True, name=f"ket{i}")
        # Variable for measurement
        M_var = cp.Variable(shape=(vec_size, vec_size), complex=True, name=f"M{i}")
        # M_var = cp.Variable(shape=(vec_size, vec_size), Hermitian=True, name=f"M{i}")
        var_list.append(M_var)
    # constraints.append(cp.cumsum(cp.abs(ket_var)) == 1)
    # expr = cp.sum(ket_var ** 2) <= 1
    # expr = cp.sum(ket_var) <= 1
    # expr = cp.sum(cp.abs(ket_var)) == 1
    # constraints.append(expr)
    # assert expr.is_dcp()
    #
    for i in range(num_states):
        for j in range(num_states):
            if i != j:
                constraints.append(cp.matmul(var_list[i], states[j]) == 0)
    # M1 ortho M2 ...
    # for M1, M2 in combinations(var_list, 2):
        # c = cp.matmul(M1, M2) == 0
        # print(c.is_dcp())
        # print(c.is_dgp())
        # print(c.is_dqcp())
        # print(c.is_dpp())
        # constraints.append(c)
    # # M1 * M1 = M1
    for M in var_list:
        c = cp.matmul(M, M) == M
        # print(c.is_dcp())
        # print(c.is_dgp())
        # print(c.is_dqcp())
        # print(c.is_dpp())
        constraints.append(c)
    # TODO Sum = Id
    # TODO Minimize failure or maximize success
    expr = cp.sum(var_list)

    expr = (expr - np.identity(vec_size).data) == 0
    # print(expr)
    assert expr.is_dcp()
    constraints.append(expr)

    # Maximize the success probability
    target = 0
    for i in range(num_states):
        # target += cp.abs(cp.sum(cp.multiply(states[i].data, var_list[i])))
        target += cp.abs(
            cp.sum(
                cp.matmul(np.conj(states[i].data.T), cp.matmul(var_list[num_states], states[i].data))
            )
        )
    # print(target)

    # objective = cp.Maximize(expr=target)
    objective = cp.Minimize(expr=target)
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    # result = prob.solve(verbose=False)
    print(prob.is_dcp())
    print(dccp.is_dccp(prob))
    result = prob.solve(verbose=False, method="dccp")
    # result = prob.solve(verbose=True)
    print(result)

    # The optimal value for x is stored in `x.value`.
    from pprint import pprint

    for M in var_list:
        a = M.value
        b = np.matmul(M.value, M.value)
        c = b[0][0] / a[0][0]
        a = np.multiply(1 / c, a)
        # pprint(M.value)
        # pprint(np.matmul(M.value, M.value))
        # pprint(a)
        # pprint(np.matmul(a, a))
        pprint(np.isclose(a, np.matmul(a, a)))
        print()
        
    # pprint(var_list[0].value)
    # print()
    # pprint(var_list[1].value)
    # print()
    # pprint(var_list[2].value)
    # print()

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


def plot_time(test_range=range(1, 11), scheme="ccd"):
    time_data = []
    for i in test_range:
        start_time = time.time()
        qclib_impl(num_qubits=i, scheme=scheme)
        end_time = time.time()
        execution_time = end_time - start_time
        time_data.append([i, execution_time])
    np.savetxt(f"{scheme}_decomp_time.csv", time_data, fmt="%.16f", delimiter=",")



def baby_example(phi, a):
    """Extend the example of Eldar's paper in 2003."""
    np.set_printoptions(precision=3)
    phi1 = [1, 0]
    phi2 = [np.cos(phi), np.sin(phi)]
    psi = np.transpose([phi1, phi2])
    print(psi)
    recip_psi = np.matmul(psi, np.linalg.inv(np.matmul(np.matrix.getH(psi), psi)))
    # Round nearzero value
    recip_psi = recip_psi.round(15)
    print(recip_psi)
    recip_psi = recip_psi.T
    q1 = np.outer(recip_psi[0], recip_psi[0])
    # print(q1)

    dim = 3
    prior_prob = np.multiply(-1 / 2, [1, 1])
    # Measurement operators
    q = []
    # for i in range(2):
    # q.append(np.outer(recip_psi[i], recip_psi[i]).round(1))
    # a = 0.5
    q.append(np.outer([1, -1, a], [1, -1, a]).round(1))
    q.append(np.outer([0, 1.414, 1.414 / a], [0, 1.414, 1.414 / a]).round(1))
    q = np.array(q)
    print(q)

    # Handcraft the matrix because I suck
    # v = cp.Variable(2, complex=True)
    # v = cp.Variable(2)
    p = cp.Variable(2)

    I = np.identity(dim)

    objective = cp.Minimize(1 + cp.sum(prior_prob @ p))
    constraints = [
        0 <= p[0],
        0 <= p[1],
        I - p[0] * q[0] - p[1] * q[1] >> 0,  # Matrix inequality uses >>
        # I - q[0] - q[1] >> 0,  # Matrix inequality uses >>
        # I - q[0] >> 0,  # Matrix inequality uses >>
        # v[0] * v[1] - 1.414 == 0,
    ]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(
        f"Guess = {(1 - 1 / (2 * np.linalg.norm([1, -1, a]) ** 2) - 1 / (2 * np.linalg.norm([0, 1.414, 1.414 / a]) ** 2)):.3f}"
    )
    print("Result =", result.round(3))
    print("Theory =", 1 - np.cos(phi) / 2)
    # An acceptable optimal solution
    sol = p.value.round(4)
    print("Solution =", sol)
    pi1 = I - sol[0] * q[0] - sol[1] * q[1]  # Positive semidefinite
    for i in range(2):
        print(sol[i] * q[i])
    print(pi1.round(5))

    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    # print(constraints[0].dual_value)
    return


def extend_2003_Eldar_results():
    """Extend the example of Eldar's paper in 2003."""
    np.set_printoptions(precision=3)
    phi1 = np.multiply(1 / np.sqrt(3), [1, 1, 1, 0])
    phi2 = np.multiply(1 / np.sqrt(2), [1, 1, 0, 0])
    phi3 = np.multiply(1 / np.sqrt(2), [0, 1, 1, 0])
    psi = np.transpose([phi1, phi2, phi3])
    print(psi)
    recip_psi = np.matmul(psi, np.linalg.inv(np.matmul(np.matrix.getH(psi), psi)))
    # Round nearzero value
    recip_psi = recip_psi.round(15)
    print(recip_psi)
    recip_psi = recip_psi.T
    q1 = np.outer(recip_psi[0], recip_psi[0])
    # print(q1)

    # Some CVX you like
    var = cp.Variable(complex=True)
    return


if __name__ == "__main__":
    # uqsd_k_state(3, 3)
    uqsd_k_state_new(3, 3)
    # for i in range(1, 20):
    #     baby_example(np.pi / 4, 0.1 * i)
    # test_idea()
    # extend_2003_Eldar_results()
