import numpy as np
from itertools import combinations
from functools import partial
from scipy.optimize import NonlinearConstraint
from scipy.linalg import null_space
from cobyqa import minimize
from qiskit.quantum_info import random_statevector


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


def test_idea():
    np.set_printoptions(precision=4)
    num_states = 3
    n = 2
    states = gen_states(n, num_states=num_states)
    # print(states)
    prods = []
    for s1, s2 in combinations(states, 2):
        prods.append(abs(s1.inner(s2)))
    # print(prods)

    # TODO
    state_coeffs = []
    for i in range(num_states):
        # states[i].data
        print()

    # Obtain every null space for each set of orthonormal basis
    null_spaces = []

    # coeffs is a complicated list of coefficients
    coeffs = []

    # TODO Change for general case
    find_null_spaces(n, states, null_spaces, coeffs)

    # pprint.pprint(coeffs)
    print(len(coeffs))

    for i in range(12):
        print(np.array(coeffs[i]))
    # print(np.array(coeffs[1]))

    # TODO Check the formula
    # num_coeffs = C(num_states, 1) * (2 ** n - s + 1) * 2 * 2**n
    # num_vars   = C(num_states, 1) * (2 ** n - s + 1) * 2

    num_amps = 2**n
    num_ops = num_states
    num_basis = 2**n - (num_states - 1)
    num_vars_per_op = num_basis * 2
    num_vars = num_ops * num_basis * 2
    num_coeffs = num_vars * num_amps
    constraints = []

    # https://stackoverflow.com/questions/55132107/scipy-fitting-with-parameters-in-a-vector
    from operator import add

    def whole_vec_real(num_basis, op_idx, x):
        # Linear combination of the basis vectors
        # to form a different unit vector in the null space
        sum_real = [0 for _ in range(num_amps)]
        for i in range(num_basis):
            idx = op_idx * num_vars_per_op + 2 * i
            v_real = x[idx]
            v_imag = x[idx + 1]
            c_real = coeffs[idx]
            c_imag = coeffs[idx + 1]
            # print(i)
            sum_real = map(
                add,
                sum_real,
                [c_real[_] * v_real - c_imag[_] * v_imag for _ in range(num_amps)],
            )
        return list(sum_real)

    def whole_vec_imag(num_basis, op_idx, x):
        # Linear combination of the basis vectors
        # to form a different unit vector in the null space
        sum_imag = [0 for _ in range(num_amps)]
        for i in range(num_basis):
            idx = op_idx * num_vars_per_op + 2 * i
            v_real = x[idx]
            v_imag = x[idx + 1]
            c_real = coeffs[idx]
            c_imag = coeffs[idx + 1]
            # print(i)
            sum_imag = map(
                add,
                sum_imag,
                [c_real[_] * v_imag + c_imag[_] * v_real for _ in range(num_amps)],
            )
        return list(sum_imag)

    def mult_vec_and_state_real(states, num_basis, op_idx, x):
        # TODO Access the coeffs of the state data for obj
        # Sum everything after mult
        # TODO Check the signs
        s_real = np.array([i.real for i in states[op_idx].data])
        s_imag = np.array([i.imag for i in states[op_idx].data])
        vec_real = np.array(whole_vec_real(num_basis, num_ops - 1 - op_idx, x))
        vec_imag = np.array(whole_vec_imag(num_basis, num_ops - 1 - op_idx, x))
        final_real = np.inner(s_real, vec_real) - np.inner(s_imag, vec_imag)
        final_imag = np.inner(s_real, vec_imag) + np.inner(s_imag, vec_real)
        return final_real**2 + final_imag**2

    def obj(num_basis, x):
        # The final objective function
        s = 0
        for i in range(num_ops):
            s += mult_vec_and_state_real(states, num_basis, i, x)
        return s

    def mult_vec_and_vec_real(num_basis, op_idx0, op_idx1, x):
        vec0_real = np.array(whole_vec_real(num_basis, op_idx0, x))
        vec0_imag = np.array(whole_vec_imag(num_basis, op_idx0, x))
        vec1_real = np.array(whole_vec_real(num_basis, op_idx1, x))
        vec1_imag = np.array(whole_vec_imag(num_basis, op_idx1, x))
        return np.inner(vec0_real, vec1_real) + np.inner(vec0_imag, vec1_imag)

    def mult_vec_and_vec_imag(num_basis, op_idx0, op_idx1, x):
        vec0_real = np.array(whole_vec_real(num_basis, op_idx0, x))
        vec0_imag = np.array(whole_vec_imag(num_basis, op_idx0, x))
        vec1_real = np.array(whole_vec_real(num_basis, op_idx1, x))
        vec1_imag = np.array(whole_vec_imag(num_basis, op_idx1, x))
        return np.inner(vec0_real, vec1_imag) - np.inner(vec0_imag, vec1_real)

    for i, j in combinations(list(range(num_ops)), 2):
        constraints.append(
            NonlinearConstraint(partial(mult_vec_and_vec_real, num_basis, i, j), 0, 0)
        )
        constraints.append(
            NonlinearConstraint(partial(mult_vec_and_vec_imag, num_basis, i, j), 0, 0)
        )

    # Solve for a better null vector such that
    # a = np.dot(states[2].data, n_s[:, 0] * x0 + n_s[:, 1] * x1) is max

    # TODO Preserve LC of unit = unit vector
    def op_con(op_idx, x):
        s = 0
        for j in range(num_basis):
            idx = op_idx * num_vars_per_op + 2 * j
            s += x[idx] ** 2 + x[idx + 1] ** 2
        return s

    for i in range(num_ops):
        constraints.append(NonlinearConstraint(partial(op_con, i), 1, 1))

    for i in range(num_vars // 2):
        con = lambda x: x[2 * i] ** 2 + x[2 * i + 1] ** 2
        constraints.append(NonlinearConstraint(con, 0, 1))

    # TODO define callback
    def callbackF(Xi):
        global Nfeval
        # print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], rosen(Xi))
        Nfeval += 1

    # x0 = np.zeros(num_vars)
    x0 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    result = minimize(
        fun=partial(obj, num_basis),
        x0=x0,
        constraints=constraints,
        options={
            "maxfev": 30000,
            "disp": True,
        },
    )
    # result = minimize(
    #     fun=partial(obj, num_basis),
    #     x0=x0,
    #     constraints=constraints,
    #     method="COBYQA",
    # )
    print(result)
    return


def find_null_spaces(n, states, null_spaces, coeffs):
    for s1, s2 in combinations(states, 2):
        # TODO exclude the target state
        A = [s1.data, s2.data]
        null_s = null_space(np.array(A))
        null_spaces.append(null_s)

        # Verify our calculation
        # print(np.round(np.dot(s1.data, null_s[:, 0]), 4))
        # print(np.round(np.dot(s1.data, null_s[:, 1]), 4))
        # print(np.round(np.dot(s2.data, null_s[:, 0]), 4))
        # print(np.round(np.dot(s2.data, null_s[:, 1]), 4))

        # TODO
        # Is the number of basis vectors = num_states - 1?
        # Should be 2 ** n - 2
        for i in range(2**n - 2):
            basis = null_s[:, i]
            basis_real = [np.real(num) for num in basis]
            basis_imag = [np.imag(num) for num in basis]
            coeffs.append(basis_real)
            coeffs.append(basis_imag)
    # Recursively define the constraints
    # a = np.dot(states[2].data, null_s[:, 0])
    # b = np.dot(states[2].data, null_s[:, 1])
    # print(np.round(a, 4))
    # print(np.round(b, 4))
    # print(np.round(np.abs(a), 4))
    # print(np.round(np.abs(b), 4))

    # Recursively define the objective

    def find_PVM():
        return

    return


if __name__ == "__main__":
    test_idea()
