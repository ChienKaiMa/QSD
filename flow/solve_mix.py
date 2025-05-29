import logging.config
import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../flow")
from flow.interface import *
from flow.problem_spec import *
from flow.plots import *
from flow.verify_povm import *
import numpy as np
import cvxpy as cp
from scipy.linalg import null_space
from temp.get_random_seeds import get_random_seeds
from utils.prob_matrix import *
import time
import tracemalloc
from collections import defaultdict


def apply_Eldar_mix(self, prior_prob=None, p_I=0):
    """Apply the method in Eldar's paper in 2004. SIM
    p_I: The predefined portion of inconclusive results. [0, 1)
    """
    np.set_printoptions(precision=4)
    n = self.num_states
    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)

    # Measurement operators
    ## q = []
    ## for i in range(n):
    ##     q.append(np.outer(recip_psi[i], recip_psi[i]).round(1))
    ## q = np.array(q)

    I = np.identity(self.num_amps)
    X = cp.Variable(shape=(self.num_amps, self.num_amps), hermitian=True)
    delta_scalar = cp.Variable(1, name="d")
    beta = p_I  # Follow the naming in the paper

    objective = cp.Minimize(cp.trace(X) - delta_scalar * beta)

    # TODO [Priority: Low] add assertions
    Delta = np.sum(
        [np.multiply(prior_prob[i], self.states[i].data) for i in range(n)]
    )

    # Matrix inequality uses >>
    constraints = []
    for i in range(n):
        constraints.append(
            X - np.multiply(prior_prob[i], self.states[i].data) >> 0
        )
    constraints.append(X - cp.multiply(delta_scalar, Delta) >> 0)
    constraints.append(X >> 0)
    # TODO Add different constraint configurations
    # epsilon-oriented

    prob = cp.Problem(objective, constraints)
    # Solver options: The precision has to be 1e-16 ~ 1e-20 to be enough for SCS...
    ## result = prob.solve(solver=cp.CLARABEL)\
    ## No options for precision, and it couldn't find null vector
    ## result = prob.solve(solver=cp.CVXOPT, feastol=1e-10)
    ## CVXOPT failed when feastol is small (Not sure root cause)
    ## result = prob.solve(solver=cp.SCS, eps=1e-15)
    ## "eps > 1e-16" couldn't find null vector for operator
    result = prob.solve(solver=cp.SCS, eps=1e-10)
    print("Result =", result)
    print(f"CVXPY returns {prob.status}")
    logger.info(f"CVXPY returns {prob.status}")

    # Please don't round X_sol
    X_sol = X.value
    delta_sol = delta_scalar.value
    print("Solution for X =")
    print(X_sol)
    print("Solution for delta =", delta_sol)

    # Find measurement operator
    povm = []
    for i in range(n):
        op = X_sol - np.multiply(prior_prob[i], self.states[i].data)
        # The precision here (rcond) also matters
        ## if the answer is not found, check the matrix and try a larger rcond
        ns = null_space(op, rcond=1e-7)
        # print(op)
        # TODO reshape ns
        # print(ns)
        # TODO
        povm.append(ns[:, 0])
        # res = np.all(np.linalg.eigvals(op) >= 0)
        # print(np.linalg.eigvals(op))
        # print(res)
    # The inconclusive measurement operator
    if beta == 0:
        print("Info: The inconclusive measurement is disabled.")
    # else:
    #     self.num_ops += 1
    #     op = X_sol - np.multiply(delta_sol, Delta)
    #     # The precision here (rcond) also matters
    #     ## if the answer is not found, check the matrix and try a larger rcond
    #     ns = null_space(op, rcond=1e-10)
    #     print(ns)
    #     povm.append(ns[:, 0])

    # 2024/10/30 Obtain PVM
    # The error measurement operator

    last_op = np.eye(self.num_amps, dtype="complex128")
    for m in povm:
        # Add "None" to transpose
        # https://stackoverflow.com/a/11885718/13518808
        op = np.multiply(m[None].T.conj(), m)
        last_op -= op
        # print(m)
    u, s, v = np.linalg.svd(last_op, hermitian=True)
    last_povm = u[:, 0] * np.sqrt(s[0])
    # print(last_povm)
    povm.append(last_povm.conj())

    # TODO
    # Verify solution
    # Positive semidefinite
    # Calculate all probabilities
    print("POVMs:")
    for m in povm:
        print(m)

    print("Probabilities for each state:")
    total = 0
    p_d = 0
    for i in range(n):
        probs = compute_event_probabilities(povm, self.states[i].data)
        print(np.array(probs))
        total += prior_prob[i] * sum(probs)
        p_d += prior_prob[i] * probs[i]
    print("Total probability =", total)
    print("Success probability =", p_d)
    print()

    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    # print(constraints[0].dual_value)
    return povm

    isometry = self.naimark(povm)
    print(isometry)
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
    qc_iso = decompose(isometry, scheme="csd").inverse()
    # qc_iso = decompose(isometry, scheme="ccd").inverse()

    service = qiskit_ibm_runtime.QiskitRuntimeService(
        channel="ibm_quantum",
        # instance="ibm-q/open/main",
        instance="ibm-q-hub-ntu/ntu-internal/default",
        token="e421d41292d0977e88ca2900d333e6b6789377af70e1923ba067e97afb929b2da3cd64bba701d1519067002f9c1fabe1e55c47a5539b12d8ec55b85864f6092d",
    )

    # Transpile first without the backend to avoid strange errors
    # qclib -> qiskit
    qc_iso = transpile(qc_iso)
    print(qc_iso.decompose(reps=2).count_ops())
    print("Depth,", qc_iso.decompose(reps=2).depth())

    # Transpile with the backend
    ibm_backend = service.backend("ibm_brisbane")
    qc_iso = transpile(qc_iso, backend=ibm_backend)
    print(qc_iso.count_ops())
    print("Depth,", qc_iso.depth())

    qiskit.qasm2.dump(
        qc_iso,
        f"qc_iso_q{self.num_qubits}_n{self.num_states}_noseed.qasm",
    )
    # t1_end = time.process_time()

    return


def apply_Eldar_mix_primal(
    problem_spec: ProblemSpec,
    prior_prob=None,
    beta=0,
    is_cvxpy_verbose=False,
):
    """Apply Eldar's formulation.
    The predefined portion of inconclusive results. [0, 1)
    """
    logger = logging.getLogger(__name__)
    np.set_printoptions(precision=4)
    n = problem_spec.num_states

    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)
    logger.info(f"The prior probabilities is set to uniform (n = {n})")

    logger.info(f"The inconclusive portion is set to {beta}")

    # TODO
    PI_list = []
    for i in range(n + 1):
        PI = cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        PI_list.append(PI)

    objective = cp.Maximize(
        cp.sum(
            [
                prior_prob[i]
                * cp.real(
                    cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
                )
                for i in range(n)
            ]
        )
    )

    # TODO constraints
    I = np.identity(problem_spec.num_amps)
    constraints = []
    for i in range(n + 1):
        constraints.append(PI_list[i] >> 0)

    constraints.append(
        cp.real(
            cp.sum(
                [
                    prior_prob[j]
                    * cp.trace(
                        cp.matmul(problem_spec.states[j].data, PI_list[n])
                    )
                    for j in range(n)
                ]
            )
        )
        >= beta
    )
    constraints.append(cp.sum(PI_list) == I)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(
        solver=cp.SCS,
        # eps=1e-10,
        mkl=True,
        verbose=is_cvxpy_verbose,
        acceleration_lookback=10,
    )
    print("Result =", result)
    print(f"CVXPY returns {prob.status}")
    logger.info(f"CVXPY returns {prob.status}")
    if prob.status != "optimal":
        logger.error(f"CVXPY returns {prob.status}")
        return

    povm = []
    for i in range(n + 1):
        print(f"Solution for PI_{i} =")
        print(PI_list[i].value)
        u, s, v = np.linalg.svd(PI_list[i].value, hermitian=True)
        last_povm = u[:, 0] * np.sqrt(s[0])
        povm.append(last_povm.conj())

    # TODO
    # Calculate the measurement operators
    ## last_op = np.eye(problem_spec.num_amps, dtype="complex128")
    ## for m in povm:
    ##     # Add "None" to transpose
    ##     # https://stackoverflow.com/a/11885718/13518808
    ##     op = np.multiply(m[None].T.conj(), m)
    ##     last_op -= op
    ##     # print(m)
    ## u, s, v = np.linalg.svd(last_op, hermitian=True)
    ## last_povm = u[:, 0] * np.sqrt(s[0])
    ## # print(last_povm)
    ## povm.append(last_povm.conj())

    # TODO
    # Verify solution
    print("Probabilities for each state:")
    total = 0
    p_d = 0
    inc_prob = 0
    for i in range(n):
        probs = compute_event_probabilities(povm, problem_spec.states[i].data)
        print(probs)
        total += sum(probs)

    probability_matrix = []
    for i in range(n):
        probs = compute_event_probabilities(
            prior_prob[i], povm, problem_spec.states[i].data
        )
        updated_probs = [0] * (len(prior_prob) + 1)
        for j in range(strings_used):
            target_state_index = bitstring_to_target_state[j]
            updated_probs[target_state_index] += probs[j]
        # TODO
        # Postprocessing
        probability_matrix.append(updated_probs)

    for i in range(n):
        p_d += probability_matrix[i][i]
        p_inc += probability_matrix[i][n]

    print(f"Total probability = {total:.4f}")
    print(f"Success probability = {p_d:.4f}")
    print(f"Inconclusive probability = {p_inc:.4f}")
    # TODO also return these values
    print()
    # save_prob_heatmap(
    #     prior_prob,
    #     povm,
    #     problem_spec.states,
    #     bitstring_to_target_state,
    #     strings_used,
    #     tag=rf"{n}_crossQD_$\alpha${alpha[0]:2f}_$\beta${beta[0]:2f}_l{noise_level:2f}",
    #     reuse_fig=reuse_fig,
    # )

    result_dict = defaultdict()
    result_dict["povm"] = povm
    result_dict["total_prob"] = total
    # result_dict["p_succ"] = p_succ
    # result_dict["p_err"] = p_err
    result_dict["p_inc"] = p_inc
    result_dict["PI_list"] = [PI_list[i].value for i in range(len(PI_list))]
    result_dict["bitstring_to_target_state"] = bitstring_to_target_state
    result_dict["strings_used"] = strings_used

    return result_dict


def apply_koova_mix_primal(
    problem_spec: ProblemSpec, prior_prob=None, gamma=None
):
    """Apply Koova's formulation.
    gamma: Threshold probabilities. List of [0, 1)
    """
    logger = logging.getLogger(__name__)
    np.set_printoptions(precision=4)
    n = problem_spec.num_states

    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)
    logger.info(f"The prior probabilities is set to uniform (n = {n})")

    if gamma is None:
        gamma = [0.2] * n
    logger.info(f"The threshold probabilities are set to {gamma}")

    # TODO
    PI_list = []
    for i in range(n + 1):
        PI = cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        PI_list.append(PI)

    objective = cp.Maximize(
        cp.sum(
            [
                prior_prob[i]
                * cp.real(cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i])))
                for i in range(n)
            ]
        )
    )

    # TODO constraints
    I = np.identity(problem_spec.num_amps)
    constraints = []
    for i in range(n + 1):
        constraints.append(PI_list[i] >> 0)
    for i in range(n):
        # TODO Correct index for error
        constraints.append(
            cp.real(
                cp.sum(
                    [
                        prior_prob[i]
                        * cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[j]))
                        for j in range(n)
                        if i != j
                    ]
                )
            )
            <= gamma[i]
        )
    constraints.append(cp.sum(PI_list) == I)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS, eps=1e-10)
    print("Result =", result)
    print(f"CVXPY returns {prob.status}")
    logger.info(f"CVXPY returns {prob.status}")
    if prob.status != "optimal":
        logger.error(f"CVXPY returns {prob.status}")
        return

    povm = []
    for i in range(n + 1):
        print(f"Solution for PI_{i} =")
        print(PI_list[i].value)
        u, s, v = np.linalg.svd(PI_list[i].value, hermitian=True)
        last_povm = u[:, 0] * np.sqrt(s[0])
        povm.append(last_povm.conj())

    # TODO
    # Calculate the measurement operators
    ## last_op = np.eye(problem_spec.num_amps, dtype="complex128")
    ## for m in povm:
    ##     # Add "None" to transpose
    ##     # https://stackoverflow.com/a/11885718/13518808
    ##     op = np.multiply(m[None].T.conj(), m)
    ##     last_op -= op
    ##     # print(m)
    ## u, s, v = np.linalg.svd(last_op, hermitian=True)
    ## last_povm = u[:, 0] * np.sqrt(s[0])
    ## # print(last_povm)
    ## povm.append(last_povm.conj())

    # TODO
    # Verify solution
    print("Probabilities for each state:")
    total = 0
    p_d = 0
    inc_prob = 0
    for i in range(n):
        probs = compute_event_probabilities(povm, problem_spec.states[i].data)
        print(probs)
        total += prior_prob[i] * sum(probs)
        p_d += prior_prob[i] * probs[i]
        inc_prob += prior_prob[i] * probs[n]
    print("Total probability =", total)
    print("Success probability =", p_d)
    print("Inconclusive probability =", inc_prob)
    print()
    save_prob_heatmap(
        prior_prob,
        povm,
        problem_spec.states,
        tag=f"q{problem_spec.num_qubits}_n{n}_koova_{gamma[0]:2f}",
    )

    # TODO
    # Return POVM
    return


def apply_dawei_mix_primal(problem_spec: ProblemSpec, prior_prob=None, gamma=None):
    """Apply Da-wei's formulation.
    gamma: Threshold probabilities. List of [0, 1)
    """
    logger = logging.getLogger(__name__)
    np.set_printoptions(precision=4)
    n = problem_spec.num_states

    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)
    logger.info(f"The prior probabilities is set to uniform (n = {n})")

    if gamma is None or len(gamma) != n:
        gamma = [0.2] * n
    logger.info(f"The threshold probabilities are set to {gamma}")

    # TODO
    PI_list = []
    for i in range(n + 1):
        PI = cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        PI_list.append(PI)

    objective = cp.Maximize(
        cp.sum(
            [
                prior_prob[i]
                * cp.real(cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i])))
                for i in range(n)
            ]
        )
    )

    # TODO constraints
    I = np.identity(problem_spec.num_amps)
    constraints = []
    for i in range(n + 1):
        constraints.append(PI_list[i] >> 0)
    for i in range(n):
        # TODO Correct index for error
        constraints.append(
            cp.real(
                prior_prob[i]
                * cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
            )
            >= cp.real(
                cp.sum(
                    [
                        prior_prob[i]
                        * cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[j]))
                        for j in range(n)
                    ]
                )
            )
            * (1 - gamma[i])
        )
    constraints.append(cp.sum(PI_list) == I)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS, eps=1e-10, acceleration_lookback=10)
    print("Result =", result)
    print(f"CVXPY returns {prob.status}")
    logger.info(f"CVXPY returns {prob.status}")
    if prob.status != "optimal":
        logger.error(f"CVXPY returns {prob.status}")
        return

    povm = []
    for i in range(n + 1):
        print(f"Solution for PI_{i} =")
        print(PI_list[i].value)
        u, s, v = np.linalg.svd(PI_list[i].value, hermitian=True)
        last_povm = u[:, 0] * np.sqrt(s[0])
        povm.append(last_povm.conj())

    # TODO
    # Calculate the measurement operators
    ## last_op = np.eye(problem_spec.num_amps, dtype="complex128")
    ## for m in povm:
    ##     # Add "None" to transpose
    ##     # https://stackoverflow.com/a/11885718/13518808
    ##     op = np.multiply(m[None].T.conj(), m)
    ##     last_op -= op
    ##     # print(m)
    ## u, s, v = np.linalg.svd(last_op, hermitian=True)
    ## last_povm = u[:, 0] * np.sqrt(s[0])
    ## # print(last_povm)
    ## povm.append(last_povm.conj())

    # TODO
    # Verify solution
    print("Probabilities for each state:")
    total = 0
    p_d = 0
    inc_prob = 0
    for i in range(n):
        probs = compute_event_probabilities(povm, problem_spec.states[i].data)
        print(probs)
        total += prior_prob[i] * sum(probs)
        p_d += prior_prob[i] * probs[i]
        inc_prob += prior_prob[i] * probs[n]
    print("Total probability =", total)
    print("Success probability =", p_d)
    print("Inconclusive probability =", inc_prob)
    print()
    save_prob_heatmap(
        prior_prob, povm, problem_spec.states, tag=f"{n}_dawei_{gamma[0]:2f}"
    )

    # TODO
    # Return POVM
    return


def apply_dawei_mix(problem_spec: ProblemSpec, prior_prob=None, gamma=None):
    """Apply Da-wei's formulation.
    gamma: Threshold probabilities. List of [0, 1)
    """
    logger = logging.getLogger(__name__)
    # TODO Everything! What is expected?
    np.set_printoptions(precision=4)
    n = problem_spec.num_states
    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)
    logger.info(f"The prior probabilities is set to uniform (n = {n})")

    if gamma is None:
        gamma = [0.2] * n
    logger.info(f"The threshold probabilities are set to {gamma}")

    I = np.identity(problem_spec.num_amps)
    X = cp.Variable(
        shape=(problem_spec.num_amps, problem_spec.num_amps), hermitian=True
    )
    delta_arr = cp.Variable(n, name="d")

    objective = cp.Minimize(cp.trace(X) - cp.sum(cp.multiply(delta_arr, gamma)))

    # TODO [Priority: Low] add assertions
    Delta = np.sum(
        [np.multiply(prior_prob[i], problem_spec.states[i].data) for i in range(n)]
    )

    # Matrix inequality uses >>
    constraints = []
    for i in range(n):
        constraints.append(
            X
            - np.multiply(prior_prob[i], problem_spec.states[i].data)
            - cp.sum(
                [
                    cp.multiply(
                        delta_arr[j] * prior_prob[j], problem_spec.states[j].data
                    )
                    for j in range(n)
                    if i != j
                ]
            )
            >> 0
        )

    constraints.append(X >> 0)
    # TODO Add different constraint configurations
    # epsilon-oriented

    prob = cp.Problem(objective, constraints)
    # Solver options: The precision has to be 1e-16 ~ 1e-20 to be enough for SCS...
    ## result = prob.solve(solver=cp.CLARABEL)\
    ## No options for precision, and it couldn't find null vector
    ## result = prob.solve(solver=cp.CVXOPT, feastol=1e-10)
    ## CVXOPT failed when feastol is small (Not sure root cause)
    ## result = prob.solve(solver=cp.SCS, eps=1e-15)
    ## "eps > 1e-16" couldn't find null vector for operator
    result = prob.solve(solver=cp.SCS, eps=1e-10)
    print("Result =", result)
    print(f"CVXPY returns {prob.status}")
    logger.info(f"CVXPY returns {prob.status}")
    if prob.status != "optimal":
        logger.error(f"CVXPY returns {prob.status}")
        return

    # Please don't round X_sol
    X_sol = X.value
    delta_sol = delta_arr.value
    print("Solution for X =")
    print(X_sol)
    print("Solution for delta =", delta_sol)

    # Find measurement operator
    povm = []
    for i in range(n):
        op = X_sol - np.multiply(prior_prob[i], problem_spec.states[i].data)
        # print(op)
        # The precision here (rcond) also matters
        ## if the answer is not found, check the matrix and try a larger rcond
        ns = null_space(op, rcond=1e-7)
        # print(op)
        # TODO reshape ns
        # print(ns)
        # TODO
        povm.append(ns[:, 0])
        # res = np.all(np.linalg.eigvals(op) >= 0)
        # print(np.linalg.eigvals(op))
        # print(res)
    # The inconclusive measurement operator
    ## if beta == 0:
    ##     print("Info: The inconclusive measurement is disabled.")
    # else:
    #     self.num_ops += 1
    #     op = X_sol - np.multiply(delta_sol, Delta)
    #     # The precision here (rcond) also matters
    #     ## if the answer is not found, check the matrix and try a larger rcond
    #     ns = null_space(op, rcond=1e-10)
    #     print(ns)
    #     povm.append(ns[:, 0])

    # 2024/10/30 Obtain PVM
    # The error measurement operator
    # if beta == 0:
    last_op = np.eye(problem_spec.num_amps, dtype="complex128")
    for m in povm:
        # Add "None" to transpose
        # https://stackoverflow.com/a/11885718/13518808
        op = np.multiply(m[None].T.conj(), m)
        last_op -= op
        # print(m)
    u, s, v = np.linalg.svd(last_op, hermitian=True)
    last_povm = u[:, 0] * np.sqrt(s[0])
    # print(last_povm)
    povm.append(last_povm.conj())

    # TODO
    # Verify solution
    # Positive semidefinite

    ## for i in range(n):
    ##     print(np.sqrt(sol[i]) * recip_psi[i])
    ##     print(np.linalg.norm(np.sqrt(sol[i]) * recip_psi[i]))
    # print(pi1.round(5))
    # Wrong answer if we over postprocess the solution
    # sol_overround = p.value.round(2)
    # print("Overprocessed solution =", sol_overround)
    # pi1_overround = (
    #     I
    #     - sol_overround[0] * q[0]
    #     - sol_overround[1] * q[1]
    #     - sol_overround[2] * q[2]
    # )  # Not positive semidefinite
    # print(pi1_overround.round(5))

    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    # print(constraints[0].dual_value)
    print("Probabilities for each state:")
    total = 0
    p_d = 0
    for i in range(n):
        probs = compute_event_probabilities(povm, problem_spec.states[i].data)
        print(probs)
        total += prior_prob[i] * sum(probs)
        p_d += prior_prob[i] * probs[i]
    print("Total probability =", total)
    print("Success probability =", p_d)
    print()
    return povm
    isometry = self.naimark(povm)
    print(isometry)
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
    qc_iso = decompose(isometry, scheme="csd").inverse()
    # qc_iso = decompose(isometry, scheme="ccd").inverse()

    service = qiskit_ibm_runtime.QiskitRuntimeService(
        channel="ibm_quantum",
        # instance="ibm-q/open/main",
        instance="ibm-q-hub-ntu/ntu-internal/default",
        token="e421d41292d0977e88ca2900d333e6b6789377af70e1923ba067e97afb929b2da3cd64bba701d1519067002f9c1fabe1e55c47a5539b12d8ec55b85864f6092d",
    )

    # Transpile first without the backend to avoid strange errors
    # qclib -> qiskit
    qc_iso = transpile(qc_iso)
    print(qc_iso.decompose(reps=2).count_ops())
    print("Depth,", qc_iso.decompose(reps=2).depth())

    # Transpile with the backend
    ibm_backend = service.backend("ibm_brisbane")
    qc_iso = transpile(qc_iso, backend=ibm_backend)
    print(qc_iso.count_ops())
    print("Depth,", qc_iso.depth())

    qiskit.qasm2.dump(
        qc_iso,
        f"qc_iso_q{self.num_qubits}_n{self.num_states}_noseed.qasm",
    )
    # t1_end = time.process_time()

    return


def apply_crossQD(
    problem_spec: ProblemSpec, prior_prob=None, alpha=None, beta=None, noise_level=0
):
    """Apply the cross quantum discrimination method.
    beta: Threshold probabilities. List of [0, 1)

    """
    assert problem_spec.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)
    np.set_printoptions(precision=4)
    n = problem_spec.num_states

    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)
        logger.info(f"The prior probabilities is set to uniform (n = {n})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    if alpha is None or len(alpha) != n:
        alpha = [0.01] * n
    logger.info(f"The threshold probabilities (alpha) are set to {alpha}")

    if beta is None or len(beta) != n:
        beta = [0.01] * n
    logger.info(f"The threshold probabilities (beta) are set to {beta}")

    # TODO
    PI_list = []
    for i in range(n + 1):
        PI = cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        PI_list.append(PI)

    objective = cp.Maximize(
        cp.sum(
            [
                prior_prob[i]
                * cp.real(cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i])))
                for i in range(n)
            ]
        )
    )

    # TODO constraints
    I = np.identity(problem_spec.num_amps)
    constraints = []
    for i in range(n + 1):
        constraints.append(PI_list[i] >> 0)

    # Conditional probability with respect to the measurement operator
    # prior_prob[i] is divided on both sides
    for i in range(n):
        constraints.append(
            cp.real(cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i])))
            >= cp.real(
                cp.sum(
                    [
                        cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[j]))
                        for j in range(n)
                    ]
                )
            )
            * (1 - alpha[i])
        )

    # Conditional probability with respect to the input state
    for i in range(n):
        constraints.append(
            cp.real(
                prior_prob[i]
                * cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
            )
            >= cp.real(
                cp.sum(
                    [
                        prior_prob[j]
                        * cp.trace(cp.matmul(problem_spec.states[j].data, PI_list[i]))
                        for j in range(n)
                    ]
                )
            )
            * (1 - beta[i])
        )

    constraints.append(cp.sum(PI_list) == I)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS, eps=1e-7, acceleration_lookback=10)
    print("Result =", result)
    print(f"CVXPY returns {prob.status}")
    logger.info(f"CVXPY returns {prob.status}")
    if prob.status != "optimal" and prob.status != "optimal_inaccurate":
        logger.error(f"CVXPY returns {prob.status}")
        return False

    povm = []
    # TODO
    # Check the rank of the Hermitian operators
    # Log the rank of the Hermitian operators
    # Dictionary of measured bitstrings to the target states
    bitstring_to_target_state = dict()

    strings_used = 0
    for i in range(n + 1):
        print(f"Solution for PI_{i} =")
        print(PI_list[i].value)
        u, s, v = np.linalg.svd(PI_list[i].value, hermitian=True)
        for j in range(len(s)):
            # Don't add if s[j] too small
            if s[j] > 1e-7:
                last_povm = u[:, j] * np.sqrt(s[j])
                bitstring_to_target_state[strings_used] = i
                povm.append(last_povm.conj())
                strings_used += 1

    problem_spec.bitstring_to_target_state = bitstring_to_target_state.copy()
    print("bitstring_to_target_state")
    print(bitstring_to_target_state)
    print("strings_used")
    print(strings_used)
    # TODO save strings_used somewhere
    #

    # TODO
    # Calculate the measurement operators
    ## last_op = np.eye(problem_spec.num_amps, dtype="complex128")
    ## for m in povm:
    ##     # Add "None" to transpose
    ##     # https://stackoverflow.com/a/11885718/13518808
    ##     op = np.multiply(m[None].T.conj(), m)
    ##     last_op -= op
    ##     # print(m)
    ## u, s, v = np.linalg.svd(last_op, hermitian=True)
    ## last_povm = u[:, 0] * np.sqrt(s[0])
    ## # print(last_povm)
    ## povm.append(last_povm.conj())

    # TODO Uncomment these lines
    ## if not verify_povm(povm):
    ##     print("POVM check failed.")
    ##     return False
    ## else:
    ##     print("POVM check passed.")

    # TODO
    # Verify solution
    np.set_printoptions(precision=4)
    print("Probabilities for each state:")
    total = 0
    p_d = 0
    p_inc = 0
    for i in range(n):
        probs = compute_event_probabilities(
            prior_prob[i], povm, problem_spec.states[i].data
        )
        print(probs)
        total += sum(probs)

    probability_matrix = []
    for i in range(n):
        probs = compute_event_probabilities(prior_prob[i], povm, problem_spec.states[i])
        updated_probs = [0] * (len(prior_prob) + 1)
        for j in range(strings_used):
            target_state_index = bitstring_to_target_state[j]
            updated_probs[target_state_index] += probs[j]
        # TODO
        # Postprocessing
        probability_matrix.append(updated_probs)

    for i in range(n):
        p_d += probability_matrix[i][i]
        p_inc += probability_matrix[i][n]

    print(f"Total probability = {total:.4f}")
    print(f"Success probability = {p_d:.4f}")
    print(f"Inconclusive probability = {p_inc:.4f}")
    # TODO also return these values
    print()

def gen_crossQD_dpp_problem(
    problem_spec: ProblemSpec,
    prior_prob=None,
    noise_level=0,  # TODO
    noise_type=None,  # TODO
) -> cp.Problem:
    """Generate a semidefinite programming problem that satisfies the
    disciplined parametrized programming (DPP) ruleset. After the values
    of the parameters are specified, the CVXPY solver should solve for a
    POVM for the cross quantum discrimination (crossQD) method.
    Now we assume that the effect of the noise is already included in
    the density matrices of `problem_spec` (`problem_spec.states`).
    Access the parameters with cvxpy_problem.param_dict() ?
    """
    assert problem_spec.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)
    np.set_printoptions(precision=4)

    n = problem_spec.num_states
    # prior_prob = cp.Parameter(n, name="prior_prob")
    alpha = cp.Parameter(n, name="alpha")
    beta = cp.Parameter(n, name="beta")

    # https://www.cvxpy.org/tutorial/dpp/index.html
    # DPP forbids taking the product of two parametrized expressions,
    # so we will not parametrize prior probabilities.
    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)
        logger.info(f"The prior probabilities is set to uniform (n = {n})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    # PI is the variable for the POVM we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(n + 1)
    ]

    objective = cp.Maximize(
        cp.sum(
            [
                prior_prob[i]
                * cp.real(
                    cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
                )
                for i in range(n)
            ]
        )
    )

    constraints = []

    # Constraint 1. Positive operators
    for i in range(n + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Conditional probability with respect to the measurement operator
    # prior_prob[i] is divided on both sides
    for i in range(n):
        expr_lhs = cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
        expr_rhs = cp.sum(
            [
                cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[j]))
                for j in range(n)
            ]
        )
        constraints.append(
            cp.real(expr_lhs) >= cp.real(expr_rhs) * (1 - alpha[i])
        )

    # Constraint 3. Conditional probability with respect to the input state
    for i in range(n):
        expr_lhs = prior_prob[i] * cp.trace(
            cp.matmul(problem_spec.states[i].data, PI_list[i])
        )
        expr_rhs = cp.sum(
            [
                prior_prob[j]
                * cp.trace(cp.matmul(problem_spec.states[j].data, PI_list[i]))
                for j in range(n)
            ]
        )
        constraints.append(
            cp.real(expr_lhs) >= cp.real(expr_rhs) * (1 - beta[i])
        )

    # Constraint 4. Completeness
    I = np.identity(problem_spec.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    return cp.Problem(objective, constraints)


def gen_crossQD_dpp_problem_symm(
    problem_spec: ProblemSpec,
    prior_prob=None,
    noise_level=0,  # TODO
    noise_type=None,  # TODO
) -> cp.Problem:
    """Generate a semidefinite programming problem that satisfies the
    disciplined parametrized programming (DPP) ruleset. After the values
    of the parameters are specified, the CVXPY solver should solve for a
    POVM for the cross quantum discrimination (crossQD) method.
    Now we assume that the effect of the noise is already included in
    the density matrices of `problem_spec` (`problem_spec.states`).
    Access the parameters with cvxpy_problem.param_dict() ?
    Here, we use only one parameter since we know the states are symmetric.
    """
    assert problem_spec.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)
    np.set_printoptions(precision=4)

    n = problem_spec.num_states
    # prior_prob = cp.Parameter(n, name="prior_prob")
    # Tolerance
    tol = cp.Parameter(name="tol")

    # https://www.cvxpy.org/tutorial/dpp/index.html
    # DPP forbids taking the product of two parametrized expressions,
    # so we will not parametrize prior probabilities.
    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)
        logger.info(f"The prior probabilities is set to uniform (n = {n})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    # PI is the variable for the POVM we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(n + 1)
    ]

    objective = cp.Maximize(
        cp.sum(
            [
                prior_prob[i]
                * cp.real(
                    cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
                )
                for i in range(n)
            ]
        )
    )

    constraints = []

    # Constraint 1. Positive operators
    for i in range(n + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Conditional probability with respect to the measurement operator
    # prior_prob[i] is divided on both sides
    for i in range(n):
        expr_lhs = cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
        expr_rhs = cp.sum(
            [
                cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[j]))
                for j in range(n)
            ]
        )
        constraints.append(cp.real(expr_lhs) >= cp.real(expr_rhs) * (1 - tol))

    # Constraint 3. Conditional probability with respect to the input state
    for i in range(n):
        expr_lhs = prior_prob[i] * cp.trace(
            cp.matmul(problem_spec.states[i].data, PI_list[i])
        )
        expr_rhs = cp.sum(
            [
                prior_prob[j]
                * cp.trace(cp.matmul(problem_spec.states[j].data, PI_list[i]))
                for j in range(n)
            ]
        )
        constraints.append(cp.real(expr_lhs) >= cp.real(expr_rhs) * (1 - tol))

    # Constraint 4. Completeness
    I = np.identity(problem_spec.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    return cp.Problem(objective, constraints)


def gen_crossQD_jsd_dpp_problem_symm(
    problem_spec: ProblemSpec,
    prior_prob=None,
    noise_level=0,  # TODO
    noise_type=None,  # TODO
) -> cp.Problem:
    """Generate a semidefinite programming problem that satisfies the
    disciplined parametrized programming (DPP) ruleset. After the values
    of the parameters are specified, the CVXPY solver should solve for a
    POVM for the cross quantum discrimination (crossQD) method.
    Now we assume that the effect of the noise is already included in
    the density matrices of `problem_spec` (`problem_spec.states`).
    Access the parameters with cvxpy_problem.param_dict() ?
    Here, we use only one parameter since we know the states are symmetric.
    """
    assert problem_spec.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)
    np.set_printoptions(precision=4)

    n = problem_spec.num_states
    # prior_prob = cp.Parameter(n, name="prior_prob")
    # Tolerance
    tol = cp.Parameter(name="tol")

    # https://www.cvxpy.org/tutorial/dpp/index.html
    # DPP forbids taking the product of two parametrized expressions,
    # so we will not parametrize prior probabilities.
    if prior_prob is None:
        prior_prob = np.ones(n) * (1 / n)
        logger.info(f"The prior probabilities is set to uniform (n = {n})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    # PI is the variable for the POVM we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(n + 1)
    ]

    objective = cp.Maximize(
        cp.sum(
            [
                prior_prob[i]
                * cp.real(
                    cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
                )
                for i in range(n)
            ]
        )
    )

    constraints = []

    # Constraint 1. Positive operators
    for i in range(n + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Conditional probability with respect to the measurement operator
    # prior_prob[i] is divided on both sides
    for i in range(n):
        expr_lhs = cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
        expr_rhs = cp.sum(
            [
                cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[j]))
                for j in range(n)
            ]
        )
        constraints.append(cp.real(expr_lhs) >= cp.real(expr_rhs) * (1 - tol))

    # Constraint 3. Conditional probability with respect to the input state
    for i in range(n):
        expr_lhs = prior_prob[i] * cp.trace(
            cp.matmul(problem_spec.states[i].data, PI_list[i])
        )
        expr_rhs = cp.sum(
            [
                prior_prob[j]
                * cp.trace(cp.matmul(problem_spec.states[j].data, PI_list[i]))
                for j in range(n)
            ]
        )
        constraints.append(cp.real(expr_lhs) >= cp.real(expr_rhs) * (1 - tol))

    # Constraint 4. Completeness
    I = np.identity(problem_spec.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    return cp.Problem(objective, constraints)


def apply_Eldar(
    problem_spec: ProblemSpec,
    prior_prob=None,
    min_prob=0,
    is_cvxpy_verbose=False,
):
    """Apply the method in Eldar's paper in 2003."""
    assert problem_spec.state_type == "statevector"
    logger = logging.getLogger(__name__)

    np_prec = 4
    np.set_printoptions(precision=np_prec)
    logger.info(f"Numpy print precision is set to {np_prec}")

    # Default to uniform distribution
    n = problem_spec.num_states
    if prior_prob is None:
        prior_prob = np.ones(n) * (-1 / n)
        logger.info(f"The prior probabilities is set to uniform (n = {n})")

    # Equation (6): Reciprocal states
    Phi_tilde = get_Phi_tilde(problem_spec)
    np.save(f"Phi_tilde_{problem_spec.case_id}.npy", Phi_tilde)
    logger.info(f"The matrix is saved to Phi_tilde_{problem_spec.case_id}.npy")

    # Equation (4)
    # Measurement operators without its measured probability
    # q is the array of Q_i
    q = []
    for i in range(n):
        q.append(np.multiply(Phi_tilde[i][None].T.conj(), Phi_tilde[i]))
    q = np.array(q)

    # Equation (20) ~ (24): Semidefinite programming (SDP) formulation
    p = cp.Variable(n)
    objective = cp.Minimize(1 + cp.sum(prior_prob @ p))

    constraints = []
    # TODO min_prob is a list of different numbers
    assert min_prob >= 0
    for i in range(n):
        constraints.append(min_prob <= p[i])
        constraints.append(p[i] <= 1)
    I = np.eye(problem_spec.num_amps, dtype="complex128")
    # I = np.identity(problem_spec.num_amps)
    expr = I
    for i in range(n):
        expr = expr - p[i] * q[i]
    constraints.append(expr >> 0)  # Matrix inequality in CVXPY uses >>

    prob = cp.Problem(objective, constraints)
    # TODO logger.info(f"CVXPY settings {}")
    t1 = time.time()
    # result = prob.solve(solver=cp.SCS, eps=1e-20)
    result = prob.solve(
        solver=cp.SCS, verbose=is_cvxpy_verbose, acceleration_lookback=10
    )  # , eps=1e-20)
    # result = prob.solve(solver=cp.CPLEX, verbose=is_cvxpy_verbose, eps=1e-20)
    t2 = time.time()
    solver_time_str = np.format_float_scientific(t2 - t1, precision=4)
    logger.info(f"Solution time (rounded) = {solver_time_str} seconds")
    logger.info(f"CVXPY returns {prob.status}")
    if prob.status != "optimal" and prob.status != "optimal_inaccurate":
        logger.error(f"CVXPY returns {prob.status}")
        return False
    logger.info(f"Result (rounded) = {result.round(4)}")
    sol = p.value
    logger.info(f"Solution (rounded) = {sol.round(4)}")
    p_succ = 0
    for i in range(n):
        p_succ += -prior_prob[i] * sol[i]

    # Obtain POVMs
    povm = []
    for i in range(n):
        if sol[i] <= 1e-4:
            logger.warning(
                f"sol[{i}] is zero or negative ({sol[i]} <= 1e-4), skip its operator"
            )
            continue
        else:
            # povm.append(np.sqrt(sol[i]) * Phi_tilde[i].conj())
            povm.append(sol[i] * q[i])
    povm.append(expr.value)
    distrib = []
    for i in range(n):
        distrib.append(-sol[i] * prior_prob[i])
    distrib.append(1 - p_succ)
    # TODO Remember the remaining operators
    result_dict = defaultdict(int)
    result_dict["povm"] = povm
    result_dict["sol"] = sol
    result_dict["distrib"] = distrib
    result_dict["p_succ"] = p_succ
    result_dict["p_err"] = 0
    result_dict["p_inc"] = 1 - p_succ
    return result_dict


def get_Phi_tilde(problem_spec: ProblemSpec):
    logger = logging.getLogger(__name__)
    tmp_arr = []
    for s in problem_spec.states:
        tmp_arr.append(s)
    Phi = np.transpose(tmp_arr)
    # TODO use vstacks instead and check if its correct
    # Phi_1 = np.vstack(problem_spec.states)
    # assert Phi == Phi_1
    np.save(f"Phi_{problem_spec.case_id}.npy", Phi)
    logger.info(f"The matrix is saved to Phi_{problem_spec.case_id}.npy")
    Phi_tilde = np.matmul(Phi, np.linalg.inv(np.matmul(Phi.conj().T, Phi)))
    Phi_tilde = Phi_tilde.round(15)  # Remove nearzero value
    Phi_tilde = Phi_tilde.T
    return Phi_tilde


if __name__ == "__main__":
    # TODO
    # Simple tests or solver comparison?
    si = SolverInterface(__name__)
    noise_level = 0.3
    case_id = f"{si.case_id}_mix_{noise_level}"
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    tracemalloc.start()
    problem = ProblemSpec(
        num_qubits=si.nq,
        num_states=si.ns,
        case_id=case_id,
        state_type="densitymatrix",
    )

    states, disturbance_states, combined_states = ProblemSpec.gen_noisy_states(
        num_qubits=si.nq,
        num_states=si.ns,
        seeds=get_random_seeds(si.ns, seed=si.state_seed),
        noise_level=noise_level,
        noise_rank=2,
    )

    # Ideal, but use the new formulation
    problem.set_states(
        state_type="densitymatrix", states=states, overwrite=True
    )
    povm = apply_Eldar_mix(self=problem)
    povm = apply_Eldar_mix(self=problem, p_I=0.3)
    print("Eldar 0")
    povm = apply_Eldar_mix_primal(problem_spec=problem, beta=0)
    print("Eldar 0.3")
    povm = apply_Eldar_mix_primal(problem_spec=problem, beta=0.3)
    print("Eldar 0.5")
    povm = apply_Eldar_mix_primal(problem_spec=problem, beta=0.5)
    # TODO Save POVM

    # Noisy
    problem.set_states(state_type="densitymatrix", states=combined_states)

    # Cross discrimination

    # TODO Density matrices with noise
    for i in range(10):
        a = 0.01 - i * 0.001
        print(a)
        povm = apply_dawei_mix_primal(problem_spec=problem, gamma=[a] * 3)
        print()

    for i in range(10):
        a = 0.01 - i * 0.001
        print(a)
        povm = apply_dawei_mix_primal(problem_spec=problem, gamma=[a] * 3)
        print()

    for i in range(10):
        a = 0.01 - i * 0.001
        print(a)
        povm = apply_koova_mix_primal(problem_spec=problem, gamma=[a] * 3)
        print()
    # TODO draw a plot of total success probability vs. threshold
    # TODO draw with prior probabilities

    logger.info(
        f"Memory (current, peak, in bytes) = {tracemalloc.get_traced_memory()}"
    )
    tracemalloc.stop()
    np.save(f"povm_{case_id}.npy", povm)
    logger.info(f"The POVM is saved to povm_{case_id}.npy")
    # TODO Remember the remaining operators
