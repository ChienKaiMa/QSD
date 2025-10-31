import logging.config
import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../flow")
from flow.interface import *
from flow.problem_spec import *
# from flow.plots import *
from flow.verify_povm import *
import numpy as np
import cvxpy as cp
from scipy.linalg import null_space
from utils.get_random_seeds import get_random_seeds
from utils.prob_matrix import *
import time
import tracemalloc
from collections import defaultdict


def apply_frio(
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
    # print("Result =", result)
    # print(f"CVXPY returns {prob.status}")
    logger.info(f"CVXPY returns {prob.status}")
    if prob.status != "optimal":
        logger.error(f"CVXPY returns {prob.status}")
        return

    povm = []
    # TODO
    # Check the rank of the Hermitian operators
    # Log the rank of the Hermitian operators
    # Dictionary of measured bitstrings to the target states
    bitstring_to_target_state = dict()

    strings_used = 0
    for i in range(n + 1):
        ## print(f"Solution for PI_{i} =")
        ## print(PI_list[i].value)
        u, s, v = np.linalg.svd(PI_list[i].value, hermitian=True)
        for j in range(len(s)):
            # Don't add if s[j] too small
            if s[j] > 1e-7:
                last_povm = u[:, j] * np.sqrt(s[j])
                bitstring_to_target_state[strings_used] = i
                povm.append(last_povm.conj())
                strings_used += 1

    problem_spec.bitstring_to_target_state = bitstring_to_target_state.copy()
    ## print("bitstring_to_target_state")
    ## print(bitstring_to_target_state)
    ## print("strings_used")
    ## print(strings_used)
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
    # print("Probabilities for each state:")
    total = 0
    p_d = 0
    p_inc = 0
    povm = vectors_to_povm(povm)
    for i in range(n):
        probs = compute_event_probabilities(
            prior_prob[i], povm, problem_spec.states[i].data
        )
        # print(probs)
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

    ## print(f"Total probability = {total:.4f}")
    ## print(f"Success probability = {p_d:.4f}")
    ## print(f"Inconclusive probability = {p_inc:.4f}")
    ## # TODO also return these values
    ## print()
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


def apply_crossQSD(
    problem_spec: ProblemSpec,
    prior_prob=None,
    alpha=None,
    beta=None,
    noise_level=0,
    cvxpy_settings: dict | None = None,
):
    """Apply the cross quantum state discrimination method.
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

    cvxpy_problem = crossQSD_dpp_problem(
        problem_spec=problem_spec,
        prior_prob=prior_prob,
    )

    cvxpy_problem.param_dict["alpha"].value = alpha
    cvxpy_problem.param_dict["beta"].value = beta
    cvxpy_problem.solve(**cvxpy_settings)
    return cvxpy_problem


def crossQSD_dpp_problem(
    problem_spec: ProblemSpec,
    prior_prob=None,
    noise_level=0,  # TODO
    noise_type=None,  # TODO
) -> cp.Problem:
    """Generate a semidefinite programming problem that satisfies the
    disciplined parametrized programming (DPP) ruleset. After the values
    of the parameters are specified, the CVXPY solver should solve for a
    POVM for the cross quantum state discrimination (crossQSD) method.
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


def apply_crossQD_2(
    problem_spec: ProblemSpec,
    prior_prob=None,
    alpha=None,
    beta=None,
    gamma=None,
    noise_level=0,
    reuse_fig=None,
    is_cvxpy_verbose=False,
    max_iters=int(5e5),
    eps_abs=1e-7,
    eps_rel=1e-7,
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

    # TODO isnum(gamma)
    if gamma is None:
        gamma = 0
    logger.info(f"The inconclusive lower bound (gamma) is set to {gamma}")

    cvxpy_problem = gen_crossQD_2_dpp_problem(
        problem_spec=problem_spec,
        prior_prob=prior_prob,
    )

    cvxpy_problem.param_dict["alpha"].value = alpha
    cvxpy_problem.param_dict["beta"].value = beta
    cvxpy_problem.param_dict["gamma"].value = gamma
    result = cvxpy_problem.solve(
        solver=cp.SCS,
        verbose=is_cvxpy_verbose,
        requires_grad=True,
        mkl=True,
        max_iters=int(max_iters),
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        acceleration_lookback=10,
        warm_start=True,
        # canon_backend=cp.SCIPY_CANON_BACKEND,
        # time_limit_secs=180,
    )
    prob = cvxpy_problem
    PI_list = cvxpy_problem.variables()

    # print("Result =", result)
    # print(f"CVXPY returns {prob.status}")
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
        u, s, v = np.linalg.svd(PI_list[i].value, hermitian=True)
        for j in range(len(s)):
            # Don't add if s[j] too small
            if s[j] > 1e-7:
                last_povm = u[:, j] * np.sqrt(s[j])
                bitstring_to_target_state[strings_used] = i
                povm.append(last_povm.conj())
                strings_used += 1

    problem_spec.bitstring_to_target_state = bitstring_to_target_state.copy()

    # Verify solution
    np.set_printoptions(precision=4)
    # print("Probabilities for each state:")
    total = 0
    p_d = 0
    p_inc = 0
    povm = vectors_to_povm(povm)
    for i in range(n):
        probs = compute_event_probabilities(
            prior_prob[i], povm, problem_spec.states[i].data
        )
        # print(probs)
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
        # print(updated_probs)
        probability_matrix.append(updated_probs)

    for i in range(n):
        p_d += probability_matrix[i][i]
        p_inc += probability_matrix[i][n]

    # print(f"Total probability = {total:.4f}")
    # print(f"Success probability = {p_d:.4f}")
    # print(f"Inconclusive probability = {p_inc:.4f}")

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


def gen_crossQD_2_dpp_problem(
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
    gamma = cp.Parameter(name="gamma")

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

    # Constraint 5. Lower bound of inconclusive outcome
    expr_pinc = cp.sum(
        [
            prior_prob[j]
            * cp.trace(cp.matmul(problem_spec.states[j].data, PI_list[n]))
            for j in range(n)
        ]
    )
    constraints.append(cp.real(expr_pinc) >= gamma)

    return cp.Problem(objective, constraints)


def apply_Eldar(
    problem_spec: ProblemSpec,
    prior_prob=None,
    min_prob=0,
    is_cvxpy_verbose=False,
    cvxpy_settings=None,
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
        prior_prob = np.ones(n) * (1 / n)
        logger.info(f"The prior probabilities is set to uniform (n = {n})")

    if cvxpy_settings is None:
        cvxpy_settings = {
            "solver": cp.SCS,
            "verbose": is_cvxpy_verbose,
            "acceleration_lookback": 10,
        }

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
    objective = cp.Minimize(1 - cp.sum(prior_prob @ p))

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
    logger.info(f"CVXPY settings {cvxpy_settings}")
    t1 = time.time()
    result = prob.solve(**cvxpy_settings)
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
        p_succ += prior_prob[i] * sol[i]

    # Obtain POVMs
    povm = []
    povm_vectors = []
    for i in range(n):
        if sol[i] <= 1e-4:
            logger.warning(
                f"sol[{i}] is zero or negative ({sol[i]} <= 1e-4), skip its operator"
            )
            continue
        else:
            # povm.append(np.sqrt(sol[i]) * Phi_tilde[i].conj())
            povm_vectors.append(np.sqrt(sol[i]) * Phi_tilde[i].conj())
            povm.append(sol[i] * q[i].T)

    povm.append(expr.value)
    distrib = []
    for i in range(n):
        distrib.append(sol[i] * prior_prob[i])
    distrib.append(1 - p_succ)
    # TODO Remember the remaining operators
    result_dict = defaultdict(int)
    result_dict["povm"] = povm
    result_dict["povm_vectors"] = povm_vectors
    result_dict["sol"] = sol
    result_dict["distrib"] = distrib
    result_dict["p_succ"] = p_succ
    result_dict["p_err"] = 0
    result_dict["p_inc"] = 1 - p_succ
    return result_dict


def get_prob_succ_expr(
    problem_spec: ProblemSpec,
    prior_prob: list[float],
    PI_list: list[cp.Variable],
) -> cp.Expression:
    """Returns a CVXPY expression for the total probability of successful
    state discrimination.
    """

    return cp.sum(
        [
            prior_prob[i]
            * cp.real(
                cp.trace(cp.matmul(problem_spec.states[i].data, PI_list[i]))
            )
            for i in range(problem_spec.num_states)
        ]
    )


def get_l1_dist_expr(
    problem_spec: ProblemSpec,
    prior_prob: list[float],
    PI_list: list[cp.Variable],
    prob_mat: list[float],
):
    """Returns a CVXPY expression for the L1 distance between two flattened
    probability matrices.
    """
    k = problem_spec.num_states
    l1_expr = 0
    for i in range(k):
        ideal_row = prob_mat[i]
        for j in range(k + 1):
            expr_rhs = prior_prob[i] * cp.trace(
                cp.matmul(problem_spec.states[i].data, PI_list[j])
            )
            l1_expr += cp.abs(ideal_row[j] - cp.real(expr_rhs))
    return l1_expr


def med_problem(
    problem_spec: ProblemSpec,
    prior_prob: list[float] | None = None,
) -> cp.Problem:
    """Minimum error discrimination."""

    assert problem_spec.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)

    k = problem_spec.num_states

    # PI is the variable for the POVM elements we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(k)
    ]

    # https://www.cvxpy.org/tutorial/dpp/index.html
    # DPP forbids taking the product of two parametrized expressions,
    # so we will not parametrize prior probabilities.
    # Actually for this formulation, it is fine to parametrize it.
    # For code uniformity, we will not do it.
    # prior_prob = cp.Parameter(k, name="prior_prob")
    if prior_prob is None:
        prior_prob = np.ones(k) * (1 / k)
        logger.info(f"The prior probabilities is set to uniform (k = {k})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    prob_succ_expr = get_prob_succ_expr(
        problem_spec=problem_spec, prior_prob=prior_prob, PI_list=PI_list
    )

    objective = cp.Maximize(prob_succ_expr)

    constraints = []

    # Constraint 1. Positive operators
    for i in range(k):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Completeness
    I = np.identity(problem_spec.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    return cp.Problem(objective, constraints)


def med_plus_problem(
    problem_spec: ProblemSpec,
    prior_prob: list[float] | None = None,
) -> cp.Problem:
    """MED+ is minimum error discrimination that includes an additional
    POVM element that corresponds to inconclusive outcomes.
    """

    assert problem_spec.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)

    k = problem_spec.num_states

    # PI is the variable for the POVM elements we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(k + 1)
    ]

    # https://www.cvxpy.org/tutorial/dpp/index.html
    # DPP forbids taking the product of two parametrized expressions,
    # so we will not parametrize prior probabilities.
    # Actually for this formulation, it is fine to parametrize it.
    # For code uniformity, we will not do it.
    # prior_prob = cp.Parameter(k, name="prior_prob")
    if prior_prob is None:
        prior_prob = np.ones(k) * (1 / k)
        logger.info(f"The prior probabilities is set to uniform (k = {k})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    prob_succ_expr = get_prob_succ_expr(
        problem_spec=problem_spec, prior_prob=prior_prob, PI_list=PI_list
    )

    objective = cp.Maximize(prob_succ_expr)

    constraints = []

    # Constraint 1. Positive operators
    for i in range(k + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Completeness
    I = np.identity(problem_spec.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    return cp.Problem(objective, constraints)


def frio_problem(
    problem_spec: ProblemSpec,
    p_inc_lb: float,
    prior_prob: list[float] | None = None,
):
    """Apply the method in Eldar's paper in 2004 with the primal problem."""

    assert problem_spec.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)

    assert 0 <= p_inc_lb < 1

    k = problem_spec.num_states

    # PI is the variable for the POVM elements we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(k + 1)
    ]

    if prior_prob is None:
        prior_prob = np.ones(k) * (1 / k)
        logger.info(f"The prior probabilities is set to uniform (k = {k})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    prob_succ_expr = get_prob_succ_expr(
        problem_spec=problem_spec, prior_prob=prior_prob, PI_list=PI_list
    )

    objective = cp.Maximize(prob_succ_expr)

    constraints = []

    # Constraint 1. Positive operators
    for i in range(k + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Completeness
    I = np.identity(problem_spec.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    # Constraint 3.
    expr_rhs = cp.real(
        cp.sum(
            [
                prior_prob[j]
                * cp.trace(cp.matmul(problem_spec.states[j].data, PI_list[k]))
                for j in range(k)
            ]
        )
    )
    constraints.append(expr_rhs >= p_inc_lb)

    return cp.Problem(objective, constraints)


def min_l1_problem(
    ideal_distrib,
    qsd_problem: ProblemSpec,
    prior_prob: list[float] | None = None,
):
    assert qsd_problem.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)

    k = qsd_problem.num_states

    # PI is the variable for the POVM elements we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(qsd_problem.num_amps, qsd_problem.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(k + 1)
    ]

    if prior_prob is None:
        prior_prob = np.ones(k) * (1 / k)
        logger.info(f"The prior probabilities is set to uniform (k = {k})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    l1_expr = 0
    for i in range(k):
        ideal_row = ideal_distrib[i]
        for j in range(k + 1):
            expr_rhs = prior_prob[i] * cp.trace(
                cp.matmul(qsd_problem.states[i].data, PI_list[j])
            )
            l1_expr += cp.abs(ideal_row[j] - cp.real(expr_rhs))
    objective = cp.Minimize(l1_expr)

    constraints = []

    # Constraint 1. Positive operators
    for i in range(k + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Completeness
    I = np.identity(qsd_problem.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    return cp.Problem(objective, constraints)


def min_ss_problem(
    ideal_distrib,
    qsd_problem: ProblemSpec,
    prior_prob: list[float] | None = None,
):
    """SS stands for 'sum of squares'."""
    assert qsd_problem.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)

    k = qsd_problem.num_states

    # PI is the variable for the POVM elements we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(qsd_problem.num_amps, qsd_problem.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(k + 1)
    ]

    if prior_prob is None:
        prior_prob = np.ones(k) * (1 / k)
        logger.info(f"The prior probabilities is set to uniform (k = {k})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    ss_expr = 0
    for i in range(k):
        ideal_row = ideal_distrib[i]
        for j in range(k + 1):
            expr_rhs = prior_prob[i] * cp.trace(
                cp.matmul(qsd_problem.states[i].data, PI_list[j])
            )
            ss_expr += (ideal_row[j] - cp.real(expr_rhs)) ** 2
    objective = cp.Minimize(ss_expr)

    constraints = []

    # Constraint 1. Positive operators
    for i in range(k + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Completeness
    I = np.identity(qsd_problem.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    return cp.Problem(objective, constraints)


def meco_problem(
    ideal_distrib,
    qsd_problem: ProblemSpec,
    prior_prob: list[float] | None = None,
):
    # Renamed from max_psucc_min_diff_problem
    assert qsd_problem.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)

    k = qsd_problem.num_states

    # PI is the variable for the POVM elements we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(qsd_problem.num_amps, qsd_problem.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(k + 1)
    ]

    if prior_prob is None:
        prior_prob = np.ones(k) * (1 / k)
        logger.info(f"The prior probabilities is set to uniform (k = {k})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    prob_succ_expr = get_prob_succ_expr(
        problem_spec=qsd_problem, prior_prob=prior_prob, PI_list=PI_list
    )

    objective = cp.Maximize(prob_succ_expr)

    constraints = []

    # Constraint 1. Positive operators
    for i in range(k + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Completeness
    I = np.identity(qsd_problem.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    # Constraint 3. Match the ideal distribution
    for i in range(k):
        ideal_row = ideal_distrib[i]
        for j in range(k):
            expr_rhs = prior_prob[i] * cp.trace(
                cp.matmul(qsd_problem.states[i].data, PI_list[j])
            )
            if i == j:
                constraints.append(ideal_row[j] >= cp.real(expr_rhs))
            else:
                constraints.append(ideal_row[j] <= cp.real(expr_rhs))
        expr_rhs = prior_prob[i] * cp.trace(
            cp.matmul(qsd_problem.states[i].data, PI_list[k])
        )
        constraints.append(ideal_row[k] <= cp.real(expr_rhs))

    return cp.Problem(objective, constraints)


def solveQSDProblem(
    cvxpy_qsd_problem: cp.Problem,
    cvxpy_settings: dict,
):
    """Solves the CVXPY QSD problem.
    The function will return 0 if the input CVXPY problem is DPP but the
    parameters are not set.
    """

    try:
        cvxpy_qsd_problem.solve(**cvxpy_settings)
    except:
        logging.error(
            "Please fill the parameters in the cp.Problem before using this function."
        )
    return


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


def hybrid_obj_problem(
    problem_spec: ProblemSpec,
    ideal_distrib,
    param_a,
    prior_prob: list[float] | None = None,
) -> cp.Problem:
    assert problem_spec.state_type == "densitymatrix"
    logger = logging.getLogger(__name__)

    k = problem_spec.num_states

    # PI is the variable for the POVM elements we try to solve for.
    PI_list = [
        cp.Variable(
            shape=(problem_spec.num_amps, problem_spec.num_amps),
            hermitian=True,
            name=f"PI_{i}",
        )
        for i in range(k + 1)
    ]

    # https://www.cvxpy.org/tutorial/dpp/index.html
    # DPP forbids taking the product of two parametrized expressions,
    # so we will not parametrize prior probabilities.
    # Actually for this formulation, it is fine to parametrize it.
    # For code uniformity, we will not do it.
    # prior_prob = cp.Parameter(k, name="prior_prob")
    if prior_prob is None:
        prior_prob = np.ones(k) * (1 / k)
        logger.info(f"The prior probabilities is set to uniform (k = {k})")
    else:
        logger.info(f"The prior probabilities is set to {prior_prob}")

    prob_succ_expr = get_prob_succ_expr(
        problem_spec=problem_spec, prior_prob=prior_prob, PI_list=PI_list
    )

    l1_expr = get_l1_dist_expr(
        problem_spec=problem_spec,
        prior_prob=prior_prob,
        PI_list=PI_list,
        prob_mat=ideal_distrib,
    )

    obj_expr = prob_succ_expr - param_a * l1_expr

    objective = cp.Maximize(obj_expr)
    constraints = []

    # Constraint 1. Positive operators
    for i in range(k + 1):
        constraints.append(PI_list[i] >> 0)

    # Constraint 2. Completeness
    I = np.identity(problem_spec.num_amps)
    constraints.append(cp.sum(PI_list) == I)

    return cp.Problem(objective, constraints)


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

    print("Eldar 0")
    povm = apply_frio(problem_spec=problem, beta=0)
    print("Eldar 0.3")
    povm = apply_frio(problem_spec=problem, beta=0.3)
    print("Eldar 0.5")
    povm = apply_frio(problem_spec=problem, beta=0.5)
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
