from solve_mix import *
import numpy as np
import matplotlib.pyplot as plt
from utils.handy_states import *
from datetime import datetime
from qiskit.quantum_info import Statevector, DensityMatrix
from flow.interface import *

#
import argparse

# Top optimizer
from functools import partial
from skopt import gp_minimize
from skopt.space import Real
from torch import optim

# Distance
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon

# TODO Parse and plot result


# TODO Rename function
# TODO This with a single param
def get_js_distance(
    params,
    noise_level,
    problem_spec: ProblemSpec,
    cvxpy_problem: cp.Problem,
    ideal_distrib,
):
    print("Params:")
    print(params)
    # alpha = params[: problem_spec.num_states]
    # beta = params[problem_spec.num_states :]

    # cvxpy_problem = gen_crossQD_dpp_problem(problem_spec=problem_spec)
    cvxpy_problem.param_dict["tol"].value = params[0]
    # cvxpy_problem.param_dict["alpha"].value = alpha
    # cvxpy_problem.param_dict["beta"].value = beta
    result = cvxpy_problem.solve(
        solver=cp.SCS,
        # canon_backend=cp.SCIPY_CANON_BACKEND,
        verbose=True,
        mkl=True,
        requires_grad=True,
        warm_start=True,
    )

    # TODO Rewrite this (Parse result)

    """New"""
    logger = logging.getLogger(__name__)
    # print("Result =", result)
    # print(f"CVXPY returns {cvxpy_problem.status}")
    logger.info(f"CVXPY returns {cvxpy_problem.status}")
    if (
        cvxpy_problem.status != "optimal"
        and cvxpy_problem.status != "optimal_inaccurate"
    ):
        logger.error(f"CVXPY returns {cvxpy_problem.status}")
        js_dist = return_default(ideal_distrib)
        return js_dist
        # return False

    if cvxpy_problem.status == "optimal_inaccurate":
        print("Be careful")
        # TODO Prevent further errors

    # TODO
    # Check the rank of the Hermitian operators
    # Log the rank of the Hermitian operators
    n = problem_spec.num_states
    PI_list = cvxpy_problem.variables()
    povm = [PI_list[i].value for i in range(len(PI_list))]
    print(povm)

    # TODO
    # Verify solution
    np.set_printoptions(precision=4)
    print("Probabilities for each state:")
    total = 0
    p_d = 0
    p_inc = 0
    prior_prob = problem_spec.prior_prob
    # povm = vectors_to_povm(povm)
    if verify_povm_matrix(povm):
        print("Passed POVM test")
    probability_matrix = []
    for i in range(n):
        probs = compute_event_probabilities(
            prior_prob[i], povm, problem_spec.states[i].data
        )
        print(probs)
        total += sum(probs)
        probability_matrix.append(probs)
    probability_matrix = np.array(probability_matrix)
    for i in range(n):
        p_d += probability_matrix[i, i]
        p_inc += probability_matrix[i, n]

    print(f"Total probability = {total:.4f}")
    print(f"Success probability = {p_d:.4f}")
    print(f"Inconclusive probability = {p_inc:.4f}")
    # TODO also return these values

    result_dict = defaultdict()
    result_dict["povm"] = povm
    result_dict["total_prob"] = total
    # result_dict["p_succ"] = p_succ
    # result_dict["p_err"] = p_err
    result_dict["p_inc"] = p_inc
    result_dict["PI_list"] = [PI_list[i].value for i in range(len(PI_list))]
    ## result_dict["bitstring_to_target_state"] = bitstring_to_target_state
    ## result_dict["strings_used"] = strings_used

    # return result_dict
    result = result_dict
    """New end"""

    # Parse result
    # Calculate and return distribution difference
    ops = result["povm"]
    # TODO Calculate distribution
    # from flow.plots import calculate_prob_matrix

    print(ops)
    if not verify_povm_matrix(ops, rtol=1e-4):
        print("ideal_distrib")
        print(ideal_distrib)
        result_distrib = np.zeros(len(ideal_distrib))
        result_distrib[-1] = 1
        print("result_distrib")
        print(result_distrib, flush=True)
        js_dist = jensenshannon(ideal_distrib, result_distrib)
        return js_dist

    mat = calculate_prob_matrix_simple(
        prior_probs=problem_spec.prior_prob,
        povm=ops,
        states=problem_spec.states,
        ## bitstring_to_target_state=result["bitstring_to_target_state"],
        ## strings_used=result["strings_used"],
    )

    result_distrib = np.diagonal(mat)
    result_distrib = list(result_distrib)
    result_distrib.append(result["p_inc"])

    # print(type(ideal_distrib))
    # print(type(result_distrib))
    print("ideal_distrib")
    print(ideal_distrib)
    print("result_distrib")
    print(result_distrib, flush=True)
    js_dist = jensenshannon(ideal_distrib, result_distrib)
    return js_dist


def return_default(ideal_distrib):
    print("ideal_distrib")
    print(ideal_distrib)
    result_distrib = np.zeros(len(ideal_distrib))
    result_distrib[-1] = 1
    print("result_distrib")
    print(result_distrib, flush=True)
    js_dist = jensenshannon(ideal_distrib, result_distrib)
    return js_dist


def grad_descent_param(alpha_init, beta_init, cvxpy_problem):
    cvxpy_problem.param_dict["tol"].value = max(alpha_init + beta_init)
    tol_list = []
    grad_list = []
    step = 0.1
    for i in range(20):
        result = cvxpy_problem.solve(
            solver=cp.SCS,
            # canon_backend=cp.SCIPY_CANON_BACKEND,
            verbose=True,
            mkl=True,
            requires_grad=True,
            warm_start=True,
        )
        cvxpy_problem.backward()
        tol = cvxpy_problem.param_dict["tol"].value
        print("tol =", tol)
        tol_list.append(tol)
        grad = cvxpy_problem.param_dict["tol"].gradient
        print("grad =", grad)
        grad_list.append(grad)
        # AdamW
        cvxpy_problem.param_dict["tol"].value -= step * grad
        if cvxpy_problem.param_dict["tol"].value <= 0:
            cvxpy_problem.param_dict["tol"].value = 0.1

    plt.plot(range(20), tol_list, label="tolerance")
    plt.plot(range(20), grad_list, label="gradient")
    plt.legend()
    plt.savefig(f"opt_{noise_level:3f}.png")
    plt.close()


def bayesian_opt(
    get_js_distance,
    num_states,
    problem_symm_states_1,
    ideal_result,
    alpha_init,
    beta_init,
    cvxpy_problem,
):
    objective = partial(
        get_js_distance,
        noise_level=noise_level,
        problem_spec=problem_symm_states_1,
        cvxpy_problem=cvxpy_problem,
        ideal_distrib=ideal_result["distrib"],
    )
    space = [Real(0.0, 1.0 - 1 / num_states, name=f"tol")]
    gp_result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=20,
        random_state=42,
        verbose=True,
        # x0=alpha_init + beta_init,
        acq_func="EI",
        x0=[max(alpha_init + beta_init)],
        xi=0.001,
    )

    # Best parameters and value
    print("Noise level")
    print(noise_level)
    print("Result")
    print(gp_result)
    print("Best parameters:", gp_result.x)
    print("Best value:", gp_result.fun)


if __name__ == "__main__":
    si = SolverInterface(__name__)
    num_qubits = si.nq
    num_states = si.ns
    state_vec = 0
    if si.quick_access == "coh_symm":
        state_vec = sv_coh_symm_small(num_qubits=num_qubits)
        # num_qubits = si.nq
        num_states = 3
    elif si.quick_access == "sic_symm":
        state_vec = sv_sic_symm_small()["states"]
        for item in state_vec:
            print(format_complex_array(item.data))
        num_qubits = 2
        num_states = 3
        from utils.inner_product import inner_products_table

        inner_products_table([item.data for item in state_vec])

    dense_mat = [DensityMatrix(_) for _ in state_vec]

    problem_symm_states_1 = ProblemSpec(
        num_qubits=num_qubits,
        num_states=num_states,
        seed=33,
        case_id=f"q{num_qubits}_n{num_states}_symm",
        state_type="statevector",
    )
    problem_symm_states_1.set_states(
        state_type="statevector",
        states=state_vec,
        overwrite=True,
    )
    print("Is linear indep", problem_symm_states_1.is_lin_indep())
    ideal_result = apply_Eldar(problem_spec=problem_symm_states_1)

    print(verify_povm_matrix(ideal_result["povm"]))
    p_inc_ideal = ideal_result["p_inc"]

    ## TODO Modify below

    # I have two different "probability distributions"
    # One includes the (small) error probabilities in the conclusive probabilities
    # One excludes these probabilities and normalize the rest.
    # I think we have to have more than one partial functions
    # One for noise_level, one for params?

    noise_levels = [0.1 ** (6 - 0.25 * i) for i in range(10)]
    disturbance_states = [
        DensityMatrix(
            ProblemSpec.depolarizing_noise_channel(num_qubits=num_qubits)
        )
        for _ in range(num_states)
    ]

    for noise_level in noise_levels:
        # Combine these
        # TODO dense_mat_noise = []
        combined_states = [
            (1 - noise_level) * dense_mat[_]
            + noise_level * disturbance_states[_].data
            for _ in range(num_states)
        ]
        problem_symm_states_1.set_states(
            state_type="densitymatrix",
            states=combined_states,
            overwrite=True,
        )

        noisy_result = apply_Eldar_mix_primal(
            problem_spec=problem_symm_states_1,
            beta=p_inc_ideal,
            is_cvxpy_verbose=True,
        )

        # TODO Calculate initial values for alpha and beta
        mat = calculate_prob_matrix_simple(
            prior_probs=problem_symm_states_1.prior_prob,
            povm=noisy_result["PI_list"],
            states=problem_symm_states_1.states,
            ## bitstring_to_target_state=result["bitstring_to_target_state"],
            ## strings_used=result["strings_used"],
        )
        alpha_init, beta_init = calculate_errors(mat)
        print("alpha_init", alpha_init, flush=True)
        print("beta_init", beta_init, flush=True)
        print(
            "max(alpha_init + beta_init)",
            max(alpha_init + beta_init),
            flush=True,
        )

        # cvxpy_problem = gen_crossQD_dpp_problem(problem_spec=problem_symm_states_1)
        cvxpy_problem = gen_crossQD_dpp_problem_symm(
            problem_spec=problem_symm_states_1
        )

        ## bounds = [(0, 1) for _ in range(num_states * 2)]
        ## result = differential_evolution(objective, bounds, seed=42)

        print(datetime.now())

        # Method 1. Grid Search
        # Method 2. Backward propagation
        # grad_descent_param(alpha_init, beta_init, cvxpy_problem)

        # Method 3. Bayesian optimization
        # Define parameter bounds (all floats between 0 and 1)
        # space = [Real(0.0, 1.0 - 1 / num_states, name=f"alpha_{i}") for i in range(num_states)] + [
        #     Real(0.0, 1.0 - 1 / num_states, name=f"beta_{i}") for i in range(num_states)
        # ]
        bayesian_opt(
            get_js_distance,
            num_states,
            problem_symm_states_1,
            ideal_result,
            alpha_init,
            beta_init,
            cvxpy_problem,
        )
