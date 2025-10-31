from solve_mix import *
import numpy as np
import matplotlib.pyplot as plt
from utils.sparse import calculate_sparsity
from qutip import coherent, coherent_dm

# Top optimizer
from functools import partial
from scipy.optimize import differential_evolution
from skopt import gp_minimize
from skopt.space import Real


def get_distrib(
    params,
    noise_level,
    problem_spec: ProblemSpec,
    ideal_distrib,
):
    print("Params:")
    print(params)
    alpha = params[: problem_spec.num_states]
    beta = params[problem_spec.num_states :]
    result = apply_crossQD(
        problem_spec=problem_spec,
        alpha=alpha,
        beta=beta,
        isCvxpyVerbose=True,
    )
    # Parse result
    # Calculate and return distribution difference
    ops = result["PI_list"]
    # TODO Calculate distribution
    from flow.plots import calculate_prob_matrix

    # print(ops)
    # print(verify_povm_matrix(ops))
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
    from scipy.stats import entropy, wasserstein_distance
    from scipy.spatial.distance import jensenshannon

    # print(type(ideal_distrib))
    # print(type(result_distrib))
    print("ideal_distrib")
    print(ideal_distrib)
    print("result_distrib")
    print(result_distrib, flush=True)
    js_dist = jensenshannon(ideal_distrib, result_distrib)
    return js_dist


if __name__ == "__main__":
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    alphas = [np.exp(angles[i] * 1j) for i in range(len(angles))]
    symm_states_1 = [
        coherent(N=64, alpha=1 * alphas[i]) for i in range(len(alphas))
    ]
    symm_states_1_dm = [
        coherent_dm(N=64, alpha=1 * alphas[i]) for i in range(len(alphas))
    ]
    symm_states_matrix_1 = [
        symm_states_1[i].data.to_array().flatten()
        for i in range(len(symm_states_1))
    ]
    symm_states_1_dm_matrix = [
        symm_states_1_dm[i].data.to_array()
        for i in range(len(symm_states_1_dm))
    ]

    sparsity = calculate_sparsity(coherent_dm(N=64, alpha=1).data.to_array())
    print(sparsity)
    problem_symm_states_1 = ProblemSpec(
        num_qubits=6,
        num_states=3,
        seed=33,
        case_id="q6_n3_symm_cohN64a1",
        state_type="statevector",
    )
    problem_symm_states_1.set_states(
        state_type="statevector",
        states=symm_states_matrix_1,
        overwrite=True,
    )

    ideal_result = apply_Eldar(problem_spec=problem_symm_states_1)
    
    print(verify_povm_matrix(ideal_result["povm"]))
    p_inc_ideal = ideal_result["p_inc"]

    ## TODO Modify below

    noise_level = 0.0001

    disturbance_states = [
        DensityMatrix(ProblemSpec.depolarizing_noise_channel(6))
        for _ in range(4)
    ]
    # Combine these
    combined_states = [
        (1 - noise_level) * symm_states_1_dm_matrix[_]
        + noise_level * disturbance_states[_].data
        for _ in range(len(symm_states_1_dm))
    ]
    problem_symm_states_1.set_states(
        state_type="densitymatrix",
        states=combined_states,
        overwrite=True,
    )

    noisy_result = apply_Eldar_mix_primal(
        problem_spec=problem_symm_states_1,
        beta=p_inc_ideal,
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
    print(alpha_init, flush=True)
    print(beta_init, flush=True)

    # I have two different "probability distributions"
    # One includes the (small) error probabilities in the conclusive probabilities
    # One excludes these probabilities and normalize the rest.
    # I think we have to have more than one partial functions
    # One for noise_level, one for params?

    objective = partial(
        get_distrib,
        problem_spec=problem_symm_states_1,
        noise_level=0.1,
        ideal_distrib=ideal_result["distrib"],
    )

    num_states = 3
    ## bounds = [(0, 1) for _ in range(num_states * 2)]
    ## result = differential_evolution(objective, bounds, seed=42)

    # Define parameter bounds (all floats between 0 and 1)
    space = [Real(0.0, 1.0, name=f"alpha_{i}") for i in range(num_states)] + [
        Real(0.0, 1.0, name=f"beta_{i}") for i in range(num_states)
    ]
    from datetime import datetime
    print(datetime.now())

    # Run Bayesian Optimization
    gp_result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=20,
        random_state=42,
        verbose=True,
        x0=alpha_init + beta_init,
    )

    # Best parameters and value
    print("Result")
    print(gp_result)
    print("Best parameters:", gp_result.x)
    print("Best value:", gp_result.fun)
