# Replicate Fig.1 in the paper

import sys

sys.path.append("../")


from flow.solve_mix import *
from flow.interface import *
from utils.handy_states import *


# Define input states
state_info = coh_asymm_small(num_qubits=3)
num_qubits = state_info["num_qubits"]
num_states = state_info["num_states"]
state_vec = state_info["state_vec"]
dense_mat = state_info["dense_mat"]


qsd_problem = ProblemSpec(
    num_qubits=num_qubits,
    num_states=num_states,
)


qsd_problem.set_states(
    state_type="statevector",
    states=state_vec,
    overwrite=True,
)

cvxpy_settings = {"solver": cp.MOSEK, "verbose": False, "eps": 1e-8}
ideal_result = apply_Eldar(qsd_problem, cvxpy_settings=cvxpy_settings)
np.format_float_positional(ideal_result["p_succ"], 6)


ideal_result["sol"]


# Fix apply_Eldar to place 0 matrix instead of skipping the operator
ideal_povm = [
    ideal_result["povm"][0],
    np.zeros(shape=(8, 8)),
    ideal_result["povm"][1],
    np.identity(8) - ideal_result["povm"][0] - ideal_result["povm"][1],
]

print(verify_povm_matrix(ideal_povm))

ideal_prob_mat = calculate_prob_matrix_simple(
    prior_probs=[1 / num_states] * num_states,
    povm=ideal_povm,
    states=dense_mat,
)
ideal_p_succ = 0
for i in range(num_states):
    ideal_p_succ += ideal_prob_mat[i][i]
print(f"p_succ = {ideal_p_succ:.4f}")
for line in ideal_prob_mat:
    print(line)
print()


# Calculate the noise effect on the POVMs
disturbance_states = [
    DensityMatrix(ProblemSpec.depolarizing_noise_channel(num_qubits=num_qubits))
    for _ in range(num_states)
]


params = [0.1 ** (6 - 0.25 * i) for i in range(25)]


def test_crossQSD(noise_param, param):
    print(f"tol = {np.format_float_scientific(param, 5)}")
    noisy_dense_mat = [
        (1 - noise_param) * dense_mat[_]
        + noise_param * disturbance_states[_].data
        for _ in range(num_states)
    ]
    qsd_problem.set_states(
        state_type="densitymatrix",
        states=noisy_dense_mat,
        overwrite=True,
    )

    cvxpy_crossqsd_problem = apply_crossQSD(
        qsd_problem,
        alpha=[param] * num_states,
        beta=[param] * num_states,
        cvxpy_settings={"solver": cp.MOSEK, "verbose": False, "eps": 1e-8},
    )

    vars = cvxpy_crossqsd_problem.variables()
    povm = [var.value for var in vars]
    prob_mat = calculate_prob_matrix_simple(
        prior_probs=[1 / num_states] * num_states,
        povm=povm,
        states=dense_mat,
    )
    p_succ = 0
    p_inc = 0
    for i in range(num_states):
        p_succ += prob_mat[i][i]
        p_inc += prob_mat[i][num_states]
    p_err = 1 - p_succ - p_inc
    print(f"p_succ = {p_succ:.5f}")
    print(f"p_err = {p_err:.5f}")
    print(f"p_err / p_succ = {np.format_float_scientific(p_err / p_succ, 5)}")
    # for line in prob_mat:
    #     print(line)
    print()
    return povm, p_succ


povm, theo_p_succ = test_crossQSD(params[16], params[16])


disturbance_states_dense_mat = [m.data for m in disturbance_states]
noise_prob_mat = calculate_prob_matrix_simple(
    prior_probs=[1 / num_states] * num_states,
    povm=povm,
    states=disturbance_states_dense_mat,
)
for item in noise_prob_mat:
    print(item)
noise_p = 0
for i in range(num_states):
    noise_p += noise_prob_mat[i][i]
print(noise_p)


# Test different depolarizing noise 0-25
print(np.format_float_scientific(params[16], 5))
for noise_param in params:
    if noise_param == params[8]:
        print(f"---")
    noisy_dense_mat = [
        (1 - noise_param) * dense_mat[_]
        + noise_param * disturbance_states[_].data
        for _ in range(num_states)
    ]
    prob_mat = calculate_prob_matrix_simple(
        prior_probs=[1 / num_states] * num_states,
        povm=povm,
        states=noisy_dense_mat,
    )
    p_succ = 0
    p_inc = 0
    for i in range(num_states):
        p_succ += prob_mat[i][i]
        p_inc += prob_mat[i][num_states]
    p_err = 1 - p_succ - p_inc
    print(rf"\lambda = {np.format_float_scientific(noise_param, 5)}")
    print(f"p_succ = {p_succ:.5f}")
    print(f"p_err = {p_err:.5f}")
    print(f"p_err / p_succ = {np.format_float_scientific(p_err / p_succ, 5)}")
    print(
        "worst-case p_succ",
        np.format_float_positional(
            (1 - noise_param) * theo_p_succ + noise_param * noise_p, 5
        ),
    )
    # for line in prob_mat:
    #     print(line)
    print()
