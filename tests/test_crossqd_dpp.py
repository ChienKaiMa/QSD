import sys

sys.path.append("../")
sys.path.append("./")
from flow.solve_mix import *
from qutip import coherent_dm


def gen_dummy_states():
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    alphas = [np.exp(angles[i] * 1j) for i in range(len(angles))]
    symm_states_1_dm_matrix = [
        coherent_dm(N=32, alpha=1 * alphas[i]).data.to_array()
        for i in range(len(alphas))
    ]
    return symm_states_1_dm_matrix


if __name__ == "__main__":
    problem = ProblemSpec(num_qubits=5, num_states=3, case_id="test")
    states = gen_dummy_states()
    problem.set_states(
        state_type="densitymatrix",
        states=states,
        overwrite=True,
    )
    cvxpy_problem = gen_crossQD_dpp_problem(problem_spec=problem)
    cvxpy_problem.param_dict["alpha"].value = [0.1, 0.1, 0.1]
    cvxpy_problem.param_dict["beta"].value = [0.1, 0.1, 0.1]
    result = cvxpy_problem.solve(
        solver=cp.SCS,
        verbose=True,
    )
