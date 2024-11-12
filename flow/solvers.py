from problem_spec import *
import numpy as np
import cvxpy as cp
# import cplex

def apply_Eldar(problem_spec: ProblemSpec, prior_prob=None, min_prob=0):
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
    I = np.eye(problem_spec.num_amps, dtype='complex128')
    # I = np.identity(problem_spec.num_amps)
    expr = I
    for i in range(n):
        expr = expr - p[i] * q[i]
    constraints.append(expr >> 0)  # Matrix inequality in CVXPY uses >>

    prob = cp.Problem(objective, constraints)
    # TODO logger.info(f"CVXPY settings {}")
    t1 = time.time()
    # result = prob.solve(solver=cp.SCS, eps=1e-20)
    result = prob.solve(solver=cp.SCS, verbose=False)  #, eps=1e-20)
    # result = prob.solve(solver=cp.CPLEX, verbose=True, eps=1e-20)
    t2 = time.time()
    logger.info(f"CVXPY returns {prob.status}")
    # if prob.status == 'optimal':
    #     pass
    # else:
    #     pass
    logger.info(f"Solution time (rounded) = {round((t2 - t1), 4)} seconds")
    logger.info(f"Result (rounded) = {result.round(4)}")
    sol = p.value
    logger.info(f"Solution (rounded) = {sol.round(4)}")

    # Obtain POVMs
    povm = []
    for i in range(n):
        if sol[i] <= 1e-4:
            logger.warning(f"sol[{i}] is zero or negative ({sol[i]} <= 1e-4), skip its operator")
            continue
        else:
            povm.append(np.sqrt(sol[i]) * Phi_tilde[i].conj())

    # TODO Remember the remaining operators
    return povm


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
    parser = ArgumentParser()
    parser.add_argument("-q", "--nqubits", default=2)
    parser.add_argument("-n", "--nstates", default=3)
    parser.add_argument("-s", "--seed", default=42)
    args = parser.parse_args()
    nq = int(args.nqubits)
    ns = int(args.nstates)
    seed = int(args.seed)
    case_id = f"q{nq}_n{ns}_s{seed}"
    logging.basicConfig(
        filename=f"solvers_{case_id}.log",
        filemode="a",
        format="{asctime} {levelname} {filename}:{lineno}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=logging.DEBUG,
        encoding="utf-8",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Start a new program")
    logger.info(f"nq = {nq}, ns = {ns}, seed = {seed}")
    tracemalloc.start()
    problem = ProblemSpec(
        num_qubits=nq, num_states=ns, case_id=case_id, state_type="statevector"
    )
    states = ProblemSpec.gen_states(
        num_qubits=nq,
        num_states=ns,
        seeds=get_random_seeds(ns, seed=seed),
        state_type="statevector",
    )
    problem.set_states(state_type="statevector", states=states)
    povm = apply_Eldar(problem_spec=problem)
    logger.info(f"Memory (current, peak, in bytes) = {tracemalloc.get_traced_memory()}")
    tracemalloc.stop()
    np.save(f"povm_{case_id}.npy", povm)
    logger.info(f"The POVM is saved to povm_{case_id}.npy")
    # TODO Remember the remaining operators
