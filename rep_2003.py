import numpy as np
import cvxpy as cp


def reproduce_2003_Eldar_results():
    """Reproduce the example of Eldar's paper in 2003."""
    np.set_printoptions(precision=3)
    phi1 = np.multiply(1 / np.sqrt(3), [1, 1, 1])
    phi2 = np.multiply(1 / np.sqrt(2), [1, 1, 0])
    phi3 = np.multiply(1 / np.sqrt(2), [0, 1, 1])
    psi = np.transpose([phi1, phi2, phi3])
    print(psi)
    recip_psi = np.matmul(psi, np.linalg.inv(np.matmul(np.matrix.getH(psi), psi)))
    # Round nearzero value
    recip_psi = recip_psi.round(15)
    print(recip_psi)
    recip_psi = recip_psi.T
    q1 = np.outer(recip_psi[0], recip_psi[0])
    print(q1)

    n = 3
    prior_prob = np.multiply(-1 / 3, [1, 1, 1])
    # Measurement operators
    q = []
    for i in range(n):
        q.append(np.outer(recip_psi[i], recip_psi[i]).round(1))
    q = np.array(q)

    I = np.identity(n)
    p = cp.Variable(n)
    objective = cp.Minimize(1 + cp.sum(prior_prob @ p))
    constraints = [
        0 <= p[0],
        0 <= p[1],
        0 <= p[2],
        I - p[0] * q[0] - p[1] * q[1] - p[2] * q[2] >> 0,  # Matrix inequality uses >>
    ]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print("Result =", result.round(3))
    # An acceptable optimal solution
    sol = p.value.round(4)
    print("Solution =", sol)
    pi1 = I - sol[0] * q[0] - sol[1] * q[1] - sol[2] * q[2]  # Positive semidefinite
    print(pi1.round(5))
    # Wrong answer if we over postprocess the solution
    sol_overround = p.value.round(2)
    print("Overprocessed solution =", sol_overround)
    pi1_overround = (
        I - sol_overround[0] * q[0] - sol_overround[1] * q[1] - sol_overround[2] * q[2]
    )  # Not positive semidefinite
    print(pi1_overround.round(5))

    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    # print(constraints[0].dual_value)
    return


if __name__ == "__main__":
    reproduce_2003_Eldar_results()
    pass
