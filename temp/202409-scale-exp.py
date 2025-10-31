from test_scipy_opt_debug import *

if __name__ == "__main__":
    np.set_printoptions(precision=4)

    # Experiment scalability
    for nq in range(3, 7):
        for ns in range(3, 7):
            print(f"(nq, ns) = ({nq}, {ns})")
            prob = NullSpaceSearchProblem(num_qubits=nq, num_states=ns)
            prob.set_states()
            prob.expand_solve()
            prob.find_null_spaces()
            prob.build_cons()
            num_samples = 0
            stop = False
            while not stop:
                num_samples += 1
                print(num_samples)
                prob.find_init()
                prob.solve(method="SLSQP")
                # prob.solve()
                # prob.x = np.array(x_unit)
                prob.x = prob.norm_vars(prob.x)
                stop = prob.verify()
            print(num_samples)
            print(prob.x)
    