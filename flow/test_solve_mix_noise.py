from solve_mix import *
from flow.plots import *
from flow.plots import plot_total_prob, plot_psucc, plot_prob_noise_tol

if __name__ == "__main__":
    # TODO
    # Simple tests or solver comparison?
    si = SolverInterface(__name__)
    noise_level = 0.3
    # noise_levels = [0.1]
    # noise_levels = [0.01 * i for i in range(1, 3)]
    # noise_levels = [0.01 * i for i in range(1, 11)]
    noise_levels = [0.005 * i for i in range(1, 21)]

    case_id = f"{si.case_id}_mix_scan_noise"
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    tracemalloc.start()
    problem = ProblemSpec(
        num_qubits=si.nq, num_states=si.ns, case_id=case_id, state_type="densitymatrix"
    )

    dense_states, disturbance_states, combined_states = ProblemSpec.gen_noisy_states(
        num_qubits=si.nq,
        num_states=si.ns,
        seeds=get_random_seeds(si.ns, seed=si.state_seed),
        noise_level=noise_level,
        noise_rank=2,
    )

    # TODO
    # Generate basis states
    ## states = [1,2,3]
    ## states[0] = np.array([1, 0, 0, 0], dtype=np.complex128)
    ## states[1] = np.array([0, 1, 0, 0], dtype=np.complex128)
    ## states[2] = np.array([0, 0, 1, 0], dtype=np.complex128)
    ##
    ## # Convert states to DensityMatrix
    ## dense_states = [DensityMatrix(states[_]) for _ in range(problem.num_states)]
    ##

    # Ideal, but use the new formulation
    problem.set_states(state_type="densitymatrix", states=dense_states, overwrite=True)
    print("dense_states")
    print(problem.states[0].data)
    # TODO Save POVM

    # Noisy
    problem.set_states(
        state_type="densitymatrix", states=combined_states, overwrite=True
    )

    print("combined_states")
    print(problem.states[0].data)

    # TODO Different density matrices with noise

    # Separate data generation and plotting
    
    # Collect results
    x_axis = []
    y_axis = []
    z_axis = []
    # Parameters to draw the plots
    # alpha, beta, prior_prob, total_success_prob, noise_level
    # params = [0.1 ** (3 - 0.25 * i) for i in range(10)]
    # params = [0.5 - 0.04 * i for i in range(10)]
    params = [0.3 - 0.01 * i for i in range(20)]
    x_axis = params
    y_axis = params
    prob = []
    for i in noise_levels:
        t_points = []
        s_points = []
        noise_level = i
        combined_states = [
            (1 - noise_level) * dense_states[_].data
            + noise_level * disturbance_states[_].data
            for _ in range(problem.num_states)
        ]

        combined_states = [
            DensityMatrix(combined_states[_]) for _ in range(problem.num_states)
        ]
        problem.set_states(
            state_type="densitymatrix", states=combined_states, overwrite=True
        )

        prob_list = []
        for a in params:
            for b in params:
                # Postprocessing
                result = apply_crossQD(
                    problem_spec=problem,
                    alpha=[a] * 3,
                    beta=[b] * 3,
                    noise_level=noise_level,
                )
                try:
                    povm, total, p_d, inc_prob = result
                    ## z_axis.append(p_d)
                    ## points.append((a, b, p_d))
                    # z_axis.append(total)
                    t_points.append((a, b, total))
                    s_points.append((a, b, p_d))
                    if a == b:
                        prob_list.append(p_d)
                except:
                    # z_axis.append(0)
                    t_points.append((a, b, 0))
                    s_points.append((a, b, 0))
                    if a == b:
                        prob_list.append(0)
                    print("The solver did not return a tuple.")
                print()
        prob.append(prob_list)

        # TODO draw a plot of total success probability vs. threshold
        # TODO draw with prior probabilities
        fig = plt.figure(dpi=900)
        fig.set_figwidth(6)
        fig.set_figheight(4.8)
        fig.set_size_inches(8, 5)
        # z_axis = np.array(z_axis).reshape(5, 5)
        # z_axis = np.array(z_axis).reshape(10, 10)
        # save_3dplot(x_axis, y_axis, z_axis, fig)

        # Dump t_points and s_points
        np.save(f"t_points_{noise_level}.npy", t_points)
        np.save(f"s_points_{noise_level}.npy", s_points)

        # plot_total_prob(t_points, tag=f"noise_{noise_level}", noise_level=noise_level)
        # plot_psucc(s_points, tag=f"noise_{noise_level}", noise_level=noise_level)
    tol = params

    # Dump probs
    np.save(f"prob_{case_id}.npy", prob)
    
    plot_prob_noise_tol(tol, noise_levels, prob, case_id)
        # save_3dplot(points, fig)

    logger.info(f"Memory (current, peak, in bytes) = {tracemalloc.get_traced_memory()}")
    tracemalloc.stop()
    np.save(f"povm_{case_id}.npy", povm)
    logger.info(f"The POVM is saved to povm_{case_id}.npy")
    # TODO Remember the remaining operators
