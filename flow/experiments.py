from plots import disable_excessive_logging, plot_psucc_noise_tol
from interface import SolverInterface
from problem_spec import ProblemSpec
from solve_mix import apply_crossQD
from temp.get_random_seeds import get_random_seeds
import json
import logging
import tracemalloc
from qiskit.quantum_info import DensityMatrix
import matplotlib.pyplot as plt
import numpy as np


def exp_rel_psucc_noise_tol(**kwargs):
    """
    The experiment to show the relations between the success probability,
    depolarizing noise level, and the tolerance.
    """
    # Goal: Parallelize the experiment

    # Simple tests or solver comparison?
    si = SolverInterface(__name__)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    disable_excessive_logging()

    noise_levels = kwargs["noise"]
    tol = kwargs["tol"]
    logger.info(f"noise ={noise_levels}")
    logger.info(f"tol ={tol}")

    case_id = f"{si.case_id}_mix_scan_noise"

    problem = ProblemSpec(
        num_qubits=si.nq, num_states=si.ns, case_id=case_id, state_type="densitymatrix"
    )

    noise_level = 0.3  # Just a placeholder
    dense_states, disturbance_states, combined_states = ProblemSpec.gen_noisy_states(
        num_qubits=si.nq,
        num_states=si.ns,
        seeds=get_random_seeds(si.ns, seed=si.state_seed),
        noise_level=noise_level,
        noise_rank=2,
    )

    # Goal: Separate data generation and plotting
    # TODO Save POVM

    # TODO Put these on the plots or file name
    # Parameters to draw the plots
    # alpha, beta, prior_prob, total_success_prob, noise_level

    # TODO
    # Rewrite this part for less memory usage

    # Collect results
    ptot_matrix = []
    psucc_matrix = []
    pinc_matrix = []
    for noise_level in noise_levels:
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

        ptot_list = []
        psucc_list = []
        pinc_list = []
        for t in tol:
            # Postprocessing
            result = apply_crossQD(
                problem_spec=problem,
                alpha=[t] * problem.num_states,
                beta=[t] * problem.num_states,
                noise_level=noise_level,
            )
            try:
                povm, total, p_d, p_inc = result
                np.save(f"results/povm_{case_id}_noise{noise_level}.npy", povm)
                logger.info(
                    f"The POVM is saved to povm_{case_id}_noise{noise_level}.npy"
                )
                ptot_list.append(total)
                psucc_list.append(p_d)
                pinc_list.append(p_inc)
            except:
                ptot_list.append(0)
                psucc_list.append(0)
                pinc_list.append(0)
                print("The solver did not return a tuple.")
        ptot_matrix.append(ptot_list)
        psucc_matrix.append(psucc_list)
        pinc_matrix.append(pinc_list)


    # Dump psucc pinc ptot
    np.save(f"results/ptot_{case_id}.npy", ptot_matrix)
    np.save(f"results/psucc_{case_id}.npy", psucc_matrix)
    np.save(f"results/pinc_{case_id}.npy", pinc_matrix)

    plot_psucc_noise_tol(tol, noise_levels, psucc_matrix, case_id)
    return


def exp_rel_psucc_noise_fptol_fntol(**kwargs):
    """
    size
    params
    """
    # Goal: Parallelize the experiment
    # Compare this snippet from flow/test_solve_mix_noise.py:
    # from solve_mix import *
    # from flow.plots import *
    # from flow.plots import plot_total_prob, plot_success_prob, plot_prob_noise_tol
    #
    #

    # Simple tests or solver comparison?
    import numpy as np

    si = SolverInterface(__name__)
    disable_excessive_logging()
    
    noise_levels = kwargs["noise"]
    fptol = kwargs["fptol"]
    fntol = kwargs["fntol"]
    logger.info(f"noise ={noise_levels}")
    logger.info(f"fptol ={fptol}")
    logger.info(f"fntol ={fntol}")
    
    noise_level = 0.3 # Just a placeholder

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
        # plot_success_prob(s_points, tag=f"noise_{noise_level}", noise_level=noise_level)
    tol = params

    # Dump probs
    np.save(f"prob_{case_id}.npy", prob)

    plot_psucc_noise_tol(tol, noise_levels, prob, case_id)
    # save_3dplot(points, fig)

    logger.info(f"Memory (current, peak, in bytes) = {tracemalloc.get_traced_memory()}")
    tracemalloc.stop()
    np.save(f"povm_{case_id}.npy", povm)
    logger.info(f"The POVM is saved to povm_{case_id}.npy")
    # TODO Remember the remaining operators


def run_small():
    from numpy import linspace

    with open("configs/noise_small.json", "r") as f:
        noise_params = json.load(f)
        noise = linspace(**noise_params)
        del noise_params

    with open("configs/fptol_small.json", "r") as f:
        fptol_params = json.load(f)
        fptol = linspace(**fptol_params)
        del fptol_params

    ## with open("configs/fntol_small.json", "r") as f:
    ##     fntol_params = json.load(f)
    ##     fntol = linspace(**fntol_params)
    ##     del fntol_params

    exp_rel_psucc_noise_tol(noise=noise, tol=fptol)


def run_large():
    from numpy import linspace

    with open("configs/noise_large.json", "r") as f:
        noise_params = json.load(f)
        noise = linspace(**noise_params)
        del noise_params

    with open("configs/fptol_large.json", "r") as f:
        fptol_params = json.load(f)
        fptol = linspace(**fptol_params)
        del fptol_params

    ## with open("configs/fntol_large.json", "r") as f:
    ##     fntol_params = json.load(f)
    ##     fntol = linspace(**fntol_params)
    ##     del fntol_params

    exp_rel_psucc_noise_tol(noise=noise, tol=fptol)
    ## exp_rel_psucc_noise_fptol_fntol()


if __name__ == "__main__":
    run_small()
    # run_large()
