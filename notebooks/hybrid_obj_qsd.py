import numpy as np
import cvxpy as cp


import sys
sys.path.append("../")


from flow.solve_mix import *
from utils.handy_states import *

def hybrid_cases():
    # Case 1
    # Define the input states
    state_dict = simple_2(0.2, 0.5, 0.7)
    num_qubits = state_dict["num_qubits"]
    num_states = state_dict["num_states"]
    state_vec = state_dict["state_vec"]
    dense_mat = [DensityMatrix(_) for _ in state_vec]


    qsd_problem = ProblemSpec(
        num_qubits=num_qubits,
        num_states=num_states,
        case_id="0813_test",
        state_type="densitymatrix",
    )


    n = qsd_problem.num_states
    # Comment/uncomment to switch between different prior probabilities
    prior_prob = np.ones(n) * (1 / n)
    # prior_prob = [1/9, 3/9, 5/9]
    qsd_problem.prior_prob = prior_prob


    qsd_problem.set_states(
        state_type="statevector",
        states=state_vec,
        overwrite=True,
    )
    ideal_result = apply_Eldar(
        problem_spec=qsd_problem,
        prior_prob=prior_prob,
        cvxpy_settings={
            "solver": cp.MOSEK,
            "verbose": False,
            "eps": 1e-8,
        }
    )

    ideal_distrib = calculate_prob_matrix_simple(
        prior_probs=prior_prob,
        povm=ideal_result["povm"],
        states=dense_mat,
    )
    uqsd_opt_val = ideal_result["p_succ"]
    print(ideal_result["p_succ"])


    cvxpy_settings = {"solver": cp.MOSEK, "verbose": False, "eps": 1e-8}
    qsd_problem.set_states(
        state_type="densitymatrix",
        states=dense_mat,
        overwrite=True,
    )
    cvxpy_med_problem = med_problem(problem_spec=qsd_problem, prior_prob=prior_prob)
    cvxpy_med_problem.solve(**cvxpy_settings)
    med_opt_val = cvxpy_med_problem.solution.opt_val
    print(med_opt_val)


    noise_levels = [0.1 ** (6 - 0.5 * i) for i in range(11)]
    disturbance_states = [
        DensityMatrix(
            ProblemSpec.depolarizing_noise_channel(num_qubits=num_qubits)
        )
        for _ in range(num_states)
    ]


    def get_sqrt_dist(a, b):
        return np.sqrt(sum([abs(a[i] - b[i]) ** 2 for i in range(len(a))]))


    def run_qsd(
        qsd_problem: ProblemSpec,
        cvxpy_problem,
        sqrtd_result,
        psucc_result,
        eps=1e-8,
    ):
        cvxpy_settings = {"solver": cp.MOSEK, "verbose": False, "eps": eps}
        cvxpy_problem.solve(**cvxpy_settings)

        # print(cvxpy_problem.status)
        # print(cvxpy_problem.solution.opt_val)
        vars = cvxpy_problem.variables()

        fitqd_povm = [var.value for var in vars]
        # print(fitqd_povm)

        prob_mat = calculate_prob_matrix_simple(
            prior_probs=prior_prob,
            povm=fitqd_povm,
            states=qsd_problem.states,
        )
        # print(prob_mat)

        psucc = 0
        for i in range(qsd_problem.num_states):
            psucc += prob_mat[i][i]
        psucc_result.append(psucc)


        sqrt_dist = get_sqrt_dist(
            np.array(ideal_distrib).flatten(),
            np.array(prob_mat).flatten(),
        )
        sqrtd_result.append(sqrt_dist)
        #
        # print("alpha", np.array(calculate_errors(prob_mat)[0]))
        # print("beta ", max(calculate_errors(prob_mat)[1]))
        #
        # print()


    med_distrib = calculate_prob_matrix_simple(
        prior_probs=prior_prob,
        povm=[cvxpy_med_problem.variables()[i].value for i in range(len(cvxpy_med_problem.variables()))] + [np.zeros(shape=(qsd_problem.num_amps, qsd_problem.num_amps))],
        states=dense_mat,
    )
    med_l2 = sqrt_dist = get_sqrt_dist(
        np.array(med_distrib).flatten(),
        np.array(ideal_distrib).flatten(),
    )


    sqrtd_result = []
    psucc_result = []
    eps_list = [1e-8]
    lamdah_list = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    lamdah_list = [0.1 * _ for _ in range(10)]
    lamdah_list = [0.3 + 0.02 * _ for _ in range(11)]
    lamdah_list = [0.2 + 0.04 * _ for _ in range(11)]
    noise_levels = [0.1 ** (6 - 0.5 * i) for i in range(11)]
    noise_levels = [0.1 ** (6 - 0.25 * i) for i in range(25)]
    psucc_result_list = []
    sqrtd_result_list = []

    for param_a in lamdah_list:
        for eps in eps_list:
            sqrtd_result = []
            psucc_result = []
            for noise_level in noise_levels:
                combined_states = [
                    (1 - noise_level) * dense_mat[_]
                    + noise_level * disturbance_states[_].data
                    for _ in range(num_states)
                ]

                qsd_problem.set_states(
                    state_type="densitymatrix",
                    states=combined_states,
                    overwrite=True,
                )

                cvxpy_problem = hybrid_obj_problem(
                    problem_spec=qsd_problem,
                    ideal_distrib=ideal_distrib,
                    param_a=param_a,
                )

                run_qsd(
                    qsd_problem=qsd_problem,
                    cvxpy_problem=cvxpy_problem,
                    sqrtd_result=sqrtd_result,
                    psucc_result=psucc_result,
                )
            psucc_result_list.append(psucc_result)
            sqrtd_result_list.append(sqrtd_result)


    # OptUQSD under noise
    for noise_level in noise_levels:
        combined_states = [
            (1 - noise_level) * dense_mat[_]
            + noise_level * disturbance_states[_].data
            for _ in range(qsd_problem.num_states)
        ]
        # print(combined_states)
        prob_mat = calculate_prob_matrix_simple(
            prior_probs=prior_prob,
            povm=ideal_result["povm"],
            states=combined_states,
        )
        psucc = 0
        for i in range(qsd_problem.num_states):
            psucc += prob_mat[i][i]
        # for i in prob_mat:
        #     print(i)
        # print()
        #print(psucc)
        


    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 16,        # base size for everything
        "axes.labelsize": 18,   # x/y labels
        "axes.titlesize": 20,   # (if you use titles)
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,

        # make curves/markers visually match the larger fonts
        "lines.linewidth": 2.2,
        # "lines.markersize": 5.5,
        "lines.markersize": 7,
    })


    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.rcParams['image.cmap'] = 'rainbow'
    rainbow = cm.get_cmap('rainbow', len(sqrtd_result_list))  # Get 5 colors from rainbow colormap
    colors = [rainbow(i) for i in range(len(psucc_result_list))]

    plt.figure(figsize=(8, 6), dpi=900)

    for i, arr in enumerate(psucc_result_list):
        # if i >= 6:
        #     plt.plot(noise_levels, arr, ".-", label=rf'$1/w = {1/lamdah_list[i]:.1f}$')
        plt.plot(noise_levels, arr, ".-", color=colors[i], label=rf'$w = {lamdah_list[i]:.2f}$')

    plt.axhline(med_opt_val, linestyle="--", label='MED', alpha=0.5)
    plt.axhline(uqsd_opt_val, linestyle="dotted", label='UQSD', alpha=0.5)

    plt.xlabel(r'Noise level $\lambda$')
    plt.ylabel(r'$P_{succ}$')
    plt.xscale('log')

    # plt.title('3 almost-ortho 2-qubit states')
    # plt.title('2 non-orthogonal 1-qubit states')
    plt.legend(loc="upper left")
    plt.grid(True)
    # plt.savefig("figures/3_entangled_psucc.pdf", bbox_inches="tight")
    plt.savefig("results/3_entangled_psucc.png", bbox_inches="tight")
    plt.show()


    print(uqsd_opt_val)


    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    rainbow = colormaps['rainbow']
    # rainbow = cm.get_cmap('rainbow', len(sqrtd_result_list))  # Get 5 colors from rainbow colormap
    colors = [rainbow(i / (len(sqrtd_result_list) - 1)) for i in range(len(sqrtd_result_list))]

    plt.figure(figsize=(8, 6), dpi=900)

    for i, arr in enumerate(sqrtd_result_list):
        # if i >= 6:
        #     plt.plot(noise_levels, arr, ".-", label=rf'$w = {lamdah_list[i]:.2f}$')
        plt.plot(noise_levels, arr, ".-", color=colors[i], label=rf'$w = {lamdah_list[i]:.2f}$')

    plt.axhline(med_l2, linestyle="--", label='MED', alpha=0.5)
    plt.axhline(0, linestyle="dotted", label='UQSD', alpha=0.5)

    plt.xlabel(r'Noise level $\lambda$')
    plt.ylabel(r'$L_2$ distance')
    plt.xscale('log')

    # plt.title('3 almost-ortho 2-qubit states')
    # plt.title('2 non-orthogonal 1-qubit states')
    plt.legend(loc="upper left")
    plt.grid(True)
    # plt.savefig("figures/3_entangled_sqrtd.pdf", bbox_inches="tight")
    plt.savefig("results/3_entangled_sqrtd.png", bbox_inches="tight")
    plt.show()


    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=900)

    for i, arr in enumerate(psucc_result_list):
        if (i + 1) % 2:
            plt.plot(noise_levels, arr, ".-", label=rf'$a_{1} = 0.001, a_{2} = {lamdah_list[i]:.2f}$')

    plt.axhline(0.855, linestyle="--", label=rf'MED', alpha=0.5)
    plt.axhline(0.292, linestyle="dotted", label='UQSD', alpha=0.5)

    plt.xlabel(r'Noise level $\lambda$')
    plt.ylabel(r'$P_{succ}$')
    plt.xscale('log')
    # plt.title('Two 1-qubit states 45deg')
    plt.legend()
    plt.grid(True)
    plt.show()
    """

    import pandas as pd

    # Assuming psucc_result_list, noise_levels, and lamdah_list are defined
    # Create a DataFrame with noise_levels as index and each array in psucc_result_list as a column
    df = pd.DataFrame(
        {f'a_2={lamdah_list[i]:.2f}': arr for i, arr in enumerate(psucc_result_list)},
        index=noise_levels
    )

    # Save to CSV
    df.to_csv('psucc_results_3states.csv', index_label='noise_level')

    # Case 2
    num_qubits = 1
    num_states = 2

    state_vec = [
        Statevector([1, 0]),
        Statevector([1/np.sqrt(2), 1/np.sqrt(2)]),
    ]
    dense_mat = [DensityMatrix(_) for _ in state_vec]

    qsd_problem = ProblemSpec(
        num_qubits=num_qubits,
        num_states=num_states,
        case_id="0813_test",
        state_type="densitymatrix",
    )


    sqrtd_result = []
    psucc_result = []
    eps_list = [1e-8]
    # lamdah_list = [0.0001, 0.001, 0.01, 0.1, 1]
    # lamdah_list = [1, 10, 100, 1000, 10000, 100000]
    # lamdah_list = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # lamdah_list = [0.1 * _ for _ in range(1, 10)]
    lamdah_list = [0.3 + 0.02 * _ for _ in range(11)]
    noise_levels = [0.1 ** (6 - 0.5 * i) for i in range(11)]
    noise_levels = [0.1 ** (6 - 0.25 * i) for i in range(25)]
    psucc_result_list = []
    sqrtd_result_list = []
    param_rh = [0.0001 for i in range(num_states)]
    param_rv = [0.0001 for i in range(num_states + 1)]
    param_rh = [0.001 for i in range(num_states)]
    param_rv = [0.001 for i in range(num_states + 1)]

    for param_a in lamdah_list:
        for eps in eps_list:
            sqrtd_result = []
            psucc_result = []
            for noise_level in noise_levels:
                combined_states = [
                    (1 - noise_level) * dense_mat[_]
                    + noise_level * disturbance_states[_].data
                    for _ in range(num_states)
                ]

                qsd_problem.set_states(
                    state_type="densitymatrix",
                    states=combined_states,
                    overwrite=True,
                )

                cvxpy_problem = hybrid_obj_problem(
                    problem_spec=qsd_problem,
                    ideal_distrib=ideal_distrib,
                    param_a=param_a,
                )

                run_qsd(
                    qsd_problem=qsd_problem,
                    cvxpy_problem=cvxpy_problem,
                    sqrtd_result=sqrtd_result,
                    psucc_result=psucc_result,
                )
            psucc_result_list.append(psucc_result)
            sqrtd_result_list.append(sqrtd_result)


    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=900)
    plt.rcParams['image.cmap'] = 'rainbow'
    rainbow = cm.get_cmap('rainbow', len(sqrtd_result_list))  # Get 5 colors from rainbow colormap
    colors = [rainbow(i) for i in range(len(psucc_result_list))]

    for i, arr in enumerate(psucc_result_list):
        # if i >= 6:
            # plt.plot(noise_levels, arr, ".-", label=rf'$1/w = {1/lamdah_list[i]:.1f}$')
        plt.plot(noise_levels, arr, ".-", color=colors[i], label=rf'$w = {lamdah_list[i]:.2f}$')

    plt.axhline(med_opt_val, linestyle="--", label='MED', alpha=0.5)
    plt.axhline(uqsd_opt_val, linestyle="dotted", label='UQSD', alpha=0.5)

    plt.xlabel(r'Noise level $\lambda$')
    plt.ylabel(r'$P_{succ}$')
    plt.xscale('log')
    # plt.ylim(0.7, 1)

    # plt.title('3 2-qubit entangled states')
    # plt.title('2 1-qubit states 45deg')
    plt.legend(loc="center left")
    plt.grid(True)
    # plt.savefig("figures/2_nonortho_psucc.pdf", bbox_inches="tight")
    plt.savefig("results/2_nonortho_psucc.png", bbox_inches="tight")
    plt.show()


    print(psucc_result_list)


    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=900)

    for i, arr in enumerate(sqrtd_result_list):
        # if i >= 6:
        plt.plot(noise_levels, arr, ".-", color=colors[i], label=rf'$w = {lamdah_list[i]:.2f}$')
        # plt.plot(noise_levels, arr, ".-", label=rf'$1/w = {1/lamdah_list[i]:.1f}$')

    plt.axhline(med_l2, linestyle="--", label='MED', alpha=0.5)
    plt.axhline(0, linestyle="dotted", label='UQSD', alpha=0.5)

    plt.xlabel(r'Noise level $\lambda$')
    plt.ylabel(r'$L_2$ distance')
    plt.xscale('log')

    # plt.title('3 2-qubit entangled states')
    # plt.title('2 1-qubit states 45deg')
    # plt.ylim(-0.025, 0.15)
    plt.legend(loc="center left")
    plt.grid(True)
    # plt.savefig("figures/2_nonortho_sqrtd.pdf", bbox_inches="tight")
    plt.savefig("results/2_nonortho_sqrtd.png", bbox_inches="tight")
    plt.show()



