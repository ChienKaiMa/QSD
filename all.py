import sys

sys.path.append("../")

import csv
import matplotlib.pyplot as plt

from flow.solve_mix import *
from utils.handy_states import *


def sweep_CrossQSD():
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

    cvxpy_settings = {
        # "solver": cp.MOSEK,
        "solver": cp.SCS,
        "verbose": False,
        # "eps": 1e-8,
        "eps": 1e-6,
    }
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

    ideal_prob_mat = calculate_prob_matrix_simple(
        prior_probs=[1 / num_states] * num_states,
        povm=ideal_povm,
        states=dense_mat,
    )
    ideal_p_succ = 0
    for i in range(num_states):
        ideal_p_succ += ideal_prob_mat[i][i]
    print(f"p_succ = {ideal_p_succ:.4f}")

    # Calculate the noise effect on the POVMs
    disturbance_states = [
        DensityMatrix(
            ProblemSpec.depolarizing_noise_channel(num_qubits=num_qubits)
        )
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
            cvxpy_settings={
                # "solver": cp.MOSEK,
                "solver": cp.SCS,
                "verbose": False,
                # "eps": 1e-8,
                "eps": 1e-6,
            },
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
        print(
            f"p_err / p_succ = {np.format_float_scientific(p_err / p_succ, 5)}"
        )
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

    lambda_list = []
    ratio_list = []
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
        p_err = 1.0 - p_succ - p_inc
        ratio = p_err / p_succ
        # print(rf"\lambda = {np.format_float_scientific(noise_param, 5)}")
        # print(
        #     f"p_err / p_succ = {np.format_float_scientific(ratio, 5)}"
        # )
        lambda_list.append(noise_param)
        ratio_list.append(ratio)

    # =======================
    # write CSV
    # =======================
    with open("results/crossqsd_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "p_err/p_succ"])
        for lam, r in zip(lambda_list, ratio_list):
            writer.writerow([lam, r])

    print("[INFO] wrote results/crossqsd_results.csv")

    # =======================
    # make plot
    # =======================
    plt.figure(figsize=(4.0, 3.0), dpi=300)

    plt.loglog(
        lambda_list,
        ratio_list,
        marker="o",
        linestyle="-",
        label="CrossQSD",
    )

    plt.xlabel(r"Noise Level $\lambda$")
    plt.ylabel(r"Error-to-Success Ratio $P_{\mathrm{err}}/P_{\mathrm{succ}}$")
    plt.grid(True, which="both", axis="both", linestyle=":", alpha=0.5)
    plt.legend(loc="best", fontsize=7)

    plt.tight_layout()
    plt.savefig("results/crossqsd_ratio.png", bbox_inches="tight")
    plt.close()

    print("[INFO] wrote results/crossqsd_ratio.png")

    return


def sweep_FitQSD():
    # ## Define the input states

    state_dict = simple_2(0.2, 0.5, 0.7)
    num_qubits = state_dict["num_qubits"]
    num_states = state_dict["num_states"]
    state_vec = state_dict["state_vec"]
    dense_mat = [DensityMatrix(_) for _ in state_vec]

    qsd_problem = ProblemSpec(
        num_qubits=num_qubits,
        num_states=num_states,
        case_id="0809_test",
        state_type="densitymatrix",
    )

    noise_levels = [0.1 ** (6 - i) for i in range(6)]
    disturbance_states = [
        DensityMatrix(
            ProblemSpec.depolarizing_noise_channel(num_qubits=num_qubits)
        )
        for _ in range(num_states)
    ]

    # ## Compute the noiseless distribution

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
            # "solver": cp.MOSEK,
            "solver": cp.SCS,
            "verbose": False,
            # "eps": 1e-8,
            "eps": 1e-6,
        },
    )

    ideal_distrib = calculate_prob_matrix_simple(
        prior_probs=prior_prob,
        povm=ideal_result["povm"],
        states=dense_mat,
    )
    print(ideal_result["p_succ"])

    # ## Use the POVM in noisy states

    def report_ideal(ideal_distrib):
        print("Ideal input states")
        print("POVM for ideal optimal UQSD")
        print(ideal_distrib)
        print()

    def dist_1(m):
        correct_results = [m[i][i] for i in range(len(m))]
        p_succ_ni = sum(correct_results)
        return np.concatenate((correct_results, np.array([1.0 - p_succ_ni])))

    def dist_2(m):
        measured_results = [
            sum([m[j][i] for j in range(len(m))]) for i in range(len(m))
        ]
        p_succ_ni = sum(measured_results)
        return np.concatenate((measured_results, np.array([1.0 - p_succ_ni])))

    def dist_3(m):
        """Yeah. No changes.
        However, we have to calculate the ideal distribution in the same way.
        """
        return np.array(m).flatten()

    def get_sqrt_dist(a, b):
        return np.sqrt(sum([abs(a[i] - b[i]) ** 2 for i in range(len(a))]))

    def report_noisy_states_with_ideal_povm(
        noise_level,
        ideal_distrib,
        qsd_problem: ProblemSpec,
        psucc_result,
    ):
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
        # m is prob_matrix
        m = calculate_prob_matrix_simple(
            qsd_problem.prior_prob, ideal_result["povm"], combined_states
        )
        # for i in m:
        #     print(i)
        noisy_distrib_with_ideal_povm = dist_3(m)
        print(f"Noisy input states with depolarizing noise = {noise_level:3g}")
        print("POVM for ideal optimal UQSD")
        print(noisy_distrib_with_ideal_povm)
        np.set_printoptions(precision=4)
        psucc = 0
        for i in range(qsd_problem.num_states):
            psucc += m[i][i]
        print(psucc)

        print("alpha", np.array(calculate_errors(m)[0]))
        print("beta ", np.array(calculate_errors(m)[1]))

        psucc_result.append(psucc)
        sqrt_dist = get_sqrt_dist(
            np.array(ideal_distrib).flatten(), noisy_distrib_with_ideal_povm
        )
        print(
            f"Root sum of squares {np.format_float_scientific(sqrt_dist, precision=4)}"
        )

    ## sqrtd_result = []
    ## psucc_result = []
    ## for noise_level in noise_levels:
    ##     report_noisy_states_with_ideal_povm(
    ##         noise_level=noise_level,
    ##         ideal_distrib=ideal_distrib,
    ##         qsd_problem=qsd_problem,
    ##         psucc_result=psucc_result,
    ##     )

    # ## New formulations

    def run_qsd(
        qsd_problem: ProblemSpec,
        cvxpy_problem,
        sqrtd_result,
        psucc_result,
        eps=1e-8,
    ):
        cvxpy_settings = {
            "solver": cp.SCS,
            "verbose": False,
            # "verbose": True,
            # "requires_grad": True,
            # "mkl": True,
            "eps": eps,
            # "acceleration_lookback": 10,
            # "warm_start": True,
        }
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

    methods = [meco_problem, min_l1_problem, min_ss_problem]
    eps = 1e-9
    sqrtd_results = dict()
    psucc_results = dict()
    for method in methods:
        print(method.__name__)
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
            # print("Noise level", np.format_float_scientific(noise_level, precision=4))
            cvxpy_problem = method(
                ideal_distrib=ideal_distrib,
                qsd_problem=qsd_problem,
                prior_prob=prior_prob,
            )
            run_qsd(
                qsd_problem=qsd_problem,
                cvxpy_problem=cvxpy_problem,
                sqrtd_result=sqrtd_result,
                psucc_result=psucc_result,
                eps=eps,
            )
        psucc_results[method.__name__] = psucc_result
        sqrtd_results[method.__name__] = sqrtd_result
    # --- 5. Write CSV ---
    with open("results/fitqsd_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # header
        header = (
            ["lambda"]
            + [f"psucc_{method.__name__}" for method in methods]
            + [f"L2_{method.__name__}" for method in methods]
        )
        writer.writerow(header)

        # rows
        for idx, lam in enumerate(noise_levels):
            row = [lam]
            # psucc columns
            row += [psucc_results[method.__name__][idx] for method in methods]
            # L2 columns
            row += [sqrtd_results[method.__name__][idx] for method in methods]
            writer.writerow(row)
    print("[INFO] wrote results/fitqsd_results.csv")

    # --- 6. Plot figure (left panel: P_succ, right panel: L2) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=300)

    # consistent styling: MinL1 red square, MinSS green dot, MECO blue triangle
    style_order = [
        ("MinL1", "red", "s"),
        ("MinSS", "green", "o"),
        ("MECO", "blue", "^"),
    ]
    name_dict = {
        "MinL1": min_l1_problem,
        "MinSS": min_ss_problem,
        "MECO": meco_problem,
    }
    # Left: P_succ vs λ (log x, linear y)
    ax0 = axes[0]
    ax0.set_xscale("log")
    ax0.set_xlim([1e-6, 1e-1])
    # y-limits are chosen to match your figure
    ax0.set_ylim([0.76, 0.82])
    ax0.grid(True, which="both", linestyle="-", alpha=0.3)
    for name, color, marker in style_order:
        ax0.plot(
            noise_levels,
            psucc_results[name_dict[name].__name__],
            marker=marker,
            color=color,
            linewidth=1.5,
            label=name,
        )

    ax0.set_xlabel(r"Noise Level $\lambda$")
    ax0.set_ylabel(r"Success Probability $P_{\mathrm{succ}}$")
    ax0.legend(loc="lower left", frameon=True)

    # Right: L2 vs λ (log x, log y)
    ax1 = axes[1]
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim([1e-6, 1e-1])
    ax1.set_ylim([1e-7, 5e-2])
    ax1.grid(True, which="both", linestyle="-", alpha=0.3)
    for name, color, marker in style_order:
        ax1.plot(
            noise_levels,
            sqrtd_results[name_dict[name].__name__],
            marker=marker,
            color=color,
            linewidth=1.5,
            label=name,
        )
    ax1.set_xlabel(r"Noise Level $\lambda$")
    ax1.set_ylabel(r"$L_2$ Distance")
    ax1.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig("results/fitqsd.png", bbox_inches="tight")
    print("[INFO] wrote results/fitqsd.png")
    return


import numpy as np
import cvxpy as cp


import sys

sys.path.append("../")


from flow.solve_mix import *
from utils.handy_states import *


def hybrid_case_1():
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
            # "solver": cp.MOSEK,
            "solver": cp.SCS,
            "verbose": False,
            # "eps": 1e-8,
            "eps": 1e-6,
        },
    )

    ideal_distrib = calculate_prob_matrix_simple(
        prior_probs=prior_prob,
        povm=ideal_result["povm"],
        states=dense_mat,
    )
    uqsd_opt_val = ideal_result["p_succ"]
    # print(ideal_result["p_succ"])

    cvxpy_settings = {
        # "solver": cp.MOSEK,
        "solver": cp.SCS,
        "verbose": False,
        # "eps": 1e-8,
        "eps": 1e-6,
    }
    qsd_problem.set_states(
        state_type="densitymatrix",
        states=dense_mat,
        overwrite=True,
    )
    cvxpy_med_problem = med_problem(
        problem_spec=qsd_problem, prior_prob=prior_prob
    )
    cvxpy_med_problem.solve(**cvxpy_settings)
    med_opt_val = cvxpy_med_problem.solution.opt_val
    # print(med_opt_val)

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
        eps=1e-6,
    ):
        cvxpy_settings = {"solver": cp.SCS, "verbose": False, "eps": eps}
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

    med_distrib = calculate_prob_matrix_simple(
        prior_probs=prior_prob,
        povm=[
            cvxpy_med_problem.variables()[i].value
            for i in range(len(cvxpy_med_problem.variables()))
        ]
        + [np.zeros(shape=(qsd_problem.num_amps, qsd_problem.num_amps))],
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

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 16,  # base size for everything
            "axes.labelsize": 18,  # x/y labels
            "axes.titlesize": 20,  # (if you use titles)
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
            # make curves/markers visually match the larger fonts
            "lines.linewidth": 2.2,
            # "lines.markersize": 5.5,
            "lines.markersize": 7,
        }
    )

    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    rainbow = colormaps["rainbow"]
    # rainbow = cm.get_cmap('rainbow', len(sqrtd_result_list))  # Get 5 colors from rainbow colormap
    colors = [
        rainbow(i / (len(sqrtd_result_list) - 1))
        for i in range(len(sqrtd_result_list))
    ]

    plt.figure(figsize=(8, 6), dpi=900)

    for i, arr in enumerate(psucc_result_list):
        # if i >= 6:
        #     plt.plot(noise_levels, arr, ".-", label=rf'$1/w = {1/lamdah_list[i]:.1f}$')
        plt.plot(
            noise_levels,
            arr,
            ".-",
            color=colors[i],
            label=rf"$w = {lamdah_list[i]:.2f}$",
        )

    plt.axhline(med_opt_val, linestyle="--", label="MED", alpha=0.5)
    plt.axhline(uqsd_opt_val, linestyle="dotted", label="UQSD", alpha=0.5)

    plt.xlabel(r"Noise level $\lambda$")
    plt.ylabel(r"$P_{succ}$")
    plt.xscale("log")

    # plt.title('3 almost-ortho 2-qubit states')
    # plt.title('2 non-orthogonal 1-qubit states')
    plt.legend(loc="upper left")
    plt.grid(True)
    # plt.savefig("figures/hybrid_3states_psucc.pdf", bbox_inches="tight")
    plt.savefig("results/hybrid_3states_psucc.png", bbox_inches="tight")
    print("[INFO] wrote results/hybrid_3states_psucc.png")

    # print(uqsd_opt_val)

    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    rainbow = colormaps["rainbow"]
    # rainbow = cm.get_cmap('rainbow', len(sqrtd_result_list))  # Get 5 colors from rainbow colormap
    colors = [
        rainbow(i / (len(sqrtd_result_list) - 1))
        for i in range(len(sqrtd_result_list))
    ]

    plt.figure(figsize=(8, 6), dpi=900)

    for i, arr in enumerate(sqrtd_result_list):
        # if i >= 6:
        #     plt.plot(noise_levels, arr, ".-", label=rf'$w = {lamdah_list[i]:.2f}$')
        plt.plot(
            noise_levels,
            arr,
            ".-",
            color=colors[i],
            label=rf"$w = {lamdah_list[i]:.2f}$",
        )

    plt.axhline(med_l2, linestyle="--", label="MED", alpha=0.5)
    plt.axhline(0, linestyle="dotted", label="UQSD", alpha=0.5)

    plt.xlabel(r"Noise level $\lambda$")
    plt.ylabel(r"$L_2$ distance")
    plt.xscale("log")

    # plt.title('3 almost-ortho 2-qubit states')
    # plt.title('2 non-orthogonal 1-qubit states')
    plt.legend(loc="upper left")
    plt.grid(True)
    # plt.savefig("figures/hybrid_3states_sqrtd.pdf", bbox_inches="tight")
    plt.savefig("results/hybrid_3states_sqrtd.png", bbox_inches="tight")
    print("[INFO] wrote results/hybrid_3states_sqrtd.png")

    import pandas as pd

    # Assuming psucc_result_list, noise_levels, and lamdah_list are defined
    # Create a DataFrame with noise_levels as index and each array in psucc_result_list as a column
    df = pd.DataFrame(
        {
            f"w={lamdah_list[i]:.2f}": arr
            for i, arr in enumerate(psucc_result_list)
        },
        index=noise_levels,
    )

    # Save to CSV
    df.to_csv(
        "results/hybrid_psucc_results_3states.csv", index_label="noise_level"
    )
    print("[INFO] wrote results/hybrid_psucc_results_3states.csv")

    df = pd.DataFrame(
        {
            f"w={lamdah_list[i]:.2f}": arr
            for i, arr in enumerate(sqrtd_result_list)
        },
        index=noise_levels,
    )

    # Save to CSV
    df.to_csv(
        "results/hybrid_sqrtd_results_3states.csv", index_label="noise_level"
    )
    print("[INFO] wrote results/hybrid_sqrtd_results_3states.csv")
    return


def hybrid_case_2():
    # Define the input states
    num_qubits = 1
    num_states = 2

    state_vec = [
        Statevector([1, 0]),
        Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)]),
    ]
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
            # "solver": cp.MOSEK,
            "solver": cp.SCS,
            "verbose": False,
            # "eps": 1e-8,
            "eps": 1e-6,
        },
    )

    ideal_distrib = calculate_prob_matrix_simple(
        prior_probs=prior_prob,
        povm=ideal_result["povm"],
        states=dense_mat,
    )
    uqsd_opt_val = ideal_result["p_succ"]
    # print(ideal_result["p_succ"])

    cvxpy_settings = {
        # "solver": cp.MOSEK,
        "solver": cp.SCS,
        "verbose": False,
        # "eps": 1e-8,
        "eps": 1e-6,
    }
    qsd_problem.set_states(
        state_type="densitymatrix",
        states=dense_mat,
        overwrite=True,
    )
    cvxpy_med_problem = med_problem(
        problem_spec=qsd_problem, prior_prob=prior_prob
    )
    cvxpy_med_problem.solve(**cvxpy_settings)
    med_opt_val = cvxpy_med_problem.solution.opt_val
    # print(med_opt_val)

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
        cvxpy_settings = {"solver": cp.SCS, "verbose": False, "eps": eps}
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
        povm=[
            cvxpy_med_problem.variables()[i].value
            for i in range(len(cvxpy_med_problem.variables()))
        ]
        + [np.zeros(shape=(qsd_problem.num_amps, qsd_problem.num_amps))],
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
    # lamdah_list = [0.2 + 0.04 * _ for _ in range(11)]
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

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 16,  # base size for everything
            "axes.labelsize": 18,  # x/y labels
            "axes.titlesize": 20,  # (if you use titles)
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
            # make curves/markers visually match the larger fonts
            "lines.linewidth": 2.2,
            # "lines.markersize": 5.5,
            "lines.markersize": 7,
        }
    )

    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    rainbow = colormaps["rainbow"]
    # rainbow = cm.get_cmap('rainbow', len(sqrtd_result_list))  # Get 5 colors from rainbow colormap
    colors = [
        rainbow(i / (len(sqrtd_result_list) - 1))
        for i in range(len(sqrtd_result_list))
    ]

    plt.figure(figsize=(8, 6), dpi=900)

    for i, arr in enumerate(psucc_result_list):
        # if i >= 6:
        #     plt.plot(noise_levels, arr, ".-", label=rf'$1/w = {1/lamdah_list[i]:.1f}$')
        plt.plot(
            noise_levels,
            arr,
            ".-",
            color=colors[i],
            label=rf"$w = {lamdah_list[i]:.2f}$",
        )

    plt.axhline(med_opt_val, linestyle="--", label="MED", alpha=0.5)
    plt.axhline(uqsd_opt_val, linestyle="dotted", label="UQSD", alpha=0.5)

    plt.xlabel(r"Noise level $\lambda$")
    plt.ylabel(r"$P_{succ}$")
    plt.xscale("log")

    # plt.title('3 almost-ortho 2-qubit states')
    # plt.title('2 non-orthogonal 1-qubit states')
    plt.legend(loc="upper left")
    plt.grid(True)
    # plt.savefig("figures/hybrid_2states_psucc.pdf", bbox_inches="tight")
    plt.savefig("results/hybrid_2states_psucc.png", bbox_inches="tight")

    print("[INFO] wrote results/hybrid_2states_psucc.png")

    # print(uqsd_opt_val)

    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    rainbow = colormaps["rainbow"]
    # rainbow = cm.get_cmap('rainbow', len(sqrtd_result_list))  # Get 5 colors from rainbow colormap
    colors = [
        rainbow(i / (len(sqrtd_result_list) - 1))
        for i in range(len(sqrtd_result_list))
    ]

    plt.figure(figsize=(8, 6), dpi=900)

    for i, arr in enumerate(sqrtd_result_list):
        # if i >= 6:
        #     plt.plot(noise_levels, arr, ".-", label=rf'$w = {lamdah_list[i]:.2f}$')
        plt.plot(
            noise_levels,
            arr,
            ".-",
            color=colors[i],
            label=rf"$w = {lamdah_list[i]:.2f}$",
        )

    plt.axhline(med_l2, linestyle="--", label="MED", alpha=0.5)
    plt.axhline(0, linestyle="dotted", label="UQSD", alpha=0.5)

    plt.xlabel(r"Noise level $\lambda$")
    plt.ylabel(r"$L_2$ distance")
    plt.xscale("log")

    # plt.title('3 almost-ortho 2-qubit states')
    # plt.title('2 non-orthogonal 1-qubit states')
    plt.legend(loc="upper left")
    plt.grid(True)
    # plt.savefig("figures/hybrid_2states_sqrtd.pdf", bbox_inches="tight")
    plt.savefig("results/hybrid_2states_sqrtd.png", bbox_inches="tight")
    print("[INFO] wrote results/hybrid_2states_sqrtd.png")

    import pandas as pd

    # Assuming psucc_result_list, noise_levels, and lamdah_list are defined
    # Create a DataFrame with noise_levels as index and each array in psucc_result_list as a column
    df = pd.DataFrame(
        {
            f"w={lamdah_list[i]:.2f}": arr
            for i, arr in enumerate(psucc_result_list)
        },
        index=noise_levels,
    )

    # Save to CSV
    df.to_csv(
        "results/hybrid_psucc_results_2states.csv", index_label="noise_level"
    )
    print("[INFO] wrote results/hybrid_psucc_results_2states.csv")

    df = pd.DataFrame(
        {
            f"w={lamdah_list[i]:.2f}": arr
            for i, arr in enumerate(sqrtd_result_list)
        },
        index=noise_levels,
    )

    # Save to CSV
    df.to_csv(
        "results/hybrid_sqrtd_results_2states.csv", index_label="noise_level"
    )
    print("[INFO] wrote results/hybrid_sqrtd_results_2states.csv")
    return


if __name__ == "__main__":
    # Fig.1
    # sweep_CrossQSD()
    # Fig.2
    # sweep_FitQSD()
    # Fig.3
    hybrid_case_1()
    # Fig.4
    hybrid_case_2()
