import sys

sys.path.append("../")

import csv
import matplotlib.pyplot as plt

from flow.solve_mix import *
from utils.handy_states import *

# For Fig.7
from utils import *
from qiskit.circuit import QuantumCircuit
import sys
import time
import tracemalloc
import qiskit.transpiler
import qiskit.synthesis
import qiskit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector, set_statevector, SetDensityMatrix
from qiskit.quantum_info import Statevector

# Load states
from flow.problem_spec import *
from utils.handy_states import *

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
)


import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# For Table 3
from utils.utils import *
from qiskit.quantum_info import Operator
from typing import Optional, Dict, Any
import io
import re
from contextlib import redirect_stdout
import argparse
import pandas as pd
import os


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


def qc_sim():
    # ## Load the target circuit

    # nq = 2
    nq = 6
    ns = 3
    # seed = 1
    # case_id = f"q{nq}_n{ns}_s{seed}"
    case_id = f"q{nq}_n{ns}"
    # qasm_name = f"circuits/coherent/coh_{case_id}_no_backend_resynth.qasm"
    # TODO add resynth back
    qasm_name = f"coh_symm_q4_n3_optuqsd_reducedpovm_ccd_no_backend.qasm"

    qc = QuantumCircuit.from_qasm_file(qasm_name)
    print(qc.num_qubits)
    print(qc.count_ops())
    print(qc.depth())

    qc = transpile(qc, basis_gates=["rx", "ry", "rz", "cx"])
    print(int(qc.depth() * 0.05))
    qc_approx = transpile(
        circuits=qc,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config={
            "network_layout": "cart",
            "connectivity_type": "star",
            "depth": int(qc.depth() * 0.05),
        },
    )
    print("Approx")
    print(qc_approx.count_ops())
    print(qc_approx.depth())

    # Define handy simulation backends

    AerSimulator().available_methods()
    sv_backend = AerSimulator(method="statevector", seed_simulator=42)

    state_vec = coh_symm_small(num_qubits=6)["state_vec"]
    states = [Statevector(_) for _ in state_vec]

    # # Ideal sim verification

    # Our original circuit does not contain final measurements
    #
    # It is copied because measure_active creates new ClassicalRegisters every time
    #
    # https://docs.quantum.ibm.com/guides/measure-qubits

    # The state discrimination circuit without the initial state

    qc_tmp = qc_approx.copy()
    qc_tmp.save_statevector()
    qc_tmp.measure_active()

    # Assume the job always got done quickly
    result = sv_backend.run(qc_tmp).result()
    result.get_counts()
    result.get_statevector()
    result.get_statevector().probabilities().round(15)

    # ## Construct DUT
    # The circuit with one of the target initial states

    for i in range(3):
        # dut = QuantumCircuit(qc.num_qubits)
        dut = QuantumCircuit(nq)
        inst = dut.set_statevector(states[i]).instructions
        # inst = dut.set_statevector(states[i].expand(Statevector([1, 0]))).instructions
        # inst = dut.set_statevector(states[i].expand(Statevector([1, 0])).expand(Statevector([1, 0]))).instructions
        dut.append(inst[0], [_ for _ in range(nq)])
        # dut.save_statevector()
        dut.append(qc, [_ for _ in range(qc.num_qubits)])
        # dut.id(0)
        dut.save_statevector()
        dut.measure_active()
        dut = dut.decompose(reps=3)
        result = sv_backend.run(dut).result()
        # Show results
        print(result.get_counts())
        print(result.get_statevector().probabilities().round(5))
        # print(result.get_statevector().probabilities().round(10))

    # # Noisy sim extrapolation

    def test_dut(
        dut, param, target_states: list[str] | None = None, noise_model=None
    ):
        """Test DUT under the two-qubit depolarizing error"""
        noise_model = _two_qubit_depolarizing_noise_model(param)

        noise_result = sv_backend.run(dut, noise_model=noise_model).result()
        try:
            print(param)
            print("'001' count =", noise_result.get_counts()["001"])
        except:
            pass
        if target_states is None:
            return noise_result.get_statevector().probabilities()
        else:
            return [
                noise_result.get_statevector().probabilities()[state].item()
                for state in target_states
            ]

    def _two_qubit_depolarizing_noise_model(param):
        noise_model = NoiseModel()
        two_qubit_error = depolarizing_error(param, 2)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
        return noise_model

    # Check different params
    # Add references for better parameters?
    params = [
        1e-6,
        2e-6,
        5e-6,
        1e-5,
        2e-5,
        5e-5,
        1e-4,
        2e-4,
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
        1e-1,
    ]

    colors = list(TABLEAU_COLORS)

    result_0 = []
    result_1 = []
    result_2 = []
    for p in params:
        a, b, c = test_dut(dut, p, [0, 1, 2])
        # print(a, b)
        print(a)
        print(b)
        print(c)
        result_0.append(a)
        result_1.append(b)
        result_2.append(c)

    # Write the results to a CSV file
    with open("results/qc_sim.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["lambda", "result_0", "result_1", "result_2"]
        )  # Header row
        for i in range(len(params)):
            writer.writerow([params[i], result_0[i], result_1[i], result_2[i]])
    print("[INFO] wrote results/qc_sim.csv")
    # TODO Save data to .npy or .npz

    fig = plt.figure(dpi=600)
    # fig = plt.figure(dpi=200)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)
    # fig.set_figwidth(3)
    # fig.set_figheight(2.4)

    # plt.axvline(0.001, linestyle="--", alpha=0.3)
    # plt.arrow(0.0018, 0.25, -0.0005, 0, head_width=0.01, head_length=0.0003)
    # plt.text(0.0022, 0.24, "Sampling noise\n" + "starts to appear...")
    plt.axhline(
        test_dut(dut, 0, [0]), linestyle="--", alpha=0.3, color=colors[0]
    )
    plt.axhline(0, linestyle="--", alpha=0.3, color=colors[1])

    plt.xscale("log")
    plt.plot(params, result_0, ".", label="0")  # '"00" → "000"')
    plt.plot(params, result_1, "+", label="1")  # '"01" → "001"')
    plt.plot(params, result_2, "*", label="2")  # '"11" → "002"')

    # plt.xlabel(r"$\theta / \pi$" + f" ({case_id})")
    # plt.xlabel("Depolarizing noise parameter")
    # plt.ylabel("Probability amplitude")
    # plt.title(f"UQSD with noise in two-qubit gates ({case_id})")
    plt.legend(loc="center left")
    plt.grid(True, alpha=0.1)

    plt.savefig(
        fname=f"results/OptUQSD_ccd_{case_id}.png",
        bbox_inches="tight",
    )
    print(f"[INFO] wrote results/OptUQSD_ccd_{case_id}.png")
    plt.close()

    return


# Functions required for qc_opt


def get_unitary_circuit(qc) -> QuantumCircuit:
    # https://docs.quantum.ibm.com/guides/synthesize-unitary-operators#synthesize-unitary-operations

    U = Operator(qc)

    nq = qc.num_qubits
    unitary_circuit = QuantumCircuit(nq)
    unitary_circuit.unitary(U, range(nq))
    return unitary_circuit


def resynth_unitary(qc) -> QuantumCircuit:
    return get_unitary_circuit(qc).decompose(reps=3)


def resynth_unitary_approx(qc) -> QuantumCircuit:
    tmp_circuit = get_unitary_circuit(qc)

    # The approximation degree defaults to 1.0
    approx_circuit = transpile(
        tmp_circuit,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config={"seed": 42},
        approximation_degree=1.0,
        seed_transpiler=11,
    )

    return approx_circuit.decompose(reps=3)


def resynth_unitary_approx_max(qc) -> QuantumCircuit:
    tmp_circuit = get_unitary_circuit(qc)

    # It won't work without specifying aqc
    # It won't work without specifying basis_gates?
    approx_circuit = transpile(
        tmp_circuit,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config={"seed": 42},
        approximation_degree=0.97,
        basis_gates=["u", "cx"],
        seed_transpiler=11,
    )

    return approx_circuit.decompose(reps=3)


def resynth_aqc(qc, uni_synth_config=None) -> QuantumCircuit:
    tmp_circuit = get_unitary_circuit(qc)

    aqc_circuit = transpile(
        tmp_circuit,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config=uni_synth_config,
        approximation_degree=0,
        seed_transpiler=11,
    )

    return aqc_circuit.decompose(reps=3)


def try_synth(qc, case_id):
    # Show different results (depth and fidelity)
    resynth_circuit = resynth_unitary(qc)
    approx_circuit = resynth_unitary_approx(qc)
    approx_max_circuit = resynth_unitary_approx_max(qc)
    aqc_circuit_v0 = resynth_aqc(
        qc,
        {
            # "network_layout": "cart",
            "connectivity_type": "star",
            "depth": 10,
            "seed": 3,
        },
    )
    # TODO Approx again

    # Collect data
    data = []
    data.append(
        {
            "case_id": case_id,
            "circuit": "original",
            "ops": qc.count_ops(),
            "depth": qc.depth(),
            "fidelity": None,
        }
    )
    data.append(
        {
            "case_id": case_id,
            "circuit": "resynth",
            "ops": resynth_circuit.count_ops(),
            "depth": resynth_circuit.depth(),
            "fidelity": None,  # Will be computed below
        }
    )
    data.append(
        {
            "case_id": case_id,
            "circuit": "approx",
            "ops": approx_circuit.count_ops(),
            "depth": approx_circuit.depth(),
            "fidelity": None,
        }
    )
    data.append(
        {
            "case_id": case_id,
            "circuit": "approx_max",
            "ops": approx_max_circuit.count_ops(),
            "depth": approx_max_circuit.depth(),
            "fidelity": None,
        }
    )
    data.append(
        {
            "case_id": case_id,
            "circuit": "aqc_v0",
            "ops": aqc_circuit_v0.count_ops(),
            "depth": aqc_circuit_v0.depth(),
            "fidelity": None,
        }
    )

    from qiskit.quantum_info import process_fidelity, Operator

    # Two operators which differ only by phase
    op_a = Operator(qc)
    op_b = Operator(resynth_circuit)
    op_c = Operator(approx_circuit)
    op_c_1 = Operator(approx_max_circuit)
    op_d = Operator(aqc_circuit_v0)

    # Compute process fidelity
    F_resynth = process_fidelity(op_a, op_b)
    print("Process fidelity (resynth) =", F_resynth)
    F_approx = process_fidelity(op_a, op_c)
    print("Process fidelity (approx) =", F_approx)
    F_approx_max = process_fidelity(op_a, op_c_1)
    print("Process fidelity (approx_max) =", F_approx_max)
    F_aqc = process_fidelity(op_a, op_d)
    print("Process fidelity (aqc) =", F_aqc)

    # Update fidelities in data
    for row in data:
        if row["circuit"] == "resynth":
            row["fidelity"] = F_resynth
        elif row["circuit"] == "approx":
            row["fidelity"] = F_approx
        elif row["circuit"] == "approx_max":
            row["fidelity"] = F_approx_max
        elif row["circuit"] == "aqc_v0":
            row["fidelity"] = F_aqc

    # Save to CSV
    filename = "results/qc_opt_summary_table.csv"
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, mode="w", header=True, index=False)
    return


def qc_opt():
    # -----------------------------------------------------------------
    # Parse CLI argument --qubits
    # --qubits N means: run N ∈ {2,3,4,5}
    # and we will run all qubit sizes from 2 up to N inclusive.
    # Example:
    #   --qubits 4  => run [2,3,4]
    #   --qubits 5  => run [2,3,4,5]
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Run circuit synthesis experiments and emit summary_table.csv"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        required=True,
        choices=[2, 3, 4, 5],
        help="Maximum qubit count to evaluate (inclusive). "
        "Will run all sizes from 2..N. Allowed values: 2,3,4,5.",
    )
    args = parser.parse_args()

    # Build the list [2, 3, ..., args.qubits]
    qubit_list = list(range(2, args.qubits + 1))

    rows = []
    for i in qubit_list:
        try_synth(
            QuantumCircuit.from_qasm_file(
                f"circuits/symm/coh_symm_q{i}_n3_optuqsd_reducedpovm_ccd_no_backend.qasm"
            ),
            f"coh_q{i}_n3",
        )


if __name__ == "__main__":
    # Fig.1
    sweep_CrossQSD()
    # Fig.2
    sweep_FitQSD()
    # Fig.3
    hybrid_case_1()
    # Fig.4
    hybrid_case_2()
    # Fig.7
    qc_sim()
    # Table 3
    qc_opt()
